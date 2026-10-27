"""Collector and agreement statistics for the twenty-expert R1/R2 campaign.

The returned forms are flattened PDF questionnaires (no form fields).  Text is
extracted with the page layout preserved (``pdftotext -layout``; pypdf layout
mode as a fallback) and each answer is located by the label printed on the
form, never by position, so threshold rotations cannot shift values.

Literal answers are kept verbatim; ``numeric`` converts only plain numbers
(optionally followed by a minute unit).  Approximate values ("circa 70"),
"non so", "nessuno", ranges and blanks are missing numerically.  No answer is
repaired, imputed or re-ordered.
"""
from __future__ import annotations

import hashlib
import math
import re
import shutil
import statistics as st
import subprocess
import unicodedata
from abc import ABC, abstractmethod
from datetime import date as _date
from pathlib import Path

import numpy as np

SCENARIOS = ['s1', 's2', 's3', 's4']
ENDPOINTS = ['b', 'q080', 'q060', 'q030', 'q010']
LEVELS = {'0,80': 'q080', '0,60': 'q060', '0,30': 'q030', '0,10': 'q010'}
SCENARIO_TITLES = {'s1': 'S1 Spostamento urbano', 's2': 'S2 Spostamento extraurbano',
                   's3': 'S3 Consegna con accesso', 's4': 'S4 Servizio con attesa'}
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261014


# --------------------------------------------------------------------------- text helpers
def normal(text) -> str:
    """Lower-case, accent-free, whitespace-collapsed literal."""
    if text is None:
        return ''
    t = unicodedata.normalize('NFKD', str(text)).encode('ascii', 'ignore').decode()
    return re.sub(r'\s+', ' ', t).strip().lower()


_NUMBER = re.compile(r'^(\d+(?:[.,]\d+)?)\s*(?:min|minuti|minuto|m)?\.?$')


def numeric(text):
    """Plain number (comma or point decimal, optional minute unit) or None."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    m = _NUMBER.match(normal(text))
    return float(m.group(1).replace(',', '.')) if m else None


def date(text):
    """Session date written as DD/MM/YYYY, DD MM YYYY, DD-MM-YYYY or DD.MM.YYYY."""
    m = re.match(r'^(\d{1,2})[\s/.\-]+(\d{1,2})[\s/.\-]+(\d{4})$', normal(text))
    if not m:
        return None
    try:
        return _date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    except ValueError:
        return None


# --------------------------------------------------------------------------- extraction
class TextExtractor(ABC):
    @abstractmethod
    def lines(self, path: Path) -> list[str]: ...


class PdftotextExtractor(TextExtractor):
    def lines(self, path):
        out = subprocess.run(['pdftotext', '-layout', str(path), '-'], check=True,
                             capture_output=True, text=True).stdout
        return out.splitlines()


class PypdfExtractor(TextExtractor):
    def lines(self, path):
        from pypdf import PdfReader
        return '\n'.join(p.extract_text(extraction_mode='layout')
                         for p in PdfReader(str(path)).pages).splitlines()


def default_extractor() -> TextExtractor:
    return PdftotextExtractor() if shutil.which('pdftotext') else PypdfExtractor()


# --------------------------------------------------------------------------- one form
class ReturnedForm:
    """One returned R1 or R2 questionnaire, parsed by printed labels."""

    QA_ITEMS = [  # (field, label prefix on the form)
        ('consent', 'acconsento volontariamente'),
        ('eligibility', 'sono maggiorenne'),
        ('area', 'area prevalente'),
        ('experience', 'esperienza meno di 2 anni'),
        ('prior_pilot', 'ho gia compilato il precedente pilot'),
        ('final_c1', 'un superamento con possibilita 0,60 significa'),
        ('final_c2', 'se la possibilita massima ammessa'),
        ('ceiling030', 'un piano ha una cella'),
        ('session_date', 'data gg mm aaaa'),
        ('completion_minutes', 'minuti impiegati'),
        ('practicality', 'praticabilita da 1'),
        ('consulted_r1', 'ho consultato il primo questionario'),
        ('memory', 'ricordavo le risposte'),
        ('new_information', 'fra le sessioni ho acquisito'),
    ]
    VALUE_COLUMN = 70  # answers are typeset in the right-hand column

    def __init__(self, path: Path, extractor: TextExtractor):
        self.path = Path(path)
        self.raw = extractor.lines(self.path)
        self.lines = [normal(x) for x in self.raw]
        self.errors: list[str] = []

    # right-hand cell of a two-column row, or None
    def _right(self, raw: str):
        m = re.match(r'^(.*?)(\s{2,})(\S.*)$', raw.rstrip())
        if not m:
            return None
        start = len(m.group(1)) + len(m.group(2))
        if start < self.VALUE_COLUMN or (m.group(1).strip() == '' and start < self.VALUE_COLUMN):
            return None
        return m.group(3).strip()

    def _qa(self, prefix: str, starts: list[int]):
        idx = [i for i, l in enumerate(self.lines) if l.startswith(prefix)]
        if len(idx) != 1:
            self.errors.append(f'label not unique: {prefix}')
            return ''
        i = idx[0]
        stop = min([j for j in starts if j > i] + [i + 4])
        for j in range(i, stop):
            v = self._right(self.raw[j])
            if v is not None:
                return v
        return ''

    def parse(self) -> dict:
        head = next((l for l in self.lines if l.startswith('codice ')), '')
        m = re.match(r'codice (e\d{2}) sessione (r[12])', head)
        if not m:
            raise ValueError(f'{self.path.name}: missing code/session header')
        row = dict(code=m.group(1).upper(), session=m.group(2).upper(), source_file=self.path.name, form_anomaly='',
                   source_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())
        stem = self.path.stem.upper()
        if stem != f"{row['code']}_{row['session']}":
            self.errors.append(f'file name {stem} differs from header')
        starts = sorted(i for i, l in enumerate(self.lines)
                        if any(l.startswith(p) for _, p in self.QA_ITEMS))
        for field, prefix in self.QA_ITEMS:
            if field == 'session_date' and sum(l.startswith(prefix) for l in self.lines) == 0:
                row[field] = self._overwritten_date_row(row)
                continue
            row[field] = self._qa(prefix, starts)
        self._scenarios(row)
        self._limits(row)
        row['parse_errors'] = '; '.join(self.errors)
        return row

    def _overwritten_date_row(self, row):
        """Date row whose printed label was overwritten by the respondent.

        The row is the first one of the 'Informazioni sulla compilazione' table;
        its answer cell is read as for any other row and the event is logged.
        """
        i0 = next(i for i, l in enumerate(self.lines) if l.startswith('informazioni sulla compilazione'))
        for j in range(i0 + 1, i0 + 6):
            if self.lines[j].startswith('informazione '):
                for k in range(j + 1, j + 4):
                    v = self._right(self.raw[k])
                    if v is not None:
                        row['form_anomaly'] = 'date label cell overwritten; answer cell of the date row read'
                        return v
        self.errors.append('session date row not found')
        return ''

    def _scenarios(self, row):
        order = []
        for s, title in SCENARIO_TITLES.items():
            idx = [i for i, l in enumerate(self.lines) if l.startswith(normal(title))]
            if len(idx) != 1:
                self.errors.append(f'scenario block {s} not unique'); continue
            order.append((idx[0], s))
        order.sort()
        row['scenario_order'] = ''.join(s[1] for _, s in order)
        bounds = [i for i, _ in order] + [next(i for i, l in enumerate(self.lines)
                                               if l.startswith('limiti del possibile'))]
        for (i0, s), i1 in zip(order, bounds[1:]):
            levels = []
            for j in range(i0, i1):
                l, raw = self.lines[j], self.raw[j]
                if l.startswith('b durata pienamente'):
                    row[s + '_b'] = self._right(raw) or ''
                elif l.startswith('durata oltre cui'):
                    lv = re.search(r'al massimo (0,\d0)', l)
                    key = LEVELS[lv.group(1)]; levels.append(key[1:])
                    row[s + '_' + key] = self._right(raw) or ''
                elif l.startswith('fiducia nel giudizio'):
                    row[s + '_confidence'] = self._right(raw) or ''
            row[s + '_level_order'] = '-'.join(levels)
            for e in ENDPOINTS + ['confidence']:
                if s + '_' + e not in row:
                    self.errors.append(f'{s}_{e} missing'); row[s + '_' + e] = ''

    def _limits(self, row):
        i0 = next(i for i, l in enumerate(self.lines) if l.startswith('limiti del possibile'))
        found = set()
        for raw in self.raw[i0:i0 + 25]:
            m = re.match(r'^\s*S([1-4])\s{3,}(.+?)\s{3,}(.+?)\s*$', raw)
            if m:
                s = 's' + m.group(1)
                row[s + '_a'], row[s + '_c'] = m.group(2).strip(), m.group(3).strip()
                found.add(s)
        for s in SCENARIOS:
            if s not in found:
                self.errors.append(f'{s} support row missing'); row[s + '_a'] = row[s + '_c'] = ''


class FormCollection:
    """All returned forms of the two session folders."""

    def __init__(self, folders, extractor: TextExtractor | None = None):
        self.folders = [Path(f) for f in folders]
        self.extractor = extractor or default_extractor()

    def collect(self):
        rows, errors = [], []
        for folder in self.folders:
            for path in sorted(folder.glob('E*_R*.pdf')):
                form = ReturnedForm(path, self.extractor)
                try:
                    row = form.parse()
                except Exception as exc:  # report, never substitute
                    errors.append(f'{path.name}: {exc}'); continue
                if row['parse_errors']:
                    errors.append(f"{path.name}: {row['parse_errors']}")
                rows.append(row)
        keys = [(r['code'], r['session']) for r in rows]
        if len(set(keys)) != len(keys):
            errors.append('duplicate code/session')
        return rows, errors


def collect(folders):
    return FormCollection(folders).collect()


# --------------------------------------------------------------------------- agreement
class AgreementEstimator:
    """ICC(A,1), Bland-Altman limits and paired-expert percentile bootstrap."""

    def __init__(self, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
        self.replicates, self.seed = replicates, seed

    @staticmethod
    def icc_a1(x: np.ndarray) -> float:
        n, k = x.shape
        row, col, grand = x.mean(axis=1), x.mean(axis=0), x.mean()
        ms_r = k * np.square(row - grand).sum() / (n - 1)
        ms_c = n * np.square(col - grand).sum() / (k - 1)
        resid = x - row[:, None] - col[None, :] + grand
        ms_e = np.square(resid).sum() / ((n - 1) * (k - 1))
        den = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
        return float((ms_r - ms_e) / den) if den > 0 else float('nan')

    @staticmethod
    def loa(d):
        sd = d.std(ddof=1)
        return [float(d.mean() - 1.96 * sd), float(d.mean() + 1.96 * sd)]

    @staticmethod
    def _pct(v):
        v = np.sort(np.asarray(v))
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    def estimate(self, pairs):
        x = np.array(pairs, dtype=float)
        n = len(x)
        if n < 3:
            return dict(n=n)
        d = x[:, 1] - x[:, 0]
        rng = np.random.default_rng(self.seed)
        icc, md, lo, hi = [], [], [], []
        for _ in range(self.replicates):
            b = x[rng.integers(0, n, n)]
            v = self.icc_a1(b)
            if math.isfinite(v):
                icc.append(v)
            db = b[:, 1] - b[:, 0]
            md.append(db.mean())
            if db.std(ddof=1) >= 0:
                l = self.loa(db); lo.append(l[0]); hi.append(l[1])
        out = dict(n=n, icc_A1=self.icc_a1(x), icc_bootstrap_95=self._pct(icc),
                   defined_bootstrap_replicates=len(icc),
                   mean_difference_R2_minus_R1=float(d.mean()), sd_difference=float(d.std(ddof=1)),
                   limits_agreement_95=self.loa(d),
                   mean_difference_bootstrap_95=self._pct(md),
                   limits_agreement_bootstrap_95=[self._pct(lo), self._pct(hi)],
                   identical_pairs=int(np.sum(np.abs(d) < 1e-12)),
                   median_absolute_difference=float(np.median(np.abs(d))))
        if np.all(x > 0):
            lr = np.log(x[:, 1] / x[:, 0])
            out['multiplicative_agreement'] = dict(
                geometric_ratio_R2_R1=float(np.exp(lr.mean())),
                limits_95=[float(np.exp(v)) for v in self.loa(lr)])
        return out


def agreement(pairs):
    return AgreementEstimator().estimate(pairs)
