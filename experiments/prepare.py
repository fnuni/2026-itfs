"""Regenerate the instance files and the counted-evaluation allowances.

    python3 experiments/prepare.py --workers 2

Writes data/instances.json, data/instances_small.json and
data/allowances.json.  The allowance of an instance is shape-independent: the
same count is used for every membership convention and panel reading of the
same coordinates and service times.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import common
from instances_io import DESIGN, SMALL_DESIGN, DATA_DIR, CanonicalInstances

def _job(args):
    family, seed, rho = args
    inst = common.instances(rho).build(family, seed, "tfn")
    return common.AllowanceBook.key(family, seed, "tfn", rho), \
        common.AllowanceBook.compute(inst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    # data/instances.json is the canonical benchmark file and is never
    # overwritten; the small-family file is generated once.
    import os
    small = os.path.join(DATA_DIR, "instances_small.json")
    if not os.path.exists(small):
        CanonicalInstances(use_files=False).write(small, design=SMALL_DESIGN)
    book = common.AllowanceBook()
    jobs = [(f, s, 0.85) for f in list(DESIGN) + list(SMALL_DESIGN) for s in common.SEEDS]
    jobs = [j for j in jobs if common.AllowanceBook.key(j[0], j[1], "tfn", j[2]) not in book.book]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for k, v in ex.map(_job, jobs, chunksize=1):
            book.book[k] = v
    book.save()
    print(f"{len(book.book)} allowances -> {book.path}")


if __name__ == "__main__":
    main()
