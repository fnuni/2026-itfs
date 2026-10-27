"""Complete Amazon apply-sample route scaffolds with explicit scenario TFNs.

The TFN supports vary per directed arc and service stop. They are deliberately
not fitted uncertainty estimates. Only the modal matrix/service times come
from the public operational dataset; see data/amazon_v4/manifest.json.
"""
from __future__ import annotations
import json
from pathlib import Path
from fuzzy_core import TriangularFN
from problem import Instance, MembershipFactory

class ExternalMatrixInstance(Instance):
    def __init__(self, rec):
        self.n, self.K, self.P, self.L = rec['n'], rec['K'], rec['P'], rec['L']
        self.coords, self.service_modal = rec['coordinates'], rec['service_minutes']
        self.factory, self.cell_evals = MembershipFactory('tfn'), 0
        self.speed, self._dist, self.meta = 1.0, rec['travel_minutes'], rec
        zones = rec['zones']; counts = rec['package_counts']
        def travel(i,j):
            m = self._dist[i][j]
            if i == j: return TriangularFN(0,0,0)
            left = .10+.05*((i+2*j)%3)
            cross = bool(zones[i] and zones[j] and zones[i] != zones[j])
            right = .20+.10*((3*i+j)%4)+.10*cross
            return TriangularFN(m*(1-left), m, m*(1+right))
        self._travel = [[travel(i,j) for j in range(self.n+1)] for i in range(self.n+1)]
        self._service = [TriangularFN(0,0,0)] + [
            TriangularFN(m*(1-(.10+.025*(counts[i]%3))),m,
                         m*(1+.20+.05*min(counts[i],5)))
            for i,m in enumerate(self.service_modal[1:],1)]
        self.tfn_travel = [[(f.a,f.b,f.c) for f in row] for row in self._travel]
        self.tfn_service = [(f.a,f.b,f.c) for f in self._service]

    def route_endpoints(self, seq):
        if not seq: return (0.,0.,0.)
        a=b=c=0.; prev=0
        for node in seq:
            ta,tb,tc = self.tfn_travel[prev][node]
            sa,sb,sc = self.tfn_service[node]
            a+=ta+sa; b+=tb+sb; c+=tc+sc; prev=node
        ta,tb,tc = self.tfn_travel[prev][0]
        return a+ta,b+tb,c+tc

    def cell_duration(self, seq):
        self.cell_evals += 1
        a,b,c = self.route_endpoints(seq)
        return TriangularFN(a,b,c),b

def load_external_records():
    path=Path(__file__).resolve().parents[1]/'data'/'amazon_v4'/'instances.json'
    return json.loads(path.read_text())['records']
