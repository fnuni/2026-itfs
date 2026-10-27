"""Wall-clock scalarized CC-LNS with exact per-arc TFN endpoint deltas.

Every directed insertion is evaluated when all_positions=True. The modal-only
restriction is a separately labeled ablation; with heterogeneous memberships
it is not an exact dominance rule. Modal 2-opt is disabled for this experiment:
a least-modal-cost shortcut need not minimize TFN risk on directed legs.
The returned incumbent is feasible-first and is never worse in J than its
initial seed when that seed is feasible. A candidate unfinished at the deadline
is discarded. Preprocessing/model construction and seed search are timed by
the experiment driver; the deadline is shared across both hybrid stages.
"""
from __future__ import annotations
import math
import time
from aggregators import FFF_REPORT, max_risk
from cc_lns import CCLNS
from fuzzy_core import TriangularFN
from problem import Solution
from solver import _State

class _Deadline(Exception): pass

class EndpointCCLNS(CCLNS):
    def __init__(self, inst, *, alpha, seed, workload_scale, deadline,
                 initial_cells=None, all_positions=True, max_iterations=None):
        super().__init__(inst, FFF_REPORT, alpha=alpha, seed=seed,
                         workload_scale=workload_scale, use_controller=False,
                         polish=False, destroy_frac=(.10,.25))
        self.deadline = deadline
        self.initial_cells = initial_cells
        self.all_positions = all_positions
        self.max_iterations = max_iterations          # None: wall-clock deadline only
        self.endpoint_candidates = 0

    def initial(self):
        if self.initial_cells is None: return super().initial()
        sol=Solution(self.inst.K,self.inst.P)
        sol.cells=[[list(c) for c in row] for row in self.initial_cells]
        return sol

    def _rank(self, st):
        return (st.violation>1e-9,
                st.violation if st.violation>1e-9 else 0.,
                self._scalar(st.R,st.makespan,0.))

    def _guided_repair(self, st, removed):
        inst=self.inst; self.rng.shuffle(removed)
        endpoints={(k,h):inst.route_endpoints(st.sol.cells[k][h])
                   for k in range(inst.K) for h in range(inst.P)}
        for t in removed:
            best_val=float('inf'); best=None; best_triple=None; best_end=None
            for k in range(inst.K):
                for h in range(inst.P):
                    c=st.sol.cells[k][h]; old=endpoints[k,h]
                    base=(st.risk[k][h],st.modal[k][h],st.opt[k][h])
                    positions=range(len(c)+1)
                    if not self.all_positions:
                        positions=[min(positions,key=lambda p:
                            inst.modal_travel(c[p-1] if p else 0,t)
                            +inst.modal_travel(t,c[p] if p<len(c) else 0)
                            -inst.modal_travel(c[p-1] if p else 0,c[p] if p<len(c) else 0))]
                    for pos in positions:
                        if time.perf_counter()>=self.deadline: raise _Deadline
                        prev=c[pos-1] if pos else 0; nxt=c[pos] if pos<len(c) else 0
                        end=tuple(old[q]+inst.tfn_travel[prev][t][q]
                                  +inst.tfn_travel[t][nxt][q]-inst.tfn_travel[prev][nxt][q]
                                  +inst.tfn_service[t][q] for q in range(3))
                        r=TriangularFN(*end).overtime_risk(inst.L)
                        inst.cell_evals+=1; self.endpoint_candidates+=1
                        triple=(r,end[1],end[0])
                        st.risk[k][h],st.modal[k][h],st.opt[k][h]=triple
                        val=self._scalar(st.risk,st.makespan,st.violation)
                        st.risk[k][h],st.modal[k][h],st.opt[k][h]=base
                        if val<best_val:
                            best_val,best,best_triple,best_end=val,(k,h,pos),triple,end
            k,h,pos=best; st.sol.cells[k][h].insert(pos,t)
            st.risk[k][h],st.modal[k][h],st.opt[k][h]=best_triple
            endpoints[k,h]=best_end
        return st

    def run(self):
        st=_State(self.inst,self.initial()); self._lambda_scale=self.workload_scale
        cur_val=self._scalar(st.R,st.makespan,st.violation)
        best,best_rank=st.clone(),self._rank(st)
        temp=self.t0; it=0; since=0
        while time.perf_counter()<self.deadline and (
                self.max_iterations is None or it<self.max_iterations):
            out,q,temp_now=self._decide(st,since,temp,it,False)
            cand=st.clone(); _,removed=self._controlled_destroy(cand,out,q)
            if not removed: continue
            touched=[(k,h) for k in range(self.inst.K) for h in range(self.inst.P)
                     if any(t in removed for t in cand.sol.cells[k][h])]
            cand.sol.remove(removed)
            for k,h in touched: cand.refresh(k,h)
            try: cand=self._guided_repair(cand,list(removed))
            except _Deadline: break
            cval=self._scalar(cand.R,cand.makespan,cand.violation)
            if cval<cur_val or self.rng.random()<math.exp(-(cval-cur_val)/max(temp_now,1e-9)):
                st,cur_val=cand,cval
            rank=self._rank(cand)
            if rank<best_rank: best,best_rank,since=cand.clone(),rank,0
            else: since+=1
            temp*=self.cooling; it+=1
        return dict(solution=best.sol.cells,makespan=best.makespan,R=best.R,
                    fff=FFF_REPORT(best.R),max_risk=max_risk(best.R),
                    feasible=best.violation<=1e-9,violation=best.violation,
                    evaluations=self.evaluations,endpoint_candidates=self.endpoint_candidates,
                    iterations=it)
