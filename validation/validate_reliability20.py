"""Independent ANOVA/agreement arithmetic and full partition checks of transfer."""
from pathlib import Path
import csv,importlib.util,json,math,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('driver',ROOT/'experiments/run_reliability20.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)

def independent_icc(x):
    n,k=x.shape;row=x.mean(axis=1);col=x.mean(axis=0);grand=x.mean()
    ms_subject=k*np.square(row-grand).sum()/(n-1)
    ms_session=n*np.square(col-grand).sum()/(k-1)
    residual=x-row[:,None]-col[None,:]+grand
    ms_error=np.square(residual).sum()/((n-1)*(k-1))
    return (ms_subject-ms_error)/(ms_subject+(k-1)*ms_error+k*(ms_session-ms_error)/n)

def main():
    rows,pairs=driver.load();r=json.loads((ROOT/'results/reliability20.json').read_text())
    cohorts=driver.cohorts_of(pairs)
    assert {k:len(v) for k,v in cohorts.items()}==r['cohort_sizes']
    assert len(rows)==40 and len(pairs)==20
    for a,b in pairs:
        assert a['gap_days']==b['gap_days'] and 7<=int(a['gap_days'])<=14
    item_checks=0
    for cohort in list(cohorts)+['coherent']:
        for s in driver.SCENARIOS:
            pp=cohorts.get(cohort,pairs)
            if cohort=='coherent':pp=[p for p in pp if all(z[s+'_coherent']=='1' for z in p)]
            for e in driver.ENDPOINTS:
                values=[tuple(float(z[s+'_'+e]) for z in p) for p in pp if all(z[s+'_'+e] for z in p)]
                x=np.array(values);v=r['estimates'][cohort][s+'_'+e]
                assert len(x)==v['n']
                assert abs(independent_icc(x)-v['icc_A1'])<1e-12
                diff=x[:,1]-x[:,0]
                assert abs(diff.mean()-v['mean_difference_R2_minus_R1'])<1e-12
                assert np.allclose(diff.mean()+np.array([-1,1])*1.96*diff.std(ddof=1),v['limits_agreement_95'],atol=1e-12)
                assert v['defined_bootstrap_replicates']==2000
                item_checks+=1
    allowed={'code','session','gap_days','consulted_r1','remembers_numbers','new_information','checks_passed','ceiling_check_passed','completion_minutes','practicality'}
    allowed|={s+'_'+e for s in driver.SCENARIOS for e in driver.ENDPOINTS+['a','c','confidence','coherent','c_unbounded']}
    assert set(rows[0])==allowed
    assert all(all(not v or v.replace('.','',1).isdigit() for k,v in z.items() if k not in ['code','session']) for z in rows)
    from instances_io import CanonicalInstances
    from exact_small import SubsetTSP
    ci=CanonicalInstances();p=json.loads((ROOT/'results/reliability20_planning.json').read_text());checked=0
    # Complement enumeration is independent of the driver's recursive partition generator.
    groups=np.array([(m,1023^m) for m in range(1,1024) if m&1])
    assert len(groups)==512
    for rec in p['raw']:
        ts,ss=rec['context'].split('+');pp=[q for q in pairs if all(z[s+'_coherent']=='1' for z in q for s in [ts,ss])]
        assert len(pp)==rec['n']
        bases=[float(np.median([float(q[0][s+'_b']) for q in pp])) for s in [ts,ss]]
        assert bases==rec['reference_b_R1']
        inst=ci.build('S10',rec['seed']);tsp=SubsetTSP(np.array([[inst.modal_travel(i,j) for j in range(11)] for i in range(11)]))
        travel=tsp.best;service=np.array([sum(inst.service_modal[i+1] for i in range(10) if m&(1<<i)) for m in range(1024)])
        f=[np.array([[float(q[j][s+'_q060'])/b for s,b in zip([ts,ss],bases)] for q in pp]) for j in [0,1]]
        duration=[z[:,0,None]*travel+z[:,1,None]*service for z in f]
        vectors={'joint_R1':duration[0].max(axis=0),'joint_R2':duration[1].max(axis=0),'joint_union':np.vstack(duration).max(axis=0)}
        for j in [0,1]:
            ft,fs=np.median(f[j],axis=0);vectors['consensus_R'+str(j+1)]=ft*travel+fs*service
        vectors['cartesian_union']=max(z[:,0].max() for z in f)*travel+max(z[:,1].max() for z in f)*service
        for k,v in vectors.items():
            best=float(np.minimum.reduce(np.max(v[groups],axis=1)))
            assert abs(best-rec['minimum_shift_minutes'][k])<1e-7
        m=rec['R1_joint_plan_masks'];assert m[0]&m[1]==0 and m[0]|m[1]==1023
        for mask,tour in zip(m,rec['R1_joint_plan_tours']):
            assert sum(1<<(i-1) for i in tour)==mask
            seq=[0]+tour+[0];length=sum(inst.modal_travel(a,b) for a,b in zip(seq,seq[1:])) if tour else 0
            assert abs(length-travel[mask])<1e-7
        L=rec['minimum_shift_minutes']['joint_R1'];under=float(duration[1][:,m].max())
        assert abs(under-rec['R1_joint_plan_R2_duration'])<1e-7
        assert abs(max(0,under-L)-rec['R1_joint_plan_R2_excess_minutes'])<1e-7
        checked+=1
    payload=dict(status='passed',forms=40,paired_experts=20,independent_ANOVA_and_LoA_items=item_checks,
        independent_complement_partition_cases=checked,partitions_per_case=512,public_numeric_schema='passed')
    (ROOT/'results/reliability20_validation.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
