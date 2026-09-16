#!/usr/bin/env python3
"""Exact-rational checks of idea.md against exhaustive mask enumeration.

Run: python3 verification/verify_idea.py
Optional real implementation checks: --implementation (requires torch/numpy).
Finite checks supplement the proofs in verification/report.md, not replace them.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as Q
from itertools import groupby, product
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def records(v):
    """Independent oracle: enumerate masks and parse maximal runs by groupby."""
    out = []
    for mask in product((0, 1), repeat=len(v)):
        runs = [len(list(g)) for bit, g in groupby(mask) if bit]
        out.append((mask, sum(mask), len(runs), sum(x*b for x,b in zip(v, mask)), runs))
    return out


def full_dp(v):
    """Document's F recurrence; retain each stage to test all prefix states."""
    stages = [{(0, 0, 0): 0}]
    for i, vi in enumerate(v, 1):
        prev, curr = stages[-1], {}
        for r in range(i+1):
            for k in range(i+1):
                skip = [prev[t] for t in ((r,k,0),(r,k,1)) if t in prev]
                take = [prev[t]+vi for t in ((r-1,k,1),(r-1,k-1,0)) if t in prev]
                if skip: curr[r,k,0] = max(skip)
                if take: curr[r,k,1] = max(take)
        stages.append(curr)
    return stages


def g_dp(v, eta, a):
    """Document's G recurrence, storing a witness for Dinkelbach updates."""
    prev = {(0,0): (Q(0), ())}
    for i, vi in enumerate(v, 1):
        curr = {}
        for r in range(i+1):
            opts = [(prev[r,s][0], prev[r,s][1]+(0,)) for s in (0,1) if (r,s) in prev]
            if opts: curr[r,0] = max(opts)
            opts = [(prev[r-1,s][0]+vi-(eta*a if s==0 else 0), prev[r-1,s][1]+(1,))
                    for s in (0,1) if (r-1,s) in prev]
            if opts: curr[r,1] = max(opts)
        prev = curr
    return prev


def d_dp(v, lam, a, c):
    """Document's two-state relaxed coverage recurrence."""
    d0, d1 = Q(0), None
    for vi in v:
        n0 = min(x for x in (d0, d1) if x is not None)
        n1 = c-lam*vi + min(x for x in (d1, d0+a) if x is not None)
        d0, d1 = n0, n1
    return min(d0, d1)


def check_case(v, a, c, rng, counts):
    n = len(v)
    all_masks = records(v)
    counts['masks'] += len(all_masks)
    states = full_dp(v)
    for i, actual in enumerate(states):
        expected = {}
        for mask,r,k,importance,_ in records(v[:i]):
            key = (r,k,mask[-1] if i else 0)
            expected[key] = max(expected.get(key, -1), importance)
        assert actual == expected, ('F',v,i,actual,expected)
        counts['prefix_states'] += len(expected)
    frontier = {}
    for (r,k,s), importance in states[-1].items():
        frontier[r,k] = max(frontier.get((r,k), -1), importance)
    prefix = [0]
    for vi in v: prefix.append(prefix[-1]+vi)
    # Deliberately non-monotone positive costs: theorem requires no monotonicity.
    table = {r: Q(rng.randint(1,25), rng.randint(1,7)) for r in range(1,n+1)}
    for mask,r,k,importance,runs in all_masks:
        assert k == sum(b and (i==0 or not mask[i-1]) for i,b in enumerate(mask))
        if r:
            assert k <= min(r,n-r+1)
            cost = sum(table[length] for length in runs)
            chunks = []
            pos = 0
            for bit, group in groupby(mask):
                length=len(list(group))
                if bit: chunks.append(Q(sum(v[pos:pos+length]),1)/table[length])
                pos += length
            assert Q(importance,1)/cost <= max(chunks)
    for r in range(1,n+1):
        fixed = [rec for rec in all_masks if rec[1]==r]
        assert max(rec[2] for rec in fixed) == min(r,n-r+1)
        assert max(rec[3] for rec in fixed) == sum(sorted(v,reverse=True)[:r])
        assert {rec[2] for rec in fixed} == set(range(1,min(r,n-r+1)+1))
        intervals=[Q(prefix[start+length]-prefix[start],1)/table[length]
                   for length in range(1,r+1) for start in range(n-length+1)]
        assert len(intervals) == r*(2*n-r+1)//2
        brute_upper = max(Q(rec[3],1)/sum(table[x] for x in rec[4])
                          for rec in all_masks if 1<=rec[1]<=r)
        assert max(intervals)==brute_upper, ('single_chunk',v,r)
        optimum = max(Q(rec[3],1)/(a*rec[2]+c*r) for rec in fixed)
        assert optimum == max(Q(imp,1)/(a*k+c*r) for (rr,k),imp in frontier.items() if rr==r)
        eta=Q(0)
        seen=set()
        steps=0
        while True:
            assert eta not in seen
            seen.add(eta)
            result=g_dp(v,eta,a)
            value, mask=max(result[r,s] for s in (0,1) if (r,s) in result)
            residual=value-eta*c*r
            exact_residual=max(rec[3]-eta*(a*rec[2]+c*r) for rec in fixed)
            assert residual == exact_residual
            steps+=1
            if residual==0: break
            k=sum(b and (i==0 or not mask[i-1]) for i,b in enumerate(mask))
            eta_next=Q(sum(vi*b for vi,b in zip(v,mask)),1)/(a*k+c*r)
            assert eta_next>eta
            eta=eta_next
        assert eta==optimum
        counts['max_dinkelbach_iterations']=max(counts['max_dinkelbach_iterations'],steps)
        for probe,sign in ((optimum/2,1),(optimum,0),(optimum+1,-1)):
            result=g_dp(v,probe,a)
            psi=max(result[r,s][0] for s in (0,1) if (r,s) in result)-probe*c*r
            if optimum==0 and probe==0: assert psi==0
            else: assert (psi>0)-(psi<0)==sign
        counts['fixed_R_problems']+=1
    for eta in (Q(0),Q(1,3),Q(2),Q(17,2)):
        result=g_dp(v,eta,a)
        for r in range(n+1):
            for s in (0,1):
                eligible=[rec[3]-eta*a*rec[2] for rec in all_masks if rec[1]==r and rec[0][-1]==s]
                if eligible: assert result[r,s][0]==max(eligible)
                else: assert (r,s) not in result
    previous_importances=[]
    for lam in (Q(0),Q(1,3),Q(1),Q(2),Q(10)):
        costs=[a*k+c*r-lam*imp for _,r,k,imp,_ in all_masks]
        best=min(costs)
        assert d_dp(v,lam,a,c)==best
        ties=[rec[3] for rec,cost in zip(all_masks,costs) if cost==best]
        if previous_importances: assert min(ties)>=max(previous_importances)
        previous_importances=ties
        counts['lagrangian_problems']+=1
    # All attainable positive coverage levels plus intermediate levels.
    total=sum(v)
    thresholds=sorted({Q(rec[3]) for rec in all_masks if rec[3]>0} |
                      {Q(total)*alpha for alpha in (Q(1,3),Q(4,5),Q(99,100),Q(1)) if total>0})
    prev_cost=Q(0)
    for bound in thresholds:
        best=min((a*k+c*r,-imp) for _,r,k,imp,_ in all_masks if imp>=bound)
        got=min((a*k+c*r,-imp) for (r,k),imp in frontier.items() if imp>=bound)
        assert got==best
        assert got[0]>=prev_cost
        prev_cost=got[0]
        cost,neg_imp=got
        assert not any(a*k+c*r<=cost and imp>=-neg_imp and (a*k+c*r<cost or imp>-neg_imp)
                       for _,r,k,imp,_ in all_masks)
        counts['coverage_problems']+=1
    for _,r,k,imp,_ in all_masks:
        if imp>0:
            lower=min(a*kk+c*rr for (rr,kk),ii in frontier.items() if ii>=imp)
            assert lower>0 and a*k+c*r>=lower
    counts['vectors']+=1


def profile_checks():
    result={}
    for device,sat,advertised in [('agx',236,7450),('nano',348,3500)]:
        path=ROOT/f'vlm-flash/src/vlmflash/profiles/orin-{device}.json'
        table=json.loads(path.read_text())['table']
        pairs=sorted((int(k),v) for k,v in table.items())
        assert [x for x,y in pairs]==list(range(1,len(pairs)+1))
        assert all(y>0 for x,y in pairs)
        count=len(pairs)
        mx=sum(x for x,y in pairs)/count; my=sum(y for x,y in pairs)/count
        c=sum((x-mx)*(y-my) for x,y in pairs)/sum((x-mx)**2 for x,y in pairs)
        a=my-c*mx
        actual=[y for x,y in pairs]; predicted=[a+c*x for x,y in pairs]
        def r2(y,p):
            mean=sum(y)/len(y)
            return 1-sum((u-v)**2 for u,v in zip(y,p))/sum((u-mean)**2 for u in y)
        errors=[abs(y-p)/y for y,p in zip(actual,predicted)]
        beta=[1000*x/(1024*y) for x,y in pairs]
        bhat=[1000*x/(1024*(a+c*x)) for x,y in pairs]
        advertised_mib=advertised*10**6/2**20
        asymptote=1000/(1024*c)
        result[device]={
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'entries':count,
            'a_ms':a,'c_ms_per_KiB':c,'latency_R2':r2(actual,predicted),
            'latency_MAPE_percent':100*sum(errors)/count,'max_relative_error_percent':100*max(errors),
            'throughput_R2':r2(beta,bhat),'saturation_size_KiB':sat,
            'saturation_measured_MiB_s':1000*sat/(1024*table[str(sat)]),
            'saturation_predicted_MiB_s':1000*sat/(1024*(a+c*sat)),
            'asymptote_MiB_s':asymptote,'saturation_fraction_of_asymptote':c*sat/(a+c*sat),
            'advertised_MiB_s':advertised_mib,
            'asymptote_advertised_difference_percent':100*abs(asymptote-advertised_mib)/advertised_mib,
        }
    return result


def counterexamples():
    # Coverage optimum is unsupported: (I,L)=(2,3), compared to (0,0),(3,4).
    # To beat empty lambda >= 3/2; to beat full lambda <= 1: impossible.
    v=(2,1); a=Q(2); c=Q(1)
    recs=records(v)
    assert min(a*k+c*r for _,r,k,imp,_ in recs if imp>=2)==3
    assert Q(3,2)>Q(1)
    fixed=records((3,0,3))
    assert max(Q(imp,k+r) for _,r,k,imp,_ in fixed if r==2)==Q(3,2)
    assert max(Q(imp,k+r) for _,r,k,imp,_ in fixed if r==2 and k==1)==1
    decreasing=full_dp((0,10,0))[-1]
    assert max(imp for (r,k,s),imp in decreasing.items() if r==2 and k==1)==10
    assert max(imp for (r,k,s),imp in decreasing.items() if r==2 and k==2)==0
    return {
        'unsupported_coverage':{'v':[2,1],'a':2,'c':1,'B':2,'optimal_latency':3,'relaxed_feasible_latency':4,
                                'required_lambda_lower':'3/2','required_lambda_upper':'1',
                                'gap_at_requested_coverage':'1/3','gap_at_achieved_coverage':'0'},
        'fixed_R_single_chunk_fails':{'v':[3,0,3],'a':1,'c':1,'R':2,'best_ratio':'3/2','best_one_chunk_ratio':'1'},
        'exact_k_importance_decreases':{'v':[0,10,0],'R':2,'I_star_k1':10,'I_star_k2':0},
    }


def implementation_checks():
    sys.path.insert(0,str(ROOT/'vlm-flash/src'))
    import torch
    from vlmflash import LatencyTable, ChunkParams, select_chunks, SparseLinear
    from types import SimpleNamespace
    torch.set_num_threads(1)
    table=LatencyTable.bundled('orin-agx')
    empty=select_chunks(torch.ones(4),2,1.,table,impl='torch')
    assert empty.num_selected==0
    params=ChunkParams(start_kb=1,end_kb=2,step_kb=1,jump_cap_kb=1)
    merged=select_chunks(torch.ones(4),4,1.,table,params,impl='torch')
    mask_cost=table.mask_elat_ms(merged.mask,1.)
    assert merged.num_selected==4 and abs(merged.est_cost_ms-4*table.read_ms(1.))<1e-12
    assert abs(mask_cost-table.read_ms(4.))<1e-12
    assert merged.est_cost_ms != mask_cost
    seen=[]
    def capture(importance,budget,row_size):
        seen.append(budget)
        return SimpleNamespace(num_selected=budget)
    stub=SimpleNamespace(nc_sparsity=.25,in_features=5,nc_policy=capture,row_size_kb=1.)
    SparseLinear._select(stub,torch.ones(1,5),SimpleNamespace(collect=False))
    assert seen==[4] and int((1-.25)*5)==3
    # Match the documented greedy pass against the actual pure-torch path.
    rng=random.Random(47)
    samples=0
    for n in range(2,11):
        for _ in range(8):
            v=[rng.randrange(8) for _ in range(n)]
            budget=rng.randrange(1,n+1)
            params=ChunkParams(start_kb=2,end_kb=5,step_kb=1,jump_cap_kb=1)
            candidates=[]
            for size in range(2,min(4,n)+1):
                for start in range(n-size+1):
                    # Match float32 score rounding used by the implementation.
                    score=(torch.tensor(sum(v[start:start+size]),dtype=torch.float32)/
                           torch.tensor(table.read_ms(size),dtype=torch.float32)).item()
                    candidates.append((score,start,size))
            mask=[False]*n
            for _,start,size in sorted(candidates,key=lambda x:-x[0]):
                if sum(mask)+size<=budget and not any(mask[start:start+size]):
                    mask[start:start+size]=[True]*size
            got=select_chunks(torch.tensor(v,dtype=torch.float32),budget,1.,table,params,impl='torch')
            assert got.mask.tolist()==mask
            samples+=1
    return {'greedy_comparisons':samples,'empty_mask_example':{'N':4,'R':2,'selected':0},
            'adjacent_candidates':{'candidate_cost_ms':merged.est_cost_ms,'maximal_run_cost_ms':mask_cost},
            'budget_rounding':{'N':5,'rho':.25,'document_floor':3,'actual_select_budget':seen[0]},
            'torch_version':torch.__version__}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--implementation',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    rng=random.Random(20260916)
    counts=dict(vectors=0,masks=0,prefix_states=0,fixed_R_problems=0,
                lagrangian_problems=0,coverage_problems=0,max_dinkelbach_iterations=0)
    # Every ternary activation vector through N=6 (including all-zero boundary inputs).
    for n in range(1,7):
        for v in product((0,1,2),repeat=n):
            a,c=rng.choice(((Q(1),Q(1)),(Q(2),Q(1)),(Q(1,3),Q(5,7))))
            check_case(v,a,c,rng,counts)
        print(f'Passed all ternary vectors at N={n}',flush=True)
    # Wider inputs and nonintegral importances; every mask is still enumerated.
    for n in range(1,11):
        for _ in range(20):
            v=tuple(Q(rng.randrange(21),rng.randrange(1,6)) for _ in range(n))
            a,c=Q(rng.randrange(1,8),3),Q(rng.randrange(1,8),5)
            check_case(v,a,c,rng,counts)
        print(f'Passed rational inputs at N={n}',flush=True)
    result={'status':'passed','idea_sha256':hashlib.sha256((ROOT/'idea.md').read_bytes()).hexdigest(),
            'submodule_commit':subprocess.check_output(['git','-C',str(ROOT/'vlm-flash'),'rev-parse','HEAD'],text=True).strip(),
            'seed':20260916,'counts':counts,'profiles':profile_checks(),'counterexamples':counterexamples()}
    if args.implementation: result['implementation']=implementation_checks()
    if args.output: args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
