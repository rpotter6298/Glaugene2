#!/usr/bin/env python3
"""Table S5 - does the conclusion depend on the count arm? One row per arm x tissue."""
import numpy as np, pandas as pd
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ARMS=['BEA-raw','BEA-dedup','nfcore','nfcore-dedup']
def pdir(arm): return ROOT/'plots' if arm=='BEA-raw' else ROOT/'plots/branches'/arm
def rd(arm,stage,name):
    f=pdir(arm)/stage/name
    return pd.read_csv(f,sep='\t') if f.exists() else None
rows=[]
for arm in ARMS:
    corr=rd(arm,'stage_4','correction_summary.tsv'); perm=rd(arm,'stage_4','permutation_summary.tsv')
    stab=rd(arm,'stage_4','stability.tsv'); eq=rd(arm,'stage_4','equivalence_summary.tsv')
    rf=rd(arm,'stage_5','rf_loocv.tsv'); sb=rd(arm,'stage_5','rf_selection_bias.tsv')
    for tis in ['lens','tear']:
        r={'arm':arm,'tissue':tis}
        if corr is not None:
            for m,lab,norm in [('edgeR-QL','edgeR','CPM'),('NBSR','NBSR','internal-RPM')]:   # NBSR normalises internally (RPM)
                x=corr[(corr.tissue==tis)&(corr.norm==norm)&(corr.method==m)&(corr.kind=='relative')]   # primary (relative) analysis only
                if len(x): r[f'{lab} tested']=int(x.tested.iloc[0]); r[f'{lab} p<.05']=int(x.p05.iloc[0]); r[f'{lab} BH<.10']=int(x.BH10.iloc[0]); r[f'{lab} minBH']=round(float(x.min_BH.iloc[0]),2)
        if perm is not None:
            p=perm[(perm.tissue==tis)&(perm.norm=='CPM')&(perm.metric=='p05')]
            if len(p): r['null p<.05 (median)']=float(p.null_median.iloc[0]); r['perm p']=round(float(p.perm_p.iloc[0]),2)
        if stab is not None:
            s=stab[stab.tissue==tis]; r['stable (leave-one-out)']=int(s.stable.sum()) if len(s) else 0
        if eq is not None:
            e=eq[(eq.tissue==tis)&(eq.method=='NBSR')]
            if len(e): r['TOST-equiv (Δ=1)']=f"{int(e.equiv_1.iloc[0])}/{int(e.n.iloc[0])}"
        if rf is not None:
            x=rf[rf.tissue==tis]
            if len(x): r['RF LOOCV AUC']=round(float(x.auc.iloc[0]),2); r['RF perm p']=round(float(x.perm_p.iloc[0]),2)
        if sb is not None:
            x=sb[(sb.tissue==tis)&(sb.selection=='honest')]
            if len(x): r['RF honest AUC (5-fold)']=round(float(x.auc.iloc[0]),2)
        rows.append(r)
t=pd.DataFrame(rows).fillna('-')
out=ROOT/'plots/arm_comparison.tsv'; t.to_csv(out,sep='\t',index=False)
pd.set_option('display.width',250); print(t.to_string(index=False)); print('\nwrote',out)
