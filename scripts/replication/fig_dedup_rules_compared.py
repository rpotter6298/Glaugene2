#!/usr/bin/env python3
"""How the two deduplication rules differ on the main cohort: BEA k=2 per-position vs nf-core UMICollapse whole-read."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'plots/branches/nfcore-dedup'; OUT.mkdir(parents=True,exist_ok=True)
def load(arm):
    m=pd.read_csv(ROOT/f'counts/{arm}/counts_{arm}_main.tsv',sep='\t',index_col=0)
    m.columns=[c.split('_')[-2] if c.startswith('BEA25P077') else c for c in m.columns]   # BEA25P077_RP_19_13B_G2 -> 13B
    return m
braw,bded,nraw,nded=load('BEA-raw'),load('BEA-dedup'),load('nfcore'),load('nfcore-dedup')
common=sorted(set(braw.columns)&set(bded.columns)&set(nraw.columns)&set(nded.columns))
mirs=sorted(set(braw.index)&set(nraw.index)&set(bded.index)&set(nded.index))
braw,bded,nraw,nded=[m.loc[mirs,common] for m in (braw,bded,nraw,nded)]
# sample id suffix: B = lens capsule, A = tear fluid (patient_map.xlsx tissue_name)
lens=[c for c in common if c.endswith('B')]; tear=[c for c in common if c.endswith('A')]
def long(raw,ded,label):
    d=pd.DataFrame({'raw':raw.stack(),'ded':ded.stack()}); d=d[d.raw>=50]; d['rule']=label; d['tissue']=['tear' if s in tear else 'lens' for s in d.index.get_level_values(1)]
    d['kept']=d.ded/d.raw; return d
L=pd.concat([long(braw,bded,'BEA: directional, k=2, per position'),long(nraw,nded,'nf-core: whole-read directional, k=1')])
fig,axes=plt.subplots(1,2,figsize=(12,5.2))
ax=axes[0]; bins=np.logspace(np.log10(50),np.log10(L.raw.max()),14)
for (rule,tis),g in L.groupby(['rule','tissue']):
    g=g.copy(); g['bin']=pd.cut(g.raw,bins); med=g.groupby('bin',observed=True).kept.median(); x=[b.mid for b in med.index]
    ax.plot(x,med.values,marker='o',ms=4,lw=1.6,ls='-' if tis=='tear' else '--',color='tab:red' if rule.startswith('BEA') else 'tab:blue',label=f'{rule} - {tis}')
ax.set_xscale('log'); ax.set_ylim(0,1.05); ax.set_xlabel('raw reads per miRNA per sample'); ax.set_ylabel('fraction of reads kept after deduplication (median per bin)')
ax.set_title('A  How hard each rule collapses, by abundance',loc='left',fontsize=11); ax.grid(alpha=.25); ax.legend(fontsize=8)
ax=axes[1]; both=pd.DataFrame({'bea':bded.stack(),'nf':nded.stack()}); both=both[(both.bea>=20)&(both.nf>=20)]
ax.scatter(both.bea,both.nf,s=5,alpha=.35,color='.3',edgecolors='none'); lim=[10,max(both.max())*1.5]; ax.plot(lim,lim,'k-',lw=.8)
r=np.log2(both.nf/both.bea); ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lim); ax.set_ylim(lim)
lr=np.polyfit(np.log10(both.bea),np.log10(both.nf),1)
ax.text(.03,.97,f"n = {len(both):,} miRNA x sample\nmedian nf-core / BEA = {2**np.median(r):.2f}\nlog-log slope = {lr[0]:.2f}\nSpearman rho = {both.bea.rank().corr(both.nf.rank()):.3f}",transform=ax.transAxes,va='top',fontsize=9,bbox=dict(fc='white',ec='.7'))
ax.set_xlabel('BEA-dedup count'); ax.set_ylabel('nfcore-dedup count'); ax.set_title('B  The two deduplicated matrices',loc='left',fontsize=11); ax.grid(alpha=.25,which='both')
fig.suptitle(f'Two deduplication rules on the main cohort ({len(common)} samples)',fontsize=11); fig.tight_layout()
fig.savefig(OUT/'fig_dedup_rules_compared.png',dpi=160); print('wrote',OUT/'fig_dedup_rules_compared.png')
