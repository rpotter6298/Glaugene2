#!/usr/bin/env python3
"""Which deduplication rule reproduces BEA's delivered dedup counts: (A) exact (position, UMI) vs (B) umi_tools k=2."""
import glob, re, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; SRC=ROOT/'counts/BEA-dedup/reproduction'
OUT=ROOT/'plots/branches/BEA-dedup'; OUT.mkdir(parents=True,exist_ok=True)
frames=[]
for f in sorted(SRC.glob('*.tsv')):
    d=pd.read_csv(f,sep='\t',index_col=0); d['sample']=f.stem; frames.append(d)
d=pd.concat(frames); d=d[(d.bea_dedup>=20)&(d.our_raw>=20)]
# sample id suffix: B = lens capsule, A = tear fluid (patient_map.xlsx tissue_name)
lens=[s for s in d['sample'].unique() if s.endswith('B')]; tear=[s for s in d['sample'].unique() if s.endswith('A')]
cmap={**{s:c for s,c in zip(tear,plt.cm.Blues(np.linspace(.45,.95,len(tear))))},**{s:c for s,c in zip(lens,plt.cm.Oranges(np.linspace(.45,.95,len(lens))))}}
fig,axes=plt.subplots(1,2,figsize=(11,5.3),sharex=True,sharey=True)
for ax,col,title in [(axes[0],'exact_umi','A  Their stated rule: one count per (position, UMI)'),
                     (axes[1],'dir_k2','B  umi_tools directional, edit distance 2')]:
    for s in tear+lens:
        g=d[d['sample']==s]
        ax.scatter(g.bea_dedup,g[col],s=9,alpha=.55,color=cmap[s],edgecolors='none',label=f"{s} ({'tear' if s in tear else 'lens'})")
    lim=[10,d.bea_dedup.max()*1.6]; ax.plot(lim,lim,'k-',lw=.8); ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lim); ax.set_ylim(lim)
    ok=d[col].notna()&(d[col]>0); r=np.log2(d.loc[ok,col]/d.loc[ok,'bea_dedup']); big=d.loc[ok,'bea_dedup']>=1000
    ax.text(.03,.97,f"total ours/BEA: {d.loc[ok,col].sum()/d.loc[ok,'bea_dedup'].sum():.2f}\nmedian ratio: {2**np.median(r):.2f}\nBEA≥1000: median |log2| {np.median(np.abs(r[big])):.2f}, {np.mean(np.abs(r[big])<=np.log2(1.1)):.0%} within 10%",
            transform=ax.transAxes,va='top',fontsize=9,bbox=dict(fc='white',ec='.7',alpha=.9))
    ax.set_title(title,fontsize=11,loc='left'); ax.set_xlabel('BEA delivered deduplicated count'); ax.grid(alpha=.25,which='both')
axes[0].set_ylabel('our count from their alignment'); axes[1].legend(fontsize=8,loc='lower right',markerscale=2,frameon=True)
fig.suptitle('Reproducing BEA-dedup: 7 samples re-aligned (fastp → subread-align → miRBase v22 mature; BEA align with STAR)',fontsize=11)
fig.tight_layout(); fig.savefig(OUT/'fig_dedup_reproduction.png',dpi=160); print('wrote',OUT/'fig_dedup_reproduction.png')
rows=[]
for s,g in d.groupby('sample'):
    rows.append({'sample':s,'tissue':'tear' if s in tear else 'lens','BEA_raw':int(g.bea_raw.sum()),'our_raw':int(g.our_raw.sum()),'BEA_dedup':int(g.bea_dedup.sum()),
                 'exact_umi/BEA':round(g.exact_umi.sum()/g.bea_dedup.sum(),3),
                 'dir_k2/BEA':round(g.dir_k2.sum()/g.bea_dedup.sum(),3)})
t=pd.DataFrame(rows).sort_values(['tissue','sample']); t.to_csv(OUT/'dedup_reproduction_totals.tsv',sep='\t',index=False); print(t.to_string(index=False))
