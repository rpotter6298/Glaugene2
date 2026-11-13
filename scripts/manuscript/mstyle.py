#!/usr/bin/env python3
"""Shared style for manuscript figures."""
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MM = 1 / 25.4
W2 = 180 * MM          # two-column plate
W1 = 89 * MM           # single column
OUT = Path(os.environ['GG2_FIGDIR']) if os.environ.get('GG2_FIGDIR') else (
    Path(__file__).resolve().parents[2] / 'manuscript' / 'final_pass')
OUT.mkdir(parents=True, exist_ok=True)

C_LENS, C_TEAR = '#3a6ea5', '#c1443f'
C_CTRL, C_CASE = '#6b7780', '#c1443f'
GREY, INK = '#888888', '#222222'

RC = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 7.5,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.linewidth': .7,
    'xtick.major.width': .7,
    'ytick.major.width': .7,
    'xtick.major.size': 2.6,
    'ytick.major.size': 2.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'figure.dpi': 110,
    'savefig.bbox': 'standard',
    'pdf.fonttype': 42,        # embed as TrueType, not Type 3: editable in Illustrator
    'ps.fonttype': 42,
}


def use():
    plt.rcParams.update(RC)


def panel(ax, letter, dx=-0.17, dy=1.06, size=None):
    """Bold lower-case panel letter at the body size (Scientific Reports), placed
    outside the axes top left; size=None follows rcParams."""
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=plt.rcParams['font.size'] if size is None else size,
            fontweight='bold', va='top', ha='left', color=INK)


def save(fig, name):
    """PDF for typesetting, PNG for drafts; no bbox cropping, so panel letters do not shift."""
    for ext, kw in (('pdf', {}), ('png', dict(dpi=400))):
        f = OUT / f'{name}.{ext}'
        fig.savefig(f, facecolor='white', **kw)
    print(f'wrote {OUT}/{name}.pdf and .png  '
          f'({fig.get_size_inches()[0] / MM:.0f} x {fig.get_size_inches()[1] / MM:.0f} mm)')
