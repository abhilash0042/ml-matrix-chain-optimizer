"""
Generate 3 IEEE-standard figures for the MCM research paper.

Figure 1: MAPE by Chain Length (log-scale) — demonstrates GNN's OOD generalization
Figure 2: Mathematical Validity Gap — the paper's most critical finding
Figure 3: Multi-Metric Radar Chart — holistic model comparison

All figures use IEEE-compliant styling:
  - 3.5" (single column) or 7" (double column) widths
  - Times New Roman font
  - Minimum 8pt font size
  - High-contrast grayscale-safe colors
  - 300 DPI minimum
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.patches import FancyBboxPatch
import os

# ============================================================
# IEEE Style Configuration
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Georgia'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.6,
    'grid.linewidth': 0.4,
    'lines.linewidth': 1.2,
    'patch.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    'axes.grid': False,
    'text.usetex': False,
})

# IEEE-compliant color palette (accessible, grayscale-distinguishable)
COLORS = {
    'GNN':     '#1a5276',   # Deep navy
    'Pointer': '#2e86c1',   # Medium blue
    'XGBoost': '#c0392b',   # Brick red
    'RF':      '#e67e22',   # Orange
}

# Hatching patterns for grayscale distinguishability
HATCHES = {
    'GNN':     '',
    'Pointer': '///',
    'XGBoost': 'xxx',
    'RF':      '...',
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FIGURE 4: Structural Scaling (Line Chart) 
# Key story: GNN maintains exact matches even OOD
# ============================================================
def fig4_structural_scaling():
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    buckets = ['5-10', '11-20', '21-30', '31-40', '41-50\n(OOD)']
    x = np.arange(len(buckets))
    
    # Data from results (Exact Match %)
    gnn_exact = [98.0, 96.0, 94.0, 88.0, 89.0]
    ptr_exact = [97.0, 91.0, 89.0, 88.0, 79.0]

    ax.plot(x, gnn_exact, marker='s', markersize=5, linewidth=2.0, 
            color=COLORS['GNN'], label='GraphMCMNet (GNN)')
    ax.plot(x, ptr_exact, marker='o', markersize=5, linewidth=2.0, 
            color=COLORS['Pointer'], linestyle='--', label='PointerMCMNet')

    ax.set_ylabel('Exact Match Rate (%)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_xlabel('Chain Length ($n$)', fontweight='bold')
    
    # Y-axis formatting
    ax.set_ylim(70, 102)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))

    # Add value labels
    for i in range(len(x)):
        ax.annotate(f"{gnn_exact[i]:.0f}%", (x[i], gnn_exact[i]), 
                    textcoords="offset points", xytext=(0,6), ha='center',
                    fontsize=7, color=COLORS['GNN'], fontweight='bold')
        ax.annotate(f"{ptr_exact[i]:.0f}%", (x[i], ptr_exact[i]), 
                    textcoords="offset points", xytext=(0,-12), ha='center',
                    fontsize=7, color=COLORS['Pointer'])

    # Legend
    ax.legend(loc='lower left', frameon=True, fancybox=False, 
              edgecolor='gray', framealpha=0.95, fontsize=7)

    # Grid
    ax.grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Spine styling
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig4_structural_scaling.pdf')
    fig.savefig(path)
    path_png = os.path.join(OUTPUT_DIR, 'fig4_structural_scaling.png')
    fig.savefig(path_png)
    plt.close(fig)
    print(f"[OK] Figure 4 saved: {path}")
    return path_png


# ============================================================
# FIGURE 2: Mathematical Validity Gap
# Key story: 100% validity for neural vs 18-22% for statistical
# ============================================================
def fig2_validity_accuracy():
    fig, ax1 = plt.subplots(figsize=(3.5, 2.5))

    # --- Validity ---
    models = ['GraphMCMNet', 'PointerMCMNet', 'XGBoost', 'Random\nForest']
    validity = [100.0, 100.0, 22.0, 18.2]
    colors_list = [COLORS['GNN'], COLORS['Pointer'], COLORS['XGBoost'], COLORS['RF']]
    hatches_list = [HATCHES['GNN'], HATCHES['Pointer'], HATCHES['XGBoost'], HATCHES['RF']]

    bars = ax1.barh(models, validity, height=0.55, 
                     color=colors_list, edgecolor='white', linewidth=0.5, zorder=3)
    for bar, h in zip(bars, hatches_list):
        bar.set_hatch(h)

    # Add value labels
    for i, (v, bar) in enumerate(zip(validity, bars)):
        if v > 50:
            ax1.text(v - 2, i, f'{v:.1f}%', ha='right', va='center', 
                    fontsize=8, fontweight='bold', color='white')
        else:
            ax1.text(v + 1.5, i, f'{v:.1f}%', ha='left', va='center',
                    fontsize=8, fontweight='bold', color=colors_list[i])

    # Validity threshold line
    ax1.axvline(x=100, color='#27ae60', linestyle='--', linewidth=0.8, alpha=0.7, zorder=2)

    ax1.set_xlim(0, 115)
    ax1.set_xlabel('Mathematical Validity (%)', fontweight='bold')
    ax1.set_title('Mathematical Validity Rate by Model', fontweight='bold', fontsize=9)
    ax1.invert_yaxis()
    for spine in ['top', 'right']:
        ax1.spines[spine].set_visible(False)
    ax1.xaxis.grid(True, which='major', linestyle='-', linewidth=0.3, alpha=0.3, zorder=0)

    # Add a "STRUCTURAL" / "STATISTICAL" group label
    # Use x = -0.55 to push it left of the long 'PointerMCMNet' text
    ax1.text(-0.55, 0.75, 'Structural', fontsize=7, ha='center', va='center',
             rotation=90, color=COLORS['GNN'], fontweight='bold',
             transform=ax1.transAxes)
    ax1.text(-0.55, 0.25, 'Statistical', fontsize=7, ha='center', va='center',
             rotation=90, color=COLORS['XGBoost'], fontweight='bold',
             transform=ax1.transAxes)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig2_validity_accuracy.pdf')
    fig.savefig(path)
    path_png = os.path.join(OUTPUT_DIR, 'fig2_validity_accuracy.png')
    fig.savefig(path_png)
    plt.close(fig)
    print(f"[OK] Figure 2 saved: {path}")
    return path_png


# ============================================================
# FIGURE 3: MAPE by Distribution Type (Heatmap-style)
# Key story: GNN dominates across ALL distributions
# ============================================================
def fig3_distribution_heatmap():
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    distributions = ['Uniform', 'Spiky', 'Bottleneck', 'Monotone']
    
    # MAPE data by distribution
    gnn     = [0.003, 0.114, 0.000, 0.000]
    pointer = [0.032, 0.500, 0.059, 0.000]
    xgboost = [54.18, 41.40, 41.97, 7.19]
    rf      = [51.18, 46.68, 62.94, 12.34]

    x = np.arange(len(distributions))
    width = 0.19

    ax.bar(x - 1.5*width, gnn, width, color=COLORS['GNN'],
           hatch=HATCHES['GNN'], edgecolor='white', linewidth=0.3,
           label='GraphMCMNet', zorder=3)
    ax.bar(x - 0.5*width, pointer, width, color=COLORS['Pointer'],
           hatch=HATCHES['Pointer'], edgecolor='white', linewidth=0.3,
           label='PointerMCMNet', zorder=3)
    ax.bar(x + 0.5*width, xgboost, width, color=COLORS['XGBoost'],
           hatch=HATCHES['XGBoost'], edgecolor='white', linewidth=0.3,
           label='XGBoost', zorder=3)
    ax.bar(x + 1.5*width, rf, width, color=COLORS['RF'],
           hatch=HATCHES['RF'], edgecolor='white', linewidth=0.3,
           label='Random Forest', zorder=3)

    ax.set_yscale('log')
    ax.set_ylabel('MAPE (%)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(distributions)
    ax.set_xlabel('Dimension Distribution', fontweight='bold')

    # Set y-limits to show small values
    ax.set_ylim(0.0005, 100)

    # Custom y-axis formatter
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, _: f'{y:.3f}' if y < 0.01 else (f'{y:.2f}' if y < 1 else f'{y:.0f}')))

    # Add "perfect" annotations for 0.000% cases
    for i, v in enumerate(gnn):
        if v == 0.000:
            ax.annotate('0.000%', xy=(x[i] - 1.5*width, 0.001), 
                        fontsize=5.5, ha='center', va='bottom',
                        color=COLORS['GNN'], fontweight='bold')

    # Annotation: Spiky = hardest
    ax.annotate('Hardest\ndistribution',
                xy=(1, max(rf[1], xgboost[1]) * 1.1),
                xytext=(1.8, 55),
                fontsize=6, ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.6),
                color='gray', fontstyle='italic')

    # Reference lines
    ax.axhline(y=1, color='gray', linestyle=':', linewidth=0.4, alpha=0.5, zorder=1)
    ax.axhline(y=10, color='gray', linestyle=':', linewidth=0.4, alpha=0.5, zorder=1)

    # Legend
    ax.legend(bbox_to_anchor=(0.5, 1.05), loc='lower center', frameon=True, fancybox=False,
              edgecolor='gray', framealpha=0.95, ncol=2,
              fontsize=6.5, handletextpad=0.4)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.yaxis.grid(True, which='major', linestyle='-', linewidth=0.3, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig3_mape_distribution.pdf')
    fig.savefig(path)
    path_png = os.path.join(OUTPUT_DIR, 'fig3_mape_distribution.png')
    fig.savefig(path_png)
    plt.close(fig)
    print(f"[OK] Figure 3 saved: {path}")
    return path_png


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  Generating IEEE-Standard Figures for MCM Paper")
    print("=" * 60)
    
    p1 = fig4_structural_scaling()
    p2 = fig2_validity_accuracy()
    p3 = fig3_distribution_heatmap()
    
    print("\n" + "=" * 60)
    print("  All 3 figures generated successfully!")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nFigure descriptions:")
    print("  Fig 4: Exact Match by chain length (line chart) - structural scaling")
    print("  Fig 2: Validity + accuracy thresholds - the categorical gap")
    print("  Fig 3: MAPE by distribution - robustness across all types")
