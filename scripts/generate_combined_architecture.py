"""
Generate a high-quality, academic-standard comparison diagram of the neural network architectures.
Updated with simplified, elegant labels and clean typography to prevent text overlaps.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# IEEE Style Configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Georgia'],
    'font.size': 8.5,
    'axes.labelsize': 9.5,
    'axes.titlesize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Refined academic color palette (high contrast, clean)
COLORS = {
    'bg_pnn': '#f2f7fa',      # Very light blue
    'edge_pnn': '#1b4f72',    # Dark navy
    'bg_gnn': '#f0f9f8',      # Very light teal
    'edge_gnn': '#0e6251',    # Dark teal
    'bg_dec': '#fefcf3',      # Very light gold
    'edge_dec': '#9a7d0a',    # Dark gold
    'bg_mask': '#fdf2f2',     # Very light red
    'edge_mask': '#922b21',   # Dark red
    'text_dark': '#1c2833',
    'arrow': '#2c3e50',
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def draw_rounded_box(ax, x, y, w, h, bg_color, edge_color, label_text, fontsize=8.5, fontweight='normal'):
    """Helper to draw a perfectly proportioned rounded box with centered text."""
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.15",
        facecolor=bg_color,
        edgecolor=edge_color,
        linewidth=1.2,
        zorder=3
    )
    ax.add_patch(box)
    
    # Center text inside box
    ax.text(
        x + w/2.0, y + h/2.0,
        label_text,
        ha='center', va='center',
        color=COLORS['text_dark'],
        fontsize=fontsize,
        fontweight=fontweight,
        zorder=4
    )

def generate_combined_diagram():
    # Width: 7.2 inches (IEEE standard double-column width)
    # Height: 3.8 inches
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.8))

    for ax in [ax1, ax2]:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal') # Prevent warping of rounded boxes and circles
        ax.axis('off')

    # =========================================================================
    # LEFT PANEL: PointerMCMNet
    # =========================================================================
    ax1.set_title("(a) PointerMCMNet Neural Architecture", fontweight='bold', pad=8, fontsize=9.5, color=COLORS['edge_pnn'])

    # 1. Inputs
    draw_rounded_box(ax1, 0.4, 8.3, 5.0, 1.0, COLORS['bg_pnn'], COLORS['edge_pnn'], 
                     "Input Dimensions\n$D = [d_0, d_1, \\dots, d_n]$", fontweight='bold')

    # Arrow Input -> Encoder
    ax1.annotate("", xy=(2.9, 6.8), xytext=(2.9, 8.3), 
                 arrowprops=dict(arrowstyle="-|>", color=COLORS['arrow'], lw=1.2, mutation_scale=8))

    # 2. Transformer Encoder
    draw_rounded_box(ax1, 0.4, 4.9, 5.0, 1.9, COLORS['bg_pnn'], COLORS['edge_pnn'], 
                     "Transformer Encoder\n(6 Layers, 8 Attention Heads)\n\nSelf-Attention Representation:\n$\\mathbf{H} = [\\mathbf{h}_0, \\dots, \\mathbf{h}_n]$",
                     fontsize=8.0)

    # Arrow Encoder -> Decoder
    ax1.annotate("", xy=(2.9, 3.3), xytext=(2.9, 4.9), 
                 arrowprops=dict(arrowstyle="-|>", color=COLORS['arrow'], lw=1.2, mutation_scale=8))

    # 3. Pointer Decoder
    draw_rounded_box(ax1, 0.4, 1.4, 5.0, 1.9, COLORS['bg_dec'], COLORS['edge_dec'], 
                     "Pointer Decoder\n(Bahdanau Cross-Attention)\n\nAttention Scores:\n$u_j = \\mathrm{Attention}(\\mathbf{s}_i, \\mathbf{h}_j)$",
                     fontsize=8.0)

    # Attention flow arrow (Encoder -> Decoder query)
    ax1.annotate("", xy=(5.4, 2.35), xytext=(5.4, 5.85),
                 arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=-0.4", 
                                 color=COLORS['edge_dec'], lw=1.0, ls="--", mutation_scale=6))
    ax1.text(5.9, 4.1, "Attention\nMechanism", ha='left', va='center', fontsize=7.5, color=COLORS['edge_dec'], fontweight='bold')

    # 4. Validity Mask
    draw_rounded_box(ax1, 5.9, 1.9, 3.8, 1.0, COLORS['bg_mask'], COLORS['edge_mask'], 
                     "Validity Mask\n$u_j = -\\infty$ if $j \\notin [i, j-1]$\n(Strict Feasibility Floor)", fontsize=7.5)

    # Arrow Mask -> Decoder
    ax1.annotate("", xy=(5.4, 2.35), xytext=(5.9, 2.35), 
                 arrowprops=dict(arrowstyle="-|>", color=COLORS['edge_mask'], lw=1.0, mutation_scale=6))

    # 5. Output Split Point
    ax1.annotate("", xy=(2.9, 0.6), xytext=(2.9, 1.4), 
                 arrowprops=dict(arrowstyle="-|>", color=COLORS['arrow'], lw=1.2, mutation_scale=8))
    
    # Draw circular split node
    circle = patches.Circle((2.9, 0.35), 0.25, fc='#d4efdf', ec='#27ae60', lw=1.2, zorder=3)
    ax1.add_patch(circle)
    ax1.text(2.9, 0.35, "$k^*$", ha='center', va='center', fontweight='bold', color='#1e8449', fontsize=9.5)
    ax1.text(3.3, 0.35, "Predicted Split Point\n$k^* \\in [i, j-1]$", ha='left', va='center', fontsize=7.5, color=COLORS['text_dark'])


    # =========================================================================
    # RIGHT PANEL: GraphMCMNet
    # =========================================================================
    ax2.set_title("(b) GraphMCMNet Neural Architecture", fontweight='bold', pad=8, fontsize=9.5, color=COLORS['edge_gnn'])

    # 1. DP Dependency Graph Representation
    # Container box for graph visualization
    ax2.add_patch(patches.FancyBboxPatch((0.5, 7.3), 4.2, 2.0, boxstyle="round,pad=0.0,rounding_size=0.1", 
                                        fc='none', ec=COLORS['edge_gnn'], lw=1.0, ls=':'))
    ax2.text(2.6, 7.33, "DP Sub-problem Graph (n=3)", ha='center', va='bottom', fontsize=7.0, color=COLORS['edge_gnn'], fontweight='bold')

    # Draw Nodes in Graph (Triangular DP sub-problem structure)
    nodes = {
        '(1,3)': (2.6, 8.95),
        '(1,2)': (1.7, 8.25),
        '(2,3)': (3.5, 8.25),
        '(1,1)': (0.8, 7.55),
        '(2,2)': (2.6, 7.55),
        '(3,3)': (4.4, 7.55)
    }

    # Draw parent-child DP dependency edges
    edges = [
        ((2.6, 8.95), (1.7, 8.25)),  # (1,3) -> (1,2)
        ((2.6, 8.95), (3.5, 8.25)),  # (1,3) -> (2,3)
        ((1.7, 8.25), (0.8, 7.55)),  # (1,2) -> (1,1)
        ((1.7, 8.25), (2.6, 7.55)),  # (1,2) -> (2,2)
        ((3.5, 8.25), (2.6, 7.55)),  # (2,3) -> (2,2)
        ((3.5, 8.25), (4.4, 7.55)),  # (2,3) -> (3,3)
    ]
    for start, end in edges:
        ax2.annotate("", xy=end, xytext=start, 
                     arrowprops=dict(arrowstyle="-", color='#b2dfdb', lw=0.8, zorder=2))

    # Draw horizontal same-length neighbor edges (dashed)
    ax2.plot([1.7, 3.5], [8.25, 8.25], color='#b2dfdb', linestyle='--', lw=0.8, zorder=2)
    ax2.plot([0.8, 2.6], [7.55, 7.55], color='#b2dfdb', linestyle='--', lw=0.8, zorder=2)
    ax2.plot([2.6, 4.4], [7.55, 7.55], color='#b2dfdb', linestyle='--', lw=0.8, zorder=2)

    # Add circles and centered text for nodes
    for label, (nx, ny) in nodes.items():
        circle = patches.Circle((nx, ny), 0.22, fc=COLORS['bg_gnn'], ec=COLORS['edge_gnn'], lw=1.0, zorder=5)
        ax2.add_patch(circle)
        ax2.text(nx, ny, label, ha='center', va='center', fontsize=6.0, fontweight='bold', zorder=6, color=COLORS['text_dark'])

    # 2. Node Features Box
    draw_rounded_box(ax2, 5.5, 7.3, 4.0, 2.0, COLORS['bg_gnn'], COLORS['edge_gnn'], 
                     "Node Features\n$\\mathbf{x}_{(i,j)} \\in \\mathbb{R}^{10}$\n\n- Boundary dimensions\n- Sub-chain length\n- Dimensions statistics", fontsize=7.5)

    # Arrow Graph/Features -> Message Passing
    ax2.annotate("", xy=(2.6, 6.2), xytext=(2.6, 7.3), 
                 arrowprops=dict(arrowstyle="-|>", color=COLORS['arrow'], lw=1.2, mutation_scale=8))
    ax2.annotate("", xy=(4.8, 6.2), xytext=(5.5, 7.3), 
                 arrowprops=dict(arrowstyle="-|>", color=COLORS['arrow'], lw=1.0, mutation_scale=6))

    # 3. Message Passing Box
    draw_rounded_box(ax2, 0.3, 4.3, 5.2, 1.9, COLORS['bg_gnn'], COLORS['edge_gnn'],
                     "Gated Graph Message Passing\n(6 Gated Attention Layers)\n\n- Neighborhood aggregation\n- Gated residual node updates",
                     fontsize=8.0)

    # Arrow Message Passing -> Split Scorer
    ax2.annotate("", xy=(2.9, 3.2), xytext=(2.9, 4.3), 
                 arrowprops=dict(arrowstyle="-|>", color=COLORS['arrow'], lw=1.2, mutation_scale=8))

    # 4. Split Scorer Box
    draw_rounded_box(ax2, 0.3, 1.3, 5.2, 1.9, COLORS['bg_dec'], COLORS['edge_dec'],
                     "Split Scorer (MLP)\n\nConcatenated Input:\n$\\mathbf{h}_{\\mathrm{parent}} \\parallel \\mathbf{h}_{\\mathrm{left\\_child}} \\parallel \\mathbf{h}_{\\mathrm{right\\_child}}$\nOutput: $\\mathrm{Softmax}(\\mathrm{MLP}(\\cdot))$",
                     fontsize=7.5)

    # 5. Auxiliary Cost Head
    draw_rounded_box(ax2, 5.8, 1.9, 3.8, 1.2, COLORS['bg_dec'], COLORS['edge_dec'],
                     "Auxiliary Cost Head\n\nLog-Cost Regression:\n$\\hat{C} = \\mathrm{MLP}(\\mathbf{h}_{(1,n)})$",
                     fontsize=7.5)

    # Connection from Message Passing to Cost Head
    ax2.annotate("", xy=(5.8, 2.5), xytext=(4.5, 4.3), 
                 arrowprops=dict(arrowstyle="-|>", color=COLORS['edge_dec'], lw=1.0, ls="--", mutation_scale=6))

    # 6. Outputs
    ax2.annotate("", xy=(2.9, 0.6), xytext=(2.9, 1.3), 
                 arrowprops=dict(arrowstyle="-|>", color=COLORS['arrow'], lw=1.2, mutation_scale=8))
    ax2.text(2.9, 0.35, "Predicted Split Table\n$\\mathrm{splits}[(i,j)] = k^*$", ha='center', va='center', fontweight='bold', fontsize=8, color='#27ae60')

    fig.tight_layout()
    
    # Save files
    png_path = os.path.join(OUTPUT_DIR, 'neural_architecture.png')
    pdf_path = os.path.join(OUTPUT_DIR, 'neural_architecture.pdf')
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[SUCCESS] Saved {png_path} and {pdf_path}")

if __name__ == '__main__':
    generate_combined_diagram()
