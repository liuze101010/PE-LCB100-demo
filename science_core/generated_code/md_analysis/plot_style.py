"""Unified matplotlib style for MD rheology plots.

Macromolecules / J Rheology submission-ready defaults: sans serif, 1.0 line,
ticks in, frame on, 600 dpi raster fallback. Use apply_rc() once per script
before any plotting; use the palette helpers for consistent topology / stress
component colors across per-case and aggregate plots.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


# Single-column = 3.3 in, double-column = 6.9 in (Macromolecules guideline).
SINGLE_COL_WIDTH_IN = 3.3
DOUBLE_COL_WIDTH_IN = 6.9
PANEL_HEIGHT_IN = 2.4

# Stress components: 3 distinct colorblind-friendly hues.
COMPONENT_COLORS = {
    "Pres-XY": "#0072B2",  # blue
    "Pres-XZ": "#D55E00",  # vermillion
    "Pres-YZ": "#009E73",  # bluish green
}

# Topology family palette (used by aggregate plots later).
TOPOLOGY_COLORS = {
    "star3":   "#0072B2",
    "center":  "#56B4E9",
    "comb":    "#009E73",
    "htype":   "#F0E442",
    "dbmid":   "#E69F00",
    "dbfar":   "#D55E00",
    "pompom":  "#CC79A7",
    "tri":     "#332288",
    "acomb":   "#117733",
    "q4comb":  "#999933",
    "q4pompom": "#882255",
    "q4clust": "#AA4499",
    "q4grad":  "#44AA99",
    "q5comb":  "#88CCEE",
    "q5pompom": "#DDCC77",
    "q5clust": "#661100",
    "q5grad":  "#888888",
}


def apply_rc():
    """Apply unified rcParams. Call once at script start."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,    # TrueType so journals can edit text in PDF
        "ps.fonttype": 42,
        "figure.constrained_layout.use": True,
    })


def get_component_color(component_name):
    return COMPONENT_COLORS.get(component_name, "#666666")


def get_topology_color(topology):
    return TOPOLOGY_COLORS.get(topology, "#666666")


def save_fig(fig, output_path, formats=("pdf", "png")):
    """Save fig to multiple formats. PDF for journals, PNG for previews."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stem = output_path.with_suffix("")
    for fmt in formats:
        fig.savefig(f"{stem}.{fmt}", format=fmt)
    plt.close(fig)
