import math
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def _numeric_comparison(comparison_table):
    df = comparison_table.copy()
    df["UNet height"] = pd.to_numeric(df["UNet height"], errors="coerce")
    df["Software height"] = pd.to_numeric(df["Software height"], errors="coerce")
    return df.dropna(subset=["UNet height", "Software height"])

# Per-compound calibration curve, U-Net vs software
def calibration_curves(comparison_table, out_path, ncols=3):
    compounds = comparison_table["Compound"].unique()
    if len(compounds) == 0:
        return None
    nrows = math.ceil(len(compounds) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for i, compound in enumerate(compounds):
        ax = axes[i // ncols][i % ncols]
        sub = comparison_table[comparison_table["Compound"] == compound].sort_values("Concentration")
        ax.plot(sub["Concentration"], sub["UNet height"], linestyle="--", marker="o", label="UNet")
        ax.plot(sub["Concentration"], sub["Software height"], linestyle="--", marker="s", label="Software")
        ax.set_title(str(compound)[:40])
        ax.set_xlabel("Concentration")
        ax.set_ylabel("Peak height")
        ax.legend()

    for j in range(len(compounds), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path

# Scatter with a through-origin fit, worst-N compounds highlighted
def unet_vs_software(comparison_table, out_path, task_name="", worst_n=5):
    df = _numeric_comparison(comparison_table)
    if df.empty:
        return None

    df["compound_id"] = df["Compound"].astype(str) + " | " + df["MS1 label"].astype(str)
    df["compound_short"] = (
        df["Compound"].astype(str).str.slice(0, 20) + "... | "
        + df["MS1 label"].astype(str).str.replace(task_name, "", regex=False).str.slice(0, 22)
    )

    x = df["Software height"].to_numpy(dtype=float)
    y = df["UNet height"].to_numpy(dtype=float)
    denom = np.sum(x * x)
    if denom == 0:
        return None
    slope = np.sum(x * y) / denom

    df["distance"] = np.abs(slope * x - y) / np.sqrt(slope ** 2 + 1)
    dist_table = (
        df.groupby("compound_id")
        .agg(mean_distance=("distance", "mean"), compound_short=("compound_short", "first"))
        .reset_index()
    )
    worst = dist_table.sort_values("mean_distance", ascending=False).head(worst_n)
    normal = df[~df["compound_id"].isin(set(worst["compound_id"]))]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(normal["Software height"], normal["UNet height"], s=18, alpha=0.6,
               color="grey", label="Other compounds")
    for _, row in worst.iterrows():
        sub = df[df["compound_id"] == row["compound_id"]]
        ax.scatter(sub["Software height"], sub["UNet height"], s=30, alpha=0.9,
                   label=f"{row['compound_short']} | rel.dist={row['mean_distance']:.2f}")

    x_line = np.linspace(0, x.max() * 1.05, 200)
    ax.plot(x_line, slope * x_line, linestyle="--", linewidth=2, color="black",
            label=f"Linear regression through 0: y={slope:.2f}x")
    ax.set_xlabel("Software height")
    ax.set_ylabel("UNet height")
    ax.set_title("UNet height vs Software height")
    ax.set_xlim(0, x.max() * 1.05)
    ax.set_ylim(0, max(y.max() * 1.05, 1e-9))
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path

def unet_vs_software_plain(comparison_table, out_path):
    df = _numeric_comparison(comparison_table)
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(df["Software height"], df["UNet height"], s=25, alpha=0.25,
               color="grey", label="All matched compounds")
    ax.set_xlabel("Software height")
    ax.set_ylabel("UNet height")
    ax.set_title("UNet height vs Software height")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
