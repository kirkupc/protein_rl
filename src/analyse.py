"""
Phase 4 — Analysis, Visualisation & Write-up

Generates publication-quality figures from evaluation results:
  - Hit rate analysis and filter breakdown
  - Metric distributions (iPTM, pLDDT, RMSD)
  - Structural diversity (secondary structure composition)
  - Interface contact analysis
  - Threshold sweep / sensitivity curves

Usage:
    from src.analyse import BinderAnalyser
    analyser = BinderAnalyser(results_df, output_dir="figures/EGFR")
    analyser.plot_metric_distributions()
    analyser.plot_hit_rate_summary()
    analyser.plot_threshold_sweep("iptm")
    analyser.plot_secondary_structure_composition()
    report = analyser.generate_report()
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend; switch to "TkAgg" for display
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not installed. pip install matplotlib")

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

from src.evaluate import threshold_sweep


# ------------------------------------------------------------------ #
# Style helpers
# ------------------------------------------------------------------ #

PALETTE = {
    "passing": "#2E86AB",   # steel blue
    "failing": "#E63946",   # red
    "neutral": "#457B9D",   # muted blue
    "highlight": "#F4A261", # orange accent
}

def _set_style():
    if HAS_SNS:
        sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
    elif HAS_MPL:
        plt.rcParams.update({
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
            "axes.grid": True,
            "grid.alpha": 0.4,
        })


class BinderAnalyser:
    """
    Analysis and visualisation hub for binder evaluation results.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of BinderEvaluator.evaluate().
    output_dir : Path
        Directory to save figures and reports.
    target_name : str, optional
        Used in figure titles.
    """

    def __init__(
        self,
        results_df: pd.DataFrame,
        output_dir: Optional[Path] = None,
        target_name: str = "Target",
    ):
        self.df = results_df.copy()
        self.output_dir = Path(output_dir or "figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_name = target_name
        _set_style()

    # ------------------------------------------------------------------ #
    # 1. Hit rate analysis
    # ------------------------------------------------------------------ #

    def plot_hit_rate_summary(self, save: bool = True) -> Optional["plt.Figure"]:
        """
        Bar chart of pass rates for each filter and combined hit rate.
        Also prints textual comparison vs. published RFdiffusion hit rates.
        """
        if not HAS_MPL:
            print("[analyse] matplotlib not available.")
            return None

        filter_cols = ["passes_iptm", "passes_plddt", "passes_rmsd", "passes_pae", "passes_all"]
        labels = ["iPTM", "pLDDT", "RMSD", "PAE", "All filters"]
        rates = [self.df[col].mean() for col in filter_cols]

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = [PALETTE["passing"] if r > 0.1 else PALETTE["failing"] for r in rates]
        bars = ax.bar(labels, rates, color=colors, width=0.6, edgecolor="white", linewidth=0.5)

        # Annotate bars with percentage
        for bar, rate in zip(bars, rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{rate:.1%}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

        # Reference line: published RFdiffusion hit rate (~5-15%)
        ax.axhline(0.10, color="grey", linestyle="--", linewidth=1.2, label="RFdiffusion avg. ~10%")
        ax.set_ylim(0, min(max(rates) * 1.3 + 0.1, 1.05))
        ax.set_ylabel("Pass Rate", fontsize=12)
        ax.set_title(
            f"{self.target_name}: Filter Pass Rates (n={len(self.df)})",
            fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=9)
        fig.tight_layout()

        if save:
            path = self.output_dir / "hit_rate_summary.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[analyse] Saved {path}")

        # Textual comparison
        combined_hit_rate = self.df["passes_all"].mean()
        print(
            f"\n[analyse] Hit rate summary for {self.target_name}:\n"
            f"  Combined hit rate:    {combined_hit_rate:.1%}\n"
            f"  Published RFdiffusion baseline: ~5–15%\n"
            f"  Ratio vs baseline:   {combined_hit_rate/0.10:.2f}x\n"
        )
        return fig

    # ------------------------------------------------------------------ #
    # 2. Metric distributions
    # ------------------------------------------------------------------ #

    def plot_metric_distributions(self, save: bool = True) -> Optional["plt.Figure"]:
        """
        2×2 grid of violin / histogram plots for iPTM, pLDDT, RMSD, PAE.
        Passing vs. failing candidates overlaid.
        """
        if not HAS_MPL:
            return None

        metrics = [
            ("iptm", "iPTM", "higher = better"),
            ("plddt_binder", "pLDDT (binder)", "higher = better"),
            ("interface_rmsd", "Interface RMSD (Å)", "lower = better"),
            ("pae_inter", "Inter-chain PAE", "lower = better"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes = axes.flatten()

        for ax, (col, label, note) in zip(axes, metrics):
            passing = self.df[self.df["passes_all"]][col].dropna()
            failing = self.df[~self.df["passes_all"]][col].dropna()

            if HAS_SNS:
                data = pd.DataFrame({
                    "value": pd.concat([passing, failing]),
                    "group": ["Passing"] * len(passing) + ["Failing"] * len(failing),
                })
                sns.violinplot(
                    data=data, x="group", y="value", ax=ax,
                    palette={"Passing": PALETTE["passing"], "Failing": PALETTE["failing"]},
                    inner="quartile", cut=0, linewidth=0.8,
                )
            else:
                ax.hist(failing, bins=30, alpha=0.6, color=PALETTE["failing"],
                        label="Failing", density=True)
                ax.hist(passing, bins=30, alpha=0.7, color=PALETTE["passing"],
                        label="Passing", density=True)
                ax.legend(fontsize=9)

            ax.set_title(f"{label}\n({note})", fontsize=11, fontweight="bold")
            ax.set_xlabel(label, fontsize=10)
            ax.set_ylabel("Density", fontsize=10)

        fig.suptitle(
            f"{self.target_name}: Metric Distributions (n={len(self.df)})",
            fontsize=14, fontweight="bold", y=1.01,
        )
        fig.tight_layout()

        if save:
            path = self.output_dir / "metric_distributions.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[analyse] Saved {path}")

        return fig

    # ------------------------------------------------------------------ #
    # 3. Structural diversity
    # ------------------------------------------------------------------ #

    def plot_secondary_structure_composition(
        self,
        seq_col: str = "sequence",
        save: bool = True,
    ) -> Optional["plt.Figure"]:
        """
        Stacked bar showing helix/coil fraction per design bucket.
        Helix-preferred amino acids used as a proxy (AELM).
        """
        if not HAS_MPL:
            return None

        helix_aa = set("AELM")

        def helix_frac(seq: str) -> float:
            if not seq:
                return 0.0
            return sum(1 for aa in seq if aa in helix_aa) / len(seq)

        self.df["helix_frac"] = self.df[seq_col].apply(helix_frac)
        self.df["coil_frac"] = 1 - self.df["helix_frac"]

        # Bin by binder length for x-axis
        self.df["length_bin"] = pd.cut(
            self.df["binder_length"] if "binder_length" in self.df.columns
            else self.df[seq_col].str.len(),
            bins=5,
        )

        grp = self.df.groupby("length_bin")[["helix_frac", "coil_frac"]].mean()

        fig, ax = plt.subplots(figsize=(9, 5))
        grp[["helix_frac", "coil_frac"]].plot(
            kind="bar", stacked=True, ax=ax,
            color=[PALETTE["passing"], PALETTE["neutral"]],
            edgecolor="white",
        )
        ax.set_xlabel("Binder Length Bin", fontsize=11)
        ax.set_ylabel("Fraction", fontsize=11)
        ax.set_title(
            f"{self.target_name}: Secondary Structure Composition by Length",
            fontsize=12, fontweight="bold",
        )
        ax.legend(["Helix (AELM proxy)", "Other"], fontsize=9)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()

        if save:
            path = self.output_dir / "secondary_structure_composition.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[analyse] Saved {path}")

        return fig

    # ------------------------------------------------------------------ #
    # 4. Interface analysis
    # ------------------------------------------------------------------ #

    def plot_interface_contacts(
        self,
        contacts_df: pd.DataFrame,
        save: bool = True,
    ) -> Optional["plt.Figure"]:
        """
        Scatter plot of interface contacts for top-ranked binders.

        Parameters
        ----------
        contacts_df : pd.DataFrame
            DataFrame with columns: candidate_id, n_contacts, n_hbonds_approx,
            n_hydrophobic, iptm. Obtained from BinderEvaluator.compute_interface_contacts.
        """
        if not HAS_MPL:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # H-bonds vs iPTM
        ax = axes[0]
        ax.scatter(
            contacts_df["n_hbonds_approx"], contacts_df["iptm"],
            c=contacts_df["iptm"], cmap="viridis", s=60, alpha=0.7, edgecolors="w",
        )
        ax.set_xlabel("Approx. H-bonds at interface", fontsize=11)
        ax.set_ylabel("iPTM", fontsize=11)
        ax.set_title("H-bonds vs iPTM", fontsize=12, fontweight="bold")

        # Hydrophobic contacts vs iPTM
        ax = axes[1]
        sc = ax.scatter(
            contacts_df["n_hydrophobic"], contacts_df["iptm"],
            c=contacts_df["n_contacts"], cmap="plasma", s=60, alpha=0.7, edgecolors="w",
        )
        fig.colorbar(sc, ax=ax, label="Total contacts")
        ax.set_xlabel("Hydrophobic contacts at interface", fontsize=11)
        ax.set_ylabel("iPTM", fontsize=11)
        ax.set_title("Hydrophobic contacts vs iPTM", fontsize=12, fontweight="bold")

        fig.suptitle(f"{self.target_name}: Interface Analysis", fontsize=13, fontweight="bold")
        fig.tight_layout()

        if save:
            path = self.output_dir / "interface_contacts.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[analyse] Saved {path}")

        return fig

    # ------------------------------------------------------------------ #
    # 5. Threshold sensitivity
    # ------------------------------------------------------------------ #

    def plot_threshold_sweep(
        self,
        metric: str = "iptm",
        n_points: int = 50,
        save: bool = True,
    ) -> Optional["plt.Figure"]:
        """
        Plot how hit rate changes as a function of the iPTM (or other) threshold.
        """
        if not HAS_MPL:
            return None

        col_values = self.df[metric].dropna()
        thresholds = np.linspace(col_values.min(), col_values.max(), n_points)
        sweep_df = threshold_sweep(self.df, metric=metric, thresholds=thresholds)

        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax2 = ax1.twinx()

        ax1.plot(
            sweep_df["threshold"], sweep_df["hit_rate"] * 100,
            color=PALETTE["passing"], linewidth=2.5, label="Hit rate (%)",
        )
        ax2.plot(
            sweep_df["threshold"], sweep_df["n_passing"],
            color=PALETTE["highlight"], linewidth=1.8, linestyle="--",
            label="# passing designs",
        )

        ax1.set_xlabel(f"{metric} threshold", fontsize=12)
        ax1.set_ylabel("Hit Rate (%)", fontsize=12, color=PALETTE["passing"])
        ax2.set_ylabel("# Passing Designs", fontsize=12, color=PALETTE["highlight"])
        ax1.tick_params(axis="y", labelcolor=PALETTE["passing"])
        ax2.tick_params(axis="y", labelcolor=PALETTE["highlight"])

        ax1.set_title(
            f"{self.target_name}: Sensitivity to {metric} threshold",
            fontsize=13, fontweight="bold",
        )

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

        fig.tight_layout()

        if save:
            path = self.output_dir / f"threshold_sweep_{metric}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[analyse] Saved {path}")

        return fig

    # ------------------------------------------------------------------ #
    # 6. Scatter matrix / correlation
    # ------------------------------------------------------------------ #

    def plot_metric_correlations(self, save: bool = True) -> Optional["plt.Figure"]:
        """Pairwise scatter matrix for the four key metrics."""
        if not HAS_MPL or not HAS_SNS:
            return None

        metrics = ["iptm", "plddt_binder", "interface_rmsd", "pae_inter"]
        plot_df = self.df[metrics + ["passes_all"]].dropna()

        fig = plt.figure(figsize=(10, 9))
        g = sns.PairGrid(
            plot_df, vars=metrics,
            hue="passes_all",
            palette={True: PALETTE["passing"], False: PALETTE["failing"]},
        )
        g.map_upper(sns.scatterplot, s=20, alpha=0.5, edgecolors="none")
        g.map_lower(sns.kdeplot, fill=True, alpha=0.4)
        g.map_diag(sns.histplot, kde=True)
        g.add_legend(title="Passes all filters")
        g.figure.suptitle(
            f"{self.target_name}: Metric Correlations", fontsize=13, fontweight="bold", y=1.01
        )
        g.figure.tight_layout()

        if save:
            path = self.output_dir / "metric_correlations.png"
            g.figure.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[analyse] Saved {path}")

        return g.figure

    # ------------------------------------------------------------------ #
    # 7. Text report
    # ------------------------------------------------------------------ #

    def generate_report(self) -> str:
        """
        Generate a concise textual summary of the evaluation results.
        Returns the report as a string (also prints to stdout).
        """
        n = len(self.df)
        n_pass = int(self.df["passes_all"].sum())
        hit_rate = n_pass / n if n > 0 else 0.0

        top5 = self.df.nsmallest(5, "rank")[["rank", "candidate_id", "iptm",
                                              "plddt_binder", "interface_rmsd",
                                              "pae_inter", "sequence"]]

        report = f"""
========================================================
  Binder Evaluation Report — {self.target_name}
========================================================
Total candidates evaluated : {n}
Passing all filters        : {n_pass}  ({hit_rate:.1%})
  └ iPTM only              : {self.df['passes_iptm'].sum()}  ({self.df['passes_iptm'].mean():.1%})
  └ pLDDT only             : {self.df['passes_plddt'].sum()}  ({self.df['passes_plddt'].mean():.1%})
  └ RMSD only              : {self.df['passes_rmsd'].sum()}  ({self.df['passes_rmsd'].mean():.1%})
  └ PAE only               : {self.df['passes_pae'].sum()}  ({self.df['passes_pae'].mean():.1%})

Metric summary (mean ± std):
  iPTM           : {self.df['iptm'].mean():.3f} ± {self.df['iptm'].std():.3f}
  pLDDT (binder) : {self.df['plddt_binder'].mean():.1f} ± {self.df['plddt_binder'].std():.1f}
  RMSD (Å)       : {self.df['interface_rmsd'].mean():.2f} ± {self.df['interface_rmsd'].std():.2f}
  inter PAE      : {self.df['pae_inter'].mean():.2f} ± {self.df['pae_inter'].std():.2f}

Top 5 candidates by iPTM:
{top5.to_string(index=False)}
========================================================
"""
        print(report)
        report_path = self.output_dir / "evaluation_report.txt"
        report_path.write_text(report)
        print(f"[analyse] Report saved to {report_path}")
        return report

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #

    def run_all(self) -> None:
        """Run all standard analysis plots and generate report."""
        print(f"[analyse] Running full analysis for {self.target_name} …")
        self.plot_hit_rate_summary()
        self.plot_metric_distributions()
        self.plot_secondary_structure_composition()
        self.plot_threshold_sweep("iptm")
        self.plot_threshold_sweep("plddt_binder")
        if HAS_SNS:
            self.plot_metric_correlations()
        self.generate_report()
        print(f"[analyse] All figures saved to {self.output_dir}")
