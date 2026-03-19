"""
Phase 3 — In Silico Evaluation Pipeline

Scores binder candidates using structure prediction models (Chai-1 / ESMFold)
and extracts iPTM, pLDDT, interface RMSD, and pAE metrics — mirroring Latent-X
paper protocol.

Usage:
    from src.evaluate import BinderEvaluator
    evaluator = BinderEvaluator(target, backend="chai1")  # or "esmfold"
    results_df = evaluator.evaluate(candidates, n_seeds=5)
    passing = evaluator.apply_filters(results_df)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.generate import BinderCandidate
from src.target_prep import TargetProtein

# Optional imports — handle gracefully
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from chai_lab.chai1 import run_inference
    HAS_CHAI1 = True
except ImportError:
    HAS_CHAI1 = False

try:
    from transformers import EsmForProteinFolding, AutoTokenizer
    HAS_ESMFOLD = True
except ImportError:
    HAS_ESMFOLD = False

try:
    from Bio.PDB import PDBParser, Superimposer
    from Bio.PDB.Polypeptide import is_aa
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False


# Default filter thresholds (Latent-X paper values)
DEFAULT_THRESHOLDS = {
    "iptm_min": 0.55,
    "plddt_min": 70.0,
    "interface_rmsd_max": 2.0,
    "pae_max": 10.0,
}


@dataclass
class EvaluationResult:
    """Evaluation metrics for a single binder candidate."""
    candidate_id: str
    target_name: str
    sequence: str
    backbone_pdb: str

    # Primary metrics
    iptm: float = 0.0             # interface predicted TM-score [0, 1]
    plddt_binder: float = 0.0     # mean pLDDT over binder chain [0, 100]
    plddt_complex: float = 0.0    # mean pLDDT over full complex
    interface_rmsd: float = 99.9  # RMSD of binder vs generated pose (Å)
    pae_inter: float = 99.9       # mean PAE for inter-chain residue pairs

    # Seed statistics (across n_seeds runs)
    iptm_std: float = 0.0
    plddt_std: float = 0.0
    n_seeds: int = 1

    # Filter results
    passes_iptm: bool = False
    passes_plddt: bool = False
    passes_rmsd: bool = False
    passes_pae: bool = False
    passes_all: bool = False

    # Rank (lower = better; assigned after batch evaluation)
    rank: int = -1

    def to_dict(self) -> Dict:
        return {
            "candidate_id": self.candidate_id,
            "target_name": self.target_name,
            "sequence": self.sequence,
            "backbone_pdb": self.backbone_pdb,
            "iptm": self.iptm,
            "plddt_binder": self.plddt_binder,
            "plddt_complex": self.plddt_complex,
            "interface_rmsd": self.interface_rmsd,
            "pae_inter": self.pae_inter,
            "iptm_std": self.iptm_std,
            "plddt_std": self.plddt_std,
            "n_seeds": self.n_seeds,
            "passes_iptm": self.passes_iptm,
            "passes_plddt": self.passes_plddt,
            "passes_rmsd": self.passes_rmsd,
            "passes_pae": self.passes_pae,
            "passes_all": self.passes_all,
            "rank": self.rank,
        }


class BinderEvaluator:
    """
    Evaluates binder candidates with structure prediction and computes
    filtering metrics matching the Latent-X protocol.

    Parameters
    ----------
    target : TargetProtein
        Parsed target protein.
    backend : str
        Structure prediction backend: "chai1", "esmfold", or "mock".
        "mock" generates synthetic scores for testing without GPU.
    thresholds : dict, optional
        Override default filter thresholds.
    device : str
        PyTorch device ("cuda", "cpu").
    """

    def __init__(
        self,
        target: TargetProtein,
        backend: str = "mock",
        thresholds: Optional[Dict] = None,
        device: str = "cuda",
    ):
        self.target = target
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.device = device if (HAS_TORCH and torch.cuda.is_available()) else "cpu"

        # Resolve backend
        self.backend = self._resolve_backend(backend)
        print(f"[evaluate] Backend: {self.backend}  |  Device: {self.device}")

        self._esmfold_model = None   # lazy-loaded

    # ------------------------------------------------------------------ #
    # Main evaluation entry point
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        candidates: Union[List[BinderCandidate], pd.DataFrame],
        n_seeds: int = 5,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Evaluate all candidates and return a ranked DataFrame.

        Parameters
        ----------
        candidates : list of BinderCandidate or DataFrame with 'sequence' column
        n_seeds : int
            Number of prediction seeds per candidate (matching Latent-X protocol).
        verbose : bool
            Print progress.

        Returns
        -------
        pd.DataFrame with all EvaluationResult fields, sorted by iPTM desc.
        """
        if isinstance(candidates, pd.DataFrame):
            cand_list = self._df_to_candidates(candidates)
        else:
            cand_list = candidates

        results = []
        for i, cand in enumerate(cand_list):
            if verbose and i % 10 == 0:
                print(f"[evaluate] {i}/{len(cand_list)} candidates …")

            result = self._evaluate_single(cand, n_seeds)
            result = self._apply_filters(result)
            results.append(result)

        # Rank by iPTM
        results.sort(key=lambda r: r.iptm, reverse=True)
        for rank, result in enumerate(results):
            result.rank = rank + 1

        df = pd.DataFrame([r.to_dict() for r in results])
        print(
            f"[evaluate] Done. {len(df)} candidates evaluated. "
            f"Passing all filters: {df['passes_all'].sum()} "
            f"({df['passes_all'].mean():.1%})"
        )
        return df

    # ------------------------------------------------------------------ #
    # Single-candidate evaluation
    # ------------------------------------------------------------------ #

    def _evaluate_single(
        self, candidate: BinderCandidate, n_seeds: int
    ) -> EvaluationResult:
        """Run structure prediction for one candidate across n_seeds."""
        seed_metrics = []
        for seed in range(n_seeds):
            metrics = self._predict_structure(
                binder_seq=candidate.sequence,
                target_seq=self.target.sequence,
                seed=seed,
            )
            seed_metrics.append(metrics)

        # Aggregate across seeds
        iptm_vals = [m["iptm"] for m in seed_metrics]
        plddt_vals = [m["plddt_binder"] for m in seed_metrics]
        plddt_complex_vals = [m["plddt_complex"] for m in seed_metrics]
        rmsd_vals = [m["interface_rmsd"] for m in seed_metrics]
        pae_vals = [m["pae_inter"] for m in seed_metrics]

        return EvaluationResult(
            candidate_id=candidate.candidate_id,
            target_name=candidate.target_name,
            sequence=candidate.sequence,
            backbone_pdb=str(candidate.backbone_pdb),
            iptm=float(np.mean(iptm_vals)),
            plddt_binder=float(np.mean(plddt_vals)),
            plddt_complex=float(np.mean(plddt_complex_vals)),
            interface_rmsd=float(np.mean(rmsd_vals)),
            pae_inter=float(np.mean(pae_vals)),
            iptm_std=float(np.std(iptm_vals)),
            plddt_std=float(np.std(plddt_vals)),
            n_seeds=n_seeds,
        )

    # ------------------------------------------------------------------ #
    # Structure prediction backends
    # ------------------------------------------------------------------ #

    def _predict_structure(
        self, binder_seq: str, target_seq: str, seed: int = 0
    ) -> Dict[str, float]:
        """
        Predict complex structure and extract metrics.
        Dispatches to the appropriate backend.
        """
        if self.backend == "chai1":
            return self._predict_chai1(binder_seq, target_seq, seed)
        elif self.backend == "esmfold":
            return self._predict_esmfold(binder_seq, target_seq, seed)
        else:
            return self._predict_mock(binder_seq, target_seq, seed)

    def _predict_chai1(
        self, binder_seq: str, target_seq: str, seed: int
    ) -> Dict[str, float]:
        """Run Chai-1 structure prediction for a binder-target complex."""
        import tempfile, pathlib

        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_path = pathlib.Path(tmpdir) / "input.fasta"
            fasta_path.write_text(
                f">binder\n{binder_seq}\n>target\n{target_seq}\n"
            )
            output_dir = pathlib.Path(tmpdir) / "output"
            output_dir.mkdir()

            candidates = run_inference(
                fasta_file=fasta_path,
                output_dir=output_dir,
                num_trunk_recycles=3,
                num_diffn_timesteps=200,
                seed=seed,
                device=self.device,
                use_esm_embeddings=True,
            )

            # Extract metrics from first candidate
            cif_scores = candidates[0].ranking_data
            iptm = float(cif_scores.interface_ptm)
            plddt_complex = float(cif_scores.aggregate_score)

            # Compute per-chain pLDDT
            binder_plddt = _extract_chain_plddt(
                candidates[0].structure, chain_idx=0, binder_length=len(binder_seq)
            )

            # Interface RMSD vs input backbone (if available)
            interface_rmsd = 99.9   # computed separately if backbone available
            pae_inter = float(getattr(cif_scores, "inter_chain_pae", 10.0))

        return {
            "iptm": iptm,
            "plddt_binder": binder_plddt,
            "plddt_complex": plddt_complex,
            "interface_rmsd": interface_rmsd,
            "pae_inter": pae_inter,
        }

    def _predict_esmfold(
        self, binder_seq: str, target_seq: str, seed: int
    ) -> Dict[str, float]:
        """Run ESMFold for structure prediction (single-chain only; use as proxy)."""
        if self._esmfold_model is None:
            print("[evaluate] Loading ESMFold model …")
            self._esmfold_model = EsmForProteinFolding.from_pretrained(
                "facebook/esmfold_v1"
            ).to(self.device)
            self._esmfold_model.eval()
            self._esmfold_tokenizer = AutoTokenizer.from_pretrained(
                "facebook/esmfold_v1"
            )

        with torch.no_grad():
            # ESMFold is single-chain; concatenate with linker for rough proxy
            linker = "G" * 25
            combined = binder_seq + linker + target_seq
            tokenized = self._esmfold_tokenizer(
                [combined], return_tensors="pt", add_special_tokens=False
            ).to(self.device)
            output = self._esmfold_model(**tokenized)

        plddt_all = output.plddt[0].cpu().numpy()
        binder_len = len(binder_seq)
        plddt_binder = float(plddt_all[:binder_len].mean())
        plddt_complex = float(plddt_all.mean())

        # ESMFold doesn't give iPTM; use cross-chain pLDDT as a proxy
        iptm_proxy = plddt_complex / 100.0

        return {
            "iptm": iptm_proxy,
            "plddt_binder": plddt_binder * 100.0,
            "plddt_complex": plddt_complex * 100.0,
            "interface_rmsd": 99.9,   # not computable from ESMFold alone
            "pae_inter": 10.0,        # not available from ESMFold
        }

    def _predict_mock(
        self, binder_seq: str, target_seq: str, seed: int
    ) -> Dict[str, float]:
        """
        Generate synthetic but realistic-looking metrics for pipeline testing.
        Scores are correlated with sequence properties to simulate a real signal.
        """
        rng = np.random.default_rng(seed=hash(binder_seq + str(seed)) % (2**32))

        # Compute basic sequence features that correlate with real scores
        seq_len = len(binder_seq)
        helix_residues = sum(1 for aa in binder_seq if aa in "AELM")
        helix_frac = helix_residues / max(seq_len, 1)

        # Simulate iPTM: higher for helix-rich, medium-length binders
        length_bonus = np.exp(-((seq_len - 65) ** 2) / (2 * 20 ** 2))
        iptm_base = 0.40 + 0.25 * helix_frac + 0.10 * length_bonus
        iptm = float(np.clip(rng.normal(iptm_base, 0.08), 0.0, 1.0))

        plddt_binder = float(np.clip(rng.normal(65 + 20 * helix_frac, 7.0), 0, 100))
        plddt_complex = float(np.clip(rng.normal(62 + 18 * helix_frac, 8.0), 0, 100))
        interface_rmsd = float(np.clip(rng.exponential(scale=1.5), 0.3, 10.0))
        pae_inter = float(np.clip(rng.normal(8.0 - 5.0 * iptm_base, 2.0), 1.0, 20.0))

        return {
            "iptm": iptm,
            "plddt_binder": plddt_binder,
            "plddt_complex": plddt_complex,
            "interface_rmsd": interface_rmsd,
            "pae_inter": pae_inter,
        }

    # ------------------------------------------------------------------ #
    # Filtering
    # ------------------------------------------------------------------ #

    def _apply_filters(self, result: EvaluationResult) -> EvaluationResult:
        """Apply per-metric filter thresholds to a single result."""
        result.passes_iptm = result.iptm >= self.thresholds["iptm_min"]
        result.passes_plddt = result.plddt_binder >= self.thresholds["plddt_min"]
        result.passes_rmsd = result.interface_rmsd <= self.thresholds["interface_rmsd_max"]
        result.passes_pae = result.pae_inter <= self.thresholds["pae_max"]
        result.passes_all = (
            result.passes_iptm
            and result.passes_plddt
            and result.passes_rmsd
            and result.passes_pae
        )
        return result

    def apply_filters(
        self,
        df: pd.DataFrame,
        thresholds: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        (Re-)apply filters to an evaluation DataFrame.
        Useful for sweeping threshold values in Phase 4 sensitivity analysis.

        Parameters
        ----------
        df : pd.DataFrame
            Output of evaluate().
        thresholds : dict, optional
            Override self.thresholds for this call.

        Returns
        -------
        DataFrame with filter columns updated; passes_all as summary.
        """
        t = {**self.thresholds, **(thresholds or {})}
        df = df.copy()
        df["passes_iptm"] = df["iptm"] >= t["iptm_min"]
        df["passes_plddt"] = df["plddt_binder"] >= t["plddt_min"]
        df["passes_rmsd"] = df["interface_rmsd"] <= t["interface_rmsd_max"]
        df["passes_pae"] = df["pae_inter"] <= t["pae_max"]
        df["passes_all"] = (
            df["passes_iptm"] & df["passes_plddt"] & df["passes_rmsd"] & df["passes_pae"]
        )
        return df

    def hit_rate(self, df: pd.DataFrame) -> float:
        """Fraction of designs passing all filters."""
        return df["passes_all"].mean()

    # ------------------------------------------------------------------ #
    # Interface geometry (Phase 4 input)
    # ------------------------------------------------------------------ #

    def compute_interface_contacts(
        self,
        binder_pdb: Path,
        target_pdb: Path,
        cutoff_angstrom: float = 5.0,
    ) -> Dict[str, int]:
        """
        Count interface contacts between binder and target PDB files.

        Returns dict with keys: n_contacts, n_hbonds_approx, n_hydrophobic
        (hydrogen bonds are approximated by N-O / O-N pairs within 3.5 Å).
        """
        if not HAS_BIOPYTHON:
            return {"n_contacts": 0, "n_hbonds_approx": 0, "n_hydrophobic": 0}

        parser = PDBParser(QUIET=True)
        binder_struct = parser.get_structure("binder", str(binder_pdb))
        target_struct = parser.get_structure("target", str(target_pdb))

        binder_atoms = list(binder_struct.get_atoms())
        target_atoms = list(target_struct.get_atoms())

        n_contacts = 0
        n_hbonds = 0
        n_hydrophobic = 0

        hydrophobic_atoms = {"C", "S"}
        hbond_atoms = {"N", "O"}

        for ba in binder_atoms:
            bv = ba.get_vector().get_array()
            for ta in target_atoms:
                tv = ta.get_vector().get_array()
                dist = np.linalg.norm(bv - tv)
                if dist <= cutoff_angstrom:
                    n_contacts += 1
                    if (ba.element in hbond_atoms and ta.element in hbond_atoms
                            and dist <= 3.5):
                        n_hbonds += 1
                    if (ba.element in hydrophobic_atoms
                            and ta.element in hydrophobic_atoms):
                        n_hydrophobic += 1

        return {
            "n_contacts": n_contacts,
            "n_hbonds_approx": n_hbonds,
            "n_hydrophobic": n_hydrophobic,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _resolve_backend(self, backend: str) -> str:
        if backend == "chai1" and not HAS_CHAI1:
            print("[evaluate] Chai-1 not installed (pip install chai_lab). Falling back to mock.")
            return "mock"
        if backend == "esmfold" and not HAS_ESMFOLD:
            print("[evaluate] ESMFold transformers not installed. Falling back to mock.")
            return "mock"
        if backend not in ("chai1", "esmfold", "mock"):
            raise ValueError(f"Unknown backend '{backend}'. Choose: chai1, esmfold, mock.")
        return backend

    @staticmethod
    def _df_to_candidates(df: pd.DataFrame) -> List[BinderCandidate]:
        """Convert a generation DataFrame back to BinderCandidate objects."""
        from src.generate import BinderCandidate
        candidates = []
        for _, row in df.iterrows():
            cand = BinderCandidate(
                candidate_id=str(row.get("candidate_id", "unknown")),
                target_name=str(row.get("target_name", "")),
                backbone_pdb=Path(str(row.get("backbone_pdb", ""))),
                sequence=str(row["sequence"]),
                binder_length=int(row.get("binder_length", len(str(row["sequence"])))),
                proteinmpnn_score=float(row.get("proteinmpnn_score", 0.0)),
            )
            candidates.append(cand)
        return candidates


# ------------------------------------------------------------------ #
# Utility functions
# ------------------------------------------------------------------ #

def _extract_chain_plddt(structure, chain_idx: int, binder_length: int) -> float:
    """Extract mean pLDDT for a chain from Chai-1 output structure."""
    try:
        # Chai-1 returns an AtomArray; iterate over binder residues
        binder_mask = structure.chain_id == structure.chain_id[0]
        plddt = structure.b_factor[binder_mask][:binder_length]
        return float(plddt.mean())
    except Exception:
        return 70.0


def compute_interface_rmsd(
    generated_pdb: Path,
    predicted_pdb: Path,
    binder_chain: str = "B",
) -> float:
    """
    Compute RMSD between binder in the generated backbone PDB and the
    re-predicted complex structure.

    Uses CA atoms of the binder chain after superimposing on the target chain.

    Parameters
    ----------
    generated_pdb : Path
        Original RFdiffusion-generated backbone.
    predicted_pdb : Path
        Structure predicted by Chai-1 / ESMFold.
    binder_chain : str
        Chain ID of the binder in both structures.

    Returns
    -------
    float : RMSD in Angstroms
    """
    if not HAS_BIOPYTHON:
        return 99.9

    parser = PDBParser(QUIET=True)
    gen = parser.get_structure("gen", str(generated_pdb))
    pred = parser.get_structure("pred", str(predicted_pdb))

    def get_ca_atoms(struct, chain_id):
        atoms = []
        try:
            for res in struct[0][chain_id].get_residues():
                if is_aa(res, standard=True) and "CA" in res:
                    atoms.append(res["CA"])
        except KeyError:
            pass
        return atoms

    gen_cas = get_ca_atoms(gen, binder_chain)
    pred_cas = get_ca_atoms(pred, binder_chain)

    n = min(len(gen_cas), len(pred_cas))
    if n < 3:
        return 99.9

    sup = Superimposer()
    sup.set_atoms(gen_cas[:n], pred_cas[:n])
    return float(sup.rms)


def threshold_sweep(
    df: pd.DataFrame,
    metric: str = "iptm",
    thresholds: Optional[np.ndarray] = None,
    evaluator: Optional["BinderEvaluator"] = None,
) -> pd.DataFrame:
    """
    Compute hit rate as a function of a swept threshold value.
    Used in Phase 4 sensitivity analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Output of BinderEvaluator.evaluate().
    metric : str
        Column name to sweep ("iptm", "plddt_binder", "interface_rmsd").
    thresholds : array, optional
        Threshold values to sweep. Auto-generated from data range if None.
    evaluator : BinderEvaluator, optional
        If provided, the other filters are kept fixed while sweeping metric.

    Returns
    -------
    DataFrame with columns [threshold, hit_rate, n_passing].
    """
    col_values = df[metric].dropna()
    if thresholds is None:
        thresholds = np.linspace(col_values.min(), col_values.max(), 50)

    rows = []
    for t in thresholds:
        if metric.endswith("_max") or metric in ("interface_rmsd", "pae_inter"):
            mask = df[metric] <= t
        else:
            mask = df[metric] >= t
        n_passing = int(mask.sum())
        rows.append({
            "threshold": t,
            "hit_rate": n_passing / len(df),
            "n_passing": n_passing,
        })
    return pd.DataFrame(rows)
