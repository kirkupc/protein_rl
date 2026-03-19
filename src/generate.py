"""
Phase 2 — Binder Generation

Wrappers around RFdiffusion (backbone generation) and ProteinMPNN (inverse
folding / sequence design). Falls back gracefully when tools are not installed.

Workflow:
    1. RFdiffusion: generate N backbone structures conditioned on hotspot residues.
    2. ProteinMPNN: for each backbone, sample K sequences.
    3. Write results to data/generated/<target_name>/.

Usage:
    from src.generate import BinderGenerator
    gen = BinderGenerator(target, output_dir="data/generated/EGFR")
    backbones = gen.run_rfdiffusion(num_designs=50)
    candidates = gen.run_proteinmpnn(backbones, seqs_per_backbone=8)
    df = gen.to_dataframe(candidates)
    df.to_csv("data/generated/EGFR/candidates.csv", index=False)
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.target_prep import TargetProtein

GENERATED_DIR = Path(__file__).parent.parent / "data" / "generated"


@dataclass
class BinderCandidate:
    """Container for a single (backbone, sequence) binder candidate."""
    candidate_id: str
    target_name: str
    backbone_pdb: Path
    sequence: str
    binder_length: int
    proteinmpnn_score: float = 0.0   # ProteinMPNN log-probability score
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "candidate_id": self.candidate_id,
            "target_name": self.target_name,
            "backbone_pdb": str(self.backbone_pdb),
            "sequence": self.sequence,
            "binder_length": self.binder_length,
            "proteinmpnn_score": self.proteinmpnn_score,
            **self.metadata,
        }


class BinderGenerator:
    """
    Orchestrates RFdiffusion + ProteinMPNN for binder candidate generation.

    If external tools are not installed, synthetic dummy data is generated for
    pipeline testing (useful when focusing on Phase 3/5 without GPU access).

    Parameters
    ----------
    target : TargetProtein
        Parsed target with hotspot residues identified.
    output_dir : Path
        Directory to store generated backbones and sequences.
    rfdiffusion_dir : Path, optional
        Path to cloned RFdiffusion repository.
    proteinmpnn_dir : Path, optional
        Path to cloned ProteinMPNN repository.
    """

    def __init__(
        self,
        target: TargetProtein,
        output_dir: Optional[Path] = None,
        rfdiffusion_dir: Optional[Path] = None,
        proteinmpnn_dir: Optional[Path] = None,
    ):
        self.target = target
        self.output_dir = Path(output_dir or GENERATED_DIR / target.pdb_id)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.rfdiffusion_dir = Path(rfdiffusion_dir) if rfdiffusion_dir else None
        self.proteinmpnn_dir = Path(proteinmpnn_dir) if proteinmpnn_dir else None

        self._rfdiffusion_available = self._check_rfdiffusion()
        self._proteinmpnn_available = self._check_proteinmpnn()

        if not self._rfdiffusion_available:
            print(
                "[generate] RFdiffusion not found. Synthetic backbones will be used. "
                "Set rfdiffusion_dir= to point to a cloned RFdiffusion repo."
            )
        if not self._proteinmpnn_available:
            print(
                "[generate] ProteinMPNN not found. Synthetic sequences will be used. "
                "Set proteinmpnn_dir= to point to a cloned ProteinMPNN repo."
            )

    # ------------------------------------------------------------------ #
    # Step 1: RFdiffusion
    # ------------------------------------------------------------------ #

    def run_rfdiffusion(
        self,
        num_designs: int = 50,
        diffusion_steps: int = 50,
        noise_scale: float = 1.0,
        binder_length: int = 70,
    ) -> List[Path]:
        """
        Generate backbone PDB files using RFdiffusion.

        Returns list of paths to generated backbone PDB files.
        """
        print(
            f"[generate] RFdiffusion: generating {num_designs} backbones "
            f"for {self.target.pdb_id} …"
        )

        if self._rfdiffusion_available:
            return self._run_rfdiffusion_real(
                num_designs, diffusion_steps, noise_scale, binder_length
            )
        else:
            return self._run_rfdiffusion_synthetic(num_designs, binder_length)

    def _run_rfdiffusion_real(
        self,
        num_designs: int,
        diffusion_steps: int,
        noise_scale: float,
        binder_length: int,
    ) -> List[Path]:
        """Invoke RFdiffusion via subprocess."""
        clean_pdb = self.target.save_clean_pdb()
        hotspot_str = self._format_hotspots_rfdiffusion()
        output_prefix = self.output_dir / "backbone"

        cmd = [
            "python",
            str(self.rfdiffusion_dir / "scripts" / "run_inference.py"),
            f"inference.input_pdb={clean_pdb}",
            f"inference.output_prefix={output_prefix}",
            f"inference.num_designs={num_designs}",
            f"diffuser.T={diffusion_steps}",
            f"denoiser.noise_scale_ca={noise_scale}",
            f"denoiser.noise_scale_frame={noise_scale}",
            f"contigmap.contigs=[{self.target.chain_id}1-{self.target.n_residues}/0 {binder_length}-{binder_length}]",
            f"ppi.hotspot_res=[{hotspot_str}]",
        ]
        print(f"[generate] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"RFdiffusion failed:\n{result.stderr}")

        backbones = sorted(self.output_dir.glob("backbone_*.pdb"))
        print(f"[generate] RFdiffusion produced {len(backbones)} backbones.")
        return backbones

    def _run_rfdiffusion_synthetic(
        self, num_designs: int, binder_length: int
    ) -> List[Path]:
        """Generate synthetic backbone PDB files for testing."""
        print(f"[generate] Generating {num_designs} synthetic backbones …")
        backbones = []
        for i in range(num_designs):
            pdb_path = self.output_dir / f"backbone_{i:04d}.pdb"
            _write_synthetic_pdb(pdb_path, binder_length, chain_id="B")
            backbones.append(pdb_path)
        print(f"[generate] Wrote {len(backbones)} synthetic backbone PDBs.")
        return backbones

    def _format_hotspots_rfdiffusion(self) -> str:
        """Format hotspot residue numbers for RFdiffusion config string."""
        return ",".join(
            f"{self.target.chain_id}{r}" for r in self.target.hotspot_residue_numbers()
        )

    # ------------------------------------------------------------------ #
    # Step 2: ProteinMPNN
    # ------------------------------------------------------------------ #

    def run_proteinmpnn(
        self,
        backbone_pdbs: List[Path],
        seqs_per_backbone: int = 8,
        temperature: float = 0.1,
    ) -> List[BinderCandidate]:
        """
        Design sequences for each backbone using ProteinMPNN.

        Returns list of BinderCandidate objects.
        """
        print(
            f"[generate] ProteinMPNN: designing {seqs_per_backbone} sequences "
            f"per backbone ({len(backbone_pdbs)} backbones) …"
        )

        if self._proteinmpnn_available:
            return self._run_proteinmpnn_real(
                backbone_pdbs, seqs_per_backbone, temperature
            )
        else:
            return self._run_proteinmpnn_synthetic(
                backbone_pdbs, seqs_per_backbone, temperature
            )

    def _run_proteinmpnn_real(
        self,
        backbone_pdbs: List[Path],
        seqs_per_backbone: int,
        temperature: float,
    ) -> List[BinderCandidate]:
        """Run ProteinMPNN via subprocess for each backbone."""
        candidates: List[BinderCandidate] = []

        for backbone_pdb in backbone_pdbs:
            jsonl_path = self.output_dir / "parsed_pdbs.jsonl"
            chains_jsonl = self.output_dir / "assigned_pdbs.jsonl"
            out_dir = self.output_dir / "mpnn_outputs"
            out_dir.mkdir(exist_ok=True)

            # Step 1: parse PDB
            parse_cmd = [
                "python",
                str(self.proteinmpnn_dir / "helper_scripts" / "parse_multiple_chains.py"),
                f"--input_path={backbone_pdb.parent}",
                f"--output_path={jsonl_path}",
            ]
            subprocess.run(parse_cmd, check=True, capture_output=True)

            # Step 2: assign chains to design
            assign_cmd = [
                "python",
                str(self.proteinmpnn_dir / "helper_scripts" / "assign_fixed_chains.py"),
                f"--input_path={jsonl_path}",
                f"--output_path={chains_jsonl}",
                "--chain_list=B",   # design binder chain B
            ]
            subprocess.run(assign_cmd, check=True, capture_output=True)

            # Step 3: run ProteinMPNN
            mpnn_cmd = [
                "python",
                str(self.proteinmpnn_dir / "protein_mpnn_run.py"),
                f"--jsonl_path={jsonl_path}",
                f"--chain_id_jsonl={chains_jsonl}",
                f"--out_folder={out_dir}",
                f"--num_seq_per_target={seqs_per_backbone}",
                f"--sampling_temp={temperature}",
                "--seed=37",
                "--batch_size=1",
            ]
            result = subprocess.run(mpnn_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[generate] ProteinMPNN failed for {backbone_pdb}: {result.stderr}")
                continue

            # Parse FASTA output
            fasta_path = out_dir / "seqs" / f"{backbone_pdb.stem}.fa"
            if fasta_path.exists():
                seqs = _parse_mpnn_fasta(fasta_path)
                for seq, score in seqs[:seqs_per_backbone]:
                    cand = BinderCandidate(
                        candidate_id=str(uuid.uuid4())[:8],
                        target_name=self.target.pdb_id,
                        backbone_pdb=backbone_pdb,
                        sequence=seq,
                        binder_length=len(seq),
                        proteinmpnn_score=score,
                    )
                    candidates.append(cand)

        print(f"[generate] ProteinMPNN produced {len(candidates)} candidates.")
        return candidates

    def _run_proteinmpnn_synthetic(
        self,
        backbone_pdbs: List[Path],
        seqs_per_backbone: int,
        temperature: float,
    ) -> List[BinderCandidate]:
        """Generate synthetic sequences for testing."""
        print(f"[generate] Generating synthetic sequences (T={temperature}) …")
        candidates = []
        rng = np.random.default_rng(seed=42)
        aa_alphabet = list("ACDEFGHIKLMNPQRSTVWY")

        for backbone_pdb in backbone_pdbs:
            # Read backbone length from filename or use default
            binder_length = _estimate_binder_length(backbone_pdb)

            for _ in range(seqs_per_backbone):
                # Sample sequence with temperature-scaled amino acid frequencies
                seq = _sample_sequence(rng, binder_length, temperature, aa_alphabet)
                score = float(rng.normal(loc=-1.5, scale=0.5))  # realistic mpnn score
                cand = BinderCandidate(
                    candidate_id=str(uuid.uuid4())[:8],
                    target_name=self.target.pdb_id,
                    backbone_pdb=backbone_pdb,
                    sequence=seq,
                    binder_length=len(seq),
                    proteinmpnn_score=score,
                    metadata={"synthetic": True, "temperature": temperature},
                )
                candidates.append(cand)

        print(f"[generate] Generated {len(candidates)} synthetic candidates.")
        return candidates

    # ------------------------------------------------------------------ #
    # I/O
    # ------------------------------------------------------------------ #

    def to_dataframe(self, candidates: List[BinderCandidate]) -> pd.DataFrame:
        """Convert list of BinderCandidates to a DataFrame."""
        return pd.DataFrame([c.to_dict() for c in candidates])

    def save_candidates(self, candidates: List[BinderCandidate]) -> Path:
        """Save candidates CSV to output directory."""
        df = self.to_dataframe(candidates)
        csv_path = self.output_dir / "candidates.csv"
        df.to_csv(csv_path, index=False)
        print(f"[generate] Saved {len(candidates)} candidates to {csv_path}")
        return csv_path

    def load_candidates(self, csv_path: Optional[Path] = None) -> List[BinderCandidate]:
        """Load candidates from a saved CSV."""
        csv_path = csv_path or self.output_dir / "candidates.csv"
        df = pd.read_csv(csv_path)
        candidates = []
        for _, row in df.iterrows():
            cand = BinderCandidate(
                candidate_id=row["candidate_id"],
                target_name=row["target_name"],
                backbone_pdb=Path(row["backbone_pdb"]),
                sequence=row["sequence"],
                binder_length=int(row["binder_length"]),
                proteinmpnn_score=float(row.get("proteinmpnn_score", 0.0)),
            )
            candidates.append(cand)
        return candidates

    # ------------------------------------------------------------------ #
    # Availability checks
    # ------------------------------------------------------------------ #

    def _check_rfdiffusion(self) -> bool:
        if self.rfdiffusion_dir is None:
            return False
        script = self.rfdiffusion_dir / "scripts" / "run_inference.py"
        return script.exists()

    def _check_proteinmpnn(self) -> bool:
        if self.proteinmpnn_dir is None:
            return False
        script = self.proteinmpnn_dir / "protein_mpnn_run.py"
        return script.exists()


# ------------------------------------------------------------------ #
# Synthetic data helpers
# ------------------------------------------------------------------ #

def _write_synthetic_pdb(path: Path, length: int, chain_id: str = "B") -> None:
    """Write a minimal single-chain PDB file with a helical backbone."""
    rng = np.random.default_rng(seed=int(path.stem.split("_")[-1]) if "_" in path.stem else 0)
    lines = []

    # Parameterise a simple alpha-helix
    for i in range(length):
        angle = i * (100.0 * np.pi / 180.0)   # ~100° per residue for helix
        x = 2.3 * np.cos(angle)
        y = 2.3 * np.sin(angle)
        z = 1.5 * i
        # Add small random perturbation
        x += rng.normal(0, 0.1)
        y += rng.normal(0, 0.1)
        z += rng.normal(0, 0.05)

        atom_num = i + 1
        res_num = i + 1
        lines.append(
            f"ATOM  {atom_num:5d}  CA  ALA {chain_id}{res_num:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C  "
        )

    lines.append("END")
    path.write_text("\n".join(lines))


def _sample_sequence(
    rng: np.random.Generator,
    length: int,
    temperature: float,
    aa_alphabet: List[str],
) -> str:
    """Sample a random amino-acid sequence (helical bias at low temperature)."""
    # Helical-preferred residues get higher weight at low temperature
    helix_preferred = set("AELM")
    weights = np.array([
        2.0 if aa in helix_preferred else 1.0 for aa in aa_alphabet
    ])
    # At high temperature, weights flatten
    weights = weights ** (1.0 / max(temperature, 0.01))
    weights /= weights.sum()
    return "".join(rng.choice(aa_alphabet, size=length, p=weights))


def _estimate_binder_length(backbone_pdb: Path) -> int:
    """Try to read binder length from PDB CA atom count; fallback to 70."""
    try:
        ca_lines = [
            line for line in backbone_pdb.read_text().splitlines()
            if line.startswith("ATOM") and " CA " in line
        ]
        return max(len(ca_lines), 40)
    except Exception:
        return 70


def _parse_mpnn_fasta(fasta_path: Path) -> List[Tuple[str, float]]:
    """Parse ProteinMPNN FASTA output → list of (sequence, score) tuples."""
    seqs = []
    current_score = 0.0
    current_seq = ""
    for line in fasta_path.read_text().splitlines():
        if line.startswith(">"):
            if current_seq:
                seqs.append((current_seq, current_score))
                current_seq = ""
            # Extract score from header: >T=0.1, score=1.2345, ...
            for part in line.split(","):
                if "score=" in part:
                    try:
                        current_score = float(part.split("=")[1])
                    except ValueError:
                        current_score = 0.0
        else:
            current_seq += line.strip()
    if current_seq:
        seqs.append((current_seq, current_score))
    return seqs
