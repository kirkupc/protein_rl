"""
Phase 1 — Target Preparation & Structural Analysis

Downloads PDB structures, parses chains/residues/atoms, identifies hotspot
residues at the binding interface, and computes basic biophysical properties.

Usage:
    from src.target_prep import TargetProtein
    target = TargetProtein("3NJP", chain_id="A")
    target.download_pdb()
    target.parse_structure()
    target.identify_hotspots()
    df = target.to_dataframe()
"""

from __future__ import annotations

import io
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from Bio import SeqIO
    from Bio.PDB import DSSP, PDBParser, PDBIO, Select
    from Bio.PDB.Polypeptide import is_aa
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False
    print("BioPython not installed. Install with: pip install biopython==1.84")

# Benchmark targets from Latent-X paper Table 1
BENCHMARK_TARGETS: Dict[str, Dict] = {
    "IL7RA":    {"pdb_id": "3DI2", "chain_id": "A", "name": "IL-7 Receptor alpha"},
    "TrkA":     {"pdb_id": "2IFG", "chain_id": "A", "name": "TrkA Neurotrophin Receptor"},
    "InsulinR": {"pdb_id": "7PG0", "chain_id": "A", "name": "Insulin Receptor"},
    "EGFR":     {"pdb_id": "3NJP", "chain_id": "A", "name": "Epidermal Growth Factor Receptor"},
    "FGFR2":    {"pdb_id": "1DJS", "chain_id": "A", "name": "Fibroblast Growth Factor Receptor 2"},
}

PDB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
DATA_DIR = Path(__file__).parent.parent / "data" / "targets"


@dataclass
class ResidueInfo:
    """Container for per-residue properties."""
    chain_id: str
    res_seq: int
    res_name: str
    one_letter: str
    is_hotspot: bool = False
    dssp_ss: str = ""         # H=helix, E=strand, C=coil
    sasa: float = 0.0         # solvent accessible surface area
    coords: Optional[np.ndarray] = None  # CA coordinates


@dataclass
class TargetProtein:
    """
    Represents a target protein structure with interface analysis.

    Parameters
    ----------
    pdb_id : str
        4-character PDB accession (e.g. "3NJP").
    chain_id : str
        Chain to use as the target (e.g. "A").
    hotspot_cutoff : float
        Distance threshold (Angstroms) to define interface residues.
    data_dir : Path
        Directory to store downloaded PDB files.
    """

    pdb_id: str
    chain_id: str = "A"
    hotspot_cutoff: float = 8.0
    data_dir: Path = DATA_DIR

    # Populated after parse_structure()
    residues: List[ResidueInfo] = field(default_factory=list)
    sequence: str = ""
    n_residues: int = 0

    # Populated after identify_hotspots()
    hotspot_residues: List[ResidueInfo] = field(default_factory=list)

    # Raw BioPython structure
    _structure: object = field(default=None, repr=False)
    _chain: object = field(default=None, repr=False)

    # ------------------------------------------------------------------ #
    # I/O
    # ------------------------------------------------------------------ #

    @property
    def pdb_path(self) -> Path:
        return self.data_dir / f"{self.pdb_id.upper()}.pdb"

    def download_pdb(self, force: bool = False) -> Path:
        """Download PDB file from RCSB. Skips if already present."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.pdb_path.exists() and not force:
            print(f"[target_prep] {self.pdb_id}: already downloaded at {self.pdb_path}")
            return self.pdb_path

        url = PDB_DOWNLOAD_URL.format(pdb_id=self.pdb_id.upper())
        print(f"[target_prep] Downloading {self.pdb_id} from {url} …")
        urllib.request.urlretrieve(url, self.pdb_path)
        print(f"[target_prep] Saved to {self.pdb_path}")
        return self.pdb_path

    # ------------------------------------------------------------------ #
    # Structure parsing
    # ------------------------------------------------------------------ #

    def parse_structure(self) -> "TargetProtein":
        """
        Parse the PDB file with BioPython.
        Extracts residues, CA coordinates, secondary structure (DSSP).
        """
        if not HAS_BIOPYTHON:
            raise RuntimeError("BioPython required. pip install biopython==1.84")

        if not self.pdb_path.exists():
            self.download_pdb()

        parser = PDBParser(QUIET=True)
        self._structure = parser.get_structure(self.pdb_id, str(self.pdb_path))
        model = self._structure[0]

        if self.chain_id not in [c.id for c in model.get_chains()]:
            available = [c.id for c in model.get_chains()]
            raise ValueError(
                f"Chain '{self.chain_id}' not in {self.pdb_id}. "
                f"Available chains: {available}"
            )

        self._chain = model[self.chain_id]

        # Try DSSP for secondary structure + SASA
        dssp_data: Dict[Tuple, Tuple] = {}
        try:
            dssp = DSSP(model, str(self.pdb_path))
            dssp_data = dict(dssp)
        except Exception as exc:
            print(f"[target_prep] DSSP failed (install mkdssp): {exc}")

        # Build residue list
        self.residues = []
        aa_sequence = []

        for residue in self._chain.get_residues():
            if not is_aa(residue, standard=True):
                continue

            res_name = residue.resname.strip()
            one_letter = _three_to_one(res_name)
            aa_sequence.append(one_letter)

            # CA coordinates
            coords = None
            if "CA" in residue:
                coords = residue["CA"].get_vector().get_array()

            # DSSP lookup
            dssp_key = (self.chain_id, residue.id)
            dssp_ss, sasa = "", 0.0
            if dssp_key in dssp_data:
                dssp_ss = dssp_data[dssp_key][2]
                sasa = dssp_data[dssp_key][3] or 0.0

            res_info = ResidueInfo(
                chain_id=self.chain_id,
                res_seq=residue.id[1],
                res_name=res_name,
                one_letter=one_letter,
                dssp_ss=dssp_ss,
                sasa=float(sasa),
                coords=coords,
            )
            self.residues.append(res_info)

        self.sequence = "".join(aa_sequence)
        self.n_residues = len(self.residues)
        print(
            f"[target_prep] Parsed {self.pdb_id} chain {self.chain_id}: "
            f"{self.n_residues} residues"
        )
        return self

    # ------------------------------------------------------------------ #
    # Hotspot identification
    # ------------------------------------------------------------------ #

    def identify_hotspots(
        self,
        partner_chain_id: Optional[str] = None,
        manual_hotspots: Optional[List[int]] = None,
    ) -> "TargetProtein":
        """
        Identify interface / hotspot residues.

        Strategy (in priority order):
        1. If manual_hotspots is given, use those residue numbers.
        2. If a partner chain is present in the structure, compute residues
           within self.hotspot_cutoff Å of the partner chain's atoms.
        3. Fall back to high-SASA surface residues as a proxy.

        Parameters
        ----------
        partner_chain_id : str, optional
            Chain ID of the binding partner (e.g. "B" for a co-crystal).
        manual_hotspots : list of int, optional
            Explicit list of residue sequence numbers to mark as hotspots.
        """
        if not self.residues:
            raise RuntimeError("Call parse_structure() first.")

        if manual_hotspots:
            hotspot_set = set(manual_hotspots)
            for res in self.residues:
                res.is_hotspot = res.res_seq in hotspot_set
        elif partner_chain_id:
            self._hotspots_from_partner(partner_chain_id)
        else:
            # Auto-detect: if another chain is present, use it
            model = self._structure[0]
            other_chains = [
                c.id for c in model.get_chains() if c.id != self.chain_id
            ]
            if other_chains:
                print(
                    f"[target_prep] Auto-detecting hotspots from partner chain "
                    f"'{other_chains[0]}'"
                )
                self._hotspots_from_partner(other_chains[0])
            else:
                print(
                    "[target_prep] No partner chain found. "
                    "Falling back to surface residue proxy (top-25% SASA)."
                )
                self._hotspots_from_sasa()

        self.hotspot_residues = [r for r in self.residues if r.is_hotspot]
        print(
            f"[target_prep] {len(self.hotspot_residues)} hotspot residues identified "
            f"(cutoff={self.hotspot_cutoff} Å)"
        )
        return self

    def _hotspots_from_partner(self, partner_chain_id: str) -> None:
        """Mark residues within cutoff distance of any atom in the partner chain."""
        model = self._structure[0]
        if partner_chain_id not in [c.id for c in model.get_chains()]:
            raise ValueError(f"Partner chain '{partner_chain_id}' not found in structure.")

        partner_atoms = list(model[partner_chain_id].get_atoms())

        for res in self.residues:
            if res.coords is None:
                continue
            ca_vec = res.coords
            for atom in partner_atoms:
                dist = np.linalg.norm(ca_vec - atom.get_vector().get_array())
                if dist <= self.hotspot_cutoff:
                    res.is_hotspot = True
                    break

    def _hotspots_from_sasa(self, top_fraction: float = 0.25) -> None:
        """Mark top SASA residues as putative surface / hotspot residues."""
        sasa_vals = [r.sasa for r in self.residues if r.sasa > 0]
        if not sasa_vals:
            print("[target_prep] No SASA data; marking all as hotspot candidates.")
            for r in self.residues:
                r.is_hotspot = True
            return

        threshold = np.percentile(sasa_vals, 100 * (1 - top_fraction))
        for r in self.residues:
            r.is_hotspot = r.sasa >= threshold

    # ------------------------------------------------------------------ #
    # Biophysical properties
    # ------------------------------------------------------------------ #

    def compute_sequence_properties(self) -> Dict[str, float]:
        """
        Compute basic sequence-level properties using BioPython.
        Returns a dict of properties (MW, charge, hydrophobicity, etc.)
        """
        if not self.sequence:
            raise RuntimeError("Call parse_structure() first.")

        # Replace 'X' with 'G' to avoid ProteinAnalysis errors
        clean_seq = self.sequence.replace("X", "G").replace("U", "C")

        analysis = ProteinAnalysis(clean_seq)
        props = {
            "length": len(self.sequence),
            "molecular_weight": analysis.molecular_weight(),
            "isoelectric_point": analysis.isoelectric_point(),
            "instability_index": analysis.instability_index(),
            "gravy": analysis.gravy(),   # grand average of hydropathicity
            "aromaticity": analysis.aromaticity(),
            "helix_fraction": analysis.secondary_structure_fraction()[0],
            "turn_fraction": analysis.secondary_structure_fraction()[1],
            "sheet_fraction": analysis.secondary_structure_fraction()[2],
        }

        if self.residues:
            dssp_counts = {"H": 0, "E": 0, "C": 0, "-": 0}
            for r in self.residues:
                key = r.dssp_ss if r.dssp_ss in dssp_counts else "-"
                dssp_counts[key] += 1
            n = len(self.residues)
            props["dssp_helix_frac"] = dssp_counts["H"] / n
            props["dssp_strand_frac"] = dssp_counts["E"] / n
            props["dssp_coil_frac"] = (dssp_counts["C"] + dssp_counts["-"]) / n

        return props

    def secondary_structure_summary(self) -> pd.DataFrame:
        """Return per-residue secondary structure assignment."""
        return pd.DataFrame(
            [
                {
                    "res_seq": r.res_seq,
                    "res_name": r.res_name,
                    "one_letter": r.one_letter,
                    "dssp_ss": r.dssp_ss,
                    "sasa": r.sasa,
                    "is_hotspot": r.is_hotspot,
                }
                for r in self.residues
            ]
        )

    # ------------------------------------------------------------------ #
    # Export helpers
    # ------------------------------------------------------------------ #

    def to_dataframe(self) -> pd.DataFrame:
        """Export all residue-level data as a DataFrame."""
        return self.secondary_structure_summary()

    def hotspot_coords(self) -> np.ndarray:
        """
        Return (N, 3) array of CA coordinates for hotspot residues.
        Used by generation pipeline to condition RFdiffusion.
        """
        coords = [
            r.coords for r in self.hotspot_residues if r.coords is not None
        ]
        if not coords:
            raise ValueError("No hotspot coordinates found. Run identify_hotspots() first.")
        return np.vstack(coords)

    def hotspot_residue_numbers(self) -> List[int]:
        """Return list of residue sequence numbers for hotspot residues."""
        return [r.res_seq for r in self.hotspot_residues]

    def save_clean_pdb(self, output_path: Optional[Path] = None) -> Path:
        """Save a cleaned PDB with only the target chain, ATOM records only."""
        if not HAS_BIOPYTHON:
            raise RuntimeError("BioPython required.")

        output_path = output_path or self.data_dir / f"{self.pdb_id}_{self.chain_id}_clean.pdb"

        class ChainSelect(Select):
            def __init__(self, chain_id: str):
                self.chain_id = chain_id

            def accept_chain(self, chain):
                return chain.id == self.chain_id

            def accept_residue(self, residue):
                return is_aa(residue, standard=True)

        io = PDBIO()
        io.set_structure(self._structure)
        io.save(str(output_path), ChainSelect(self.chain_id))
        print(f"[target_prep] Clean PDB saved to {output_path}")
        return output_path

    def visualise(self, highlight_hotspots: bool = True) -> str:
        """
        Return a py3Dmol view script string for notebook rendering.
        Call display(view) after py3Dmol.view(…) in a Jupyter cell.
        """
        hotspot_residue_nums = [str(r.res_seq) for r in self.hotspot_residues]
        hotspot_selection = "+".join(hotspot_residue_nums)

        script = f"""
// py3Dmol visualisation for {self.pdb_id} chain {self.chain_id}
// To use in Jupyter:
//   import py3Dmol, pathlib
//   view = py3Dmol.view(width=800, height=600)
//   view.addModel(pathlib.Path('{self.pdb_path}').read_text(), 'pdb')
//   view.setStyle({{'chain': '{self.chain_id}'}}, {{'cartoon': {{'color': 'spectrum'}}}})
//   view.addStyle({{'chain': '{self.chain_id}', 'resi': '{hotspot_selection}'}},
//                {{'sphere': {{'color': 'red', 'radius': 0.8}}}})
//   view.zoomTo()
//   view.show()
Hotspot residues: {hotspot_selection}
"""
        return script

    def __repr__(self) -> str:
        return (
            f"TargetProtein(pdb_id='{self.pdb_id}', chain='{self.chain_id}', "
            f"n_residues={self.n_residues}, n_hotspots={len(self.hotspot_residues)})"
        )


# ------------------------------------------------------------------ #
# Convenience functions
# ------------------------------------------------------------------ #

def load_benchmark_targets(
    target_names: Optional[List[str]] = None,
    data_dir: Path = DATA_DIR,
    download: bool = True,
) -> Dict[str, TargetProtein]:
    """
    Load all (or a subset of) the Latent-X benchmark targets.

    Parameters
    ----------
    target_names : list of str, optional
        Subset of BENCHMARK_TARGETS keys. Defaults to all.
    data_dir : Path
        Where to store PDB files.
    download : bool
        Whether to auto-download missing PDB files.

    Returns
    -------
    dict mapping target name → TargetProtein (parsed + hotspots identified)
    """
    if target_names is None:
        target_names = list(BENCHMARK_TARGETS.keys())

    targets = {}
    for name in target_names:
        if name not in BENCHMARK_TARGETS:
            raise ValueError(f"Unknown target '{name}'. Options: {list(BENCHMARK_TARGETS)}")
        meta = BENCHMARK_TARGETS[name]
        tp = TargetProtein(
            pdb_id=meta["pdb_id"],
            chain_id=meta["chain_id"],
            data_dir=data_dir,
        )
        if download:
            tp.download_pdb()
        tp.parse_structure()
        tp.identify_hotspots()
        targets[name] = tp
        print(f"[target_prep] Loaded: {tp}")

    return targets


def compute_pairwise_distances(
    coords_a: np.ndarray, coords_b: np.ndarray
) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between two sets of 3D coordinates.

    Parameters
    ----------
    coords_a : (N, 3) array
    coords_b : (M, 3) array

    Returns
    -------
    (N, M) distance matrix
    """
    diff = coords_a[:, None, :] - coords_b[None, :, :]   # (N, M, 3)
    return np.sqrt((diff ** 2).sum(axis=-1))


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #

_THREE_TO_ONE: Dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O",
}


def _three_to_one(three: str) -> str:
    return _THREE_TO_ONE.get(three.upper(), "X")
