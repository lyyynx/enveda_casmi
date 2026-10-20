"""In-silico MS2 fragmentation.

Enumerates candidate fragments of a molecule and matches the theoretical
fragment ions to observed MS2 peaks within a ppm tolerance.

Fragment generation:
  * single cuts of acyclic bonds (side-chain cleavages),
  * double cuts of ring bonds (cross-ring cleavages / ring fissions),
  * the intact precursor with neutral losses,
each combined with hydrogen rearrangements and neutral-loss cascades
(up to ``max_serial_losses`` losses taken from the charged side).

Fragment charge states: singly-charged precursors emit singly-charged
fragments carrying the precursor's adduct; multiply-protonated precursors
(e.g. ``[M+2H]2+``) emit both the singly-protonated series ``[frag+H]+``
and the charge-retained series ``[frag+2H]2+`` (see FRAGMENT_CARRIERS).

Chemistry-aware constraints:
  * functional-group gating: a neutral loss is only allowed if the charged
    side contains the matching substructure (SMARTS),
  * even-electron rule: in ESI-like ionization, only fragment ions with an
    even valence-electron count are kept (disabled for radical adducts).

Defaults are chosen for decoy-discrimination (see scripts/benchmark_decoys.py):
loss cascades beyond a single loss and the extended loss dictionary in
LOSS_DEFS raise coverage but lower discrimination under unweighted scoring;
enable them once likelihood-weighted scoring is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, combinations_with_replacement

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

MONOISOTOPIC_MASSES = {
    "H": 1.00782503223,
    "C": 12.0,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "F": 18.99840316273,
    "Si": 27.97692653465,
    "P": 30.97376199842,
    "S": 31.9720711744,
    "Cl": 34.968852682,
    "Br": 78.9183376,
    "I": 126.9044719,
}
H_MASS = MONOISOTOPIC_MASSES["H"]
PROTON_MASS = 1.00727647

ADDUCTS = {
    "[M+H]+": (PROTON_MASS, 1),
    "[M+Na]+": (22.989220, 1),
    "[M+K]+": (38.963158, 1),
    "[M+NH4]+": (18.033823, 1),
    "[M+Cl]-": (34.969402, 1),
    "[M+CH2O2-H]-": (44.998201, 1),
    "[M-H]-": (-PROTON_MASS, 1),
    "[M+H-H2O]+": (-17.003288, 1),
    "[M+H-2H2O]+": (-35.013853, 1),
    "[M-H-H2O]-": (-19.017841, 1),
    "[M]+": (0.0, 1),
    "[M]-": (0.0, 1),
    "[M+2H]2+": (2 * PROTON_MASS, 2),
}

ADDUCT_ALIASES = {
    "[2M+H]+": "[M+H]+",
    "[2M+Na]+": "[M+Na]+",
    "[2M-H]-": "[M-H]-",
    "[2M+NH4]+": "[M+NH4]+",
    "[2M+K]+": "[M+K]+",
    "[M-H2O+H]+": "[M+H-H2O]+",
    "[M-2H2O+H]+": "[M+H-2H2O]+",
    "[M-H2O-H]-": "[M-H-H2O]-",
    "[M-H2O]+": "[M+H-H2O]+",
}

RADICAL_ADDUCTS = {"[M]+", "[M]-"}

# Fragment ion charge states per precursor adduct. Singly-charged precursors keep
# their carrier at charge 1 (doubly-charged fragments would be charge-separated
# exotica in small-molecule MS2). Multiply-protonated precursors additionally
# emit the dominant singly-protonated series alongside the charge-retained one.
FRAGMENT_CARRIERS = {
    "[M+2H]2+": [(PROTON_MASS, 1), (2 * PROTON_MASS, 2)],
}

LOSS_DEFS = {
    "H2O": (18.0105646, "[OX2H]"),
    "NH3": (17.0265481, "[NX3;H2,H3]"),
    "CO": (27.9949146, "[CX3]=[OX1]"),
    "CO2": (43.989829, "[CX3](=[OX1])[OX2]"),
    "CH4": (16.0313001, "[#6X4H3]"),
    "CH3OH": (32.0262147, "[#6X4H3][OX2]"),
    "CH2O": (30.0105646, "[CX4H2][OX2]"),
    "CH2CO": (42.0105646, "[#6X4H3][CX3]=[OX1]"),
    "C2H4O2": (60.0211294, "[#6X4H3][CX3]=[OX1]"),
    "HCOOH": (46.0054793, "[CX3](=[OX1])[OX2H]"),
    "HCN": (27.010899, "[CX2]#[NX1]"),
    "C2H2": (26.0156501, "[a]"),
    "HCl": (35.976677, "[Cl]"),
    "HF": (20.006228, "[F]"),
    "H2S": (33.987721, "[SX2H]"),
    "C2H5OH": (46.0418648, "[#6X4H3][CX4H2][OX2]"),
}
DEFAULT_LOSSES = ("H2O", "NH3", "CO", "CO2", "CH3OH")

_SMARTS_CACHE: dict[str, Chem.Mol | None] = {}


def _smarts_query(pattern: str):
    if pattern not in _SMARTS_CACHE:
        _SMARTS_CACHE[pattern] = Chem.MolFromSmarts(pattern)
    return _SMARTS_CACHE[pattern]


@dataclass(frozen=True)
class TheoreticalFragment:
    mz: float
    side_atoms: tuple[int, ...]
    kind: str
    cut_bonds: tuple[tuple[int, int], ...]
    h_shift: int
    neutral_loss: str
    smiles: str


class Fragmentor:
    """Enumerate in-silico fragments of a molecule and match them to MS2 peaks."""

    def __init__(
        self,
        smiles: str,
        adduct: str = "[M+H]+",
        h_shifts: tuple[int, ...] = (-2, -1, 0, 1, 2),
        neutral_losses: tuple[str, ...] = DEFAULT_LOSSES,
        tolerance_ppm: float = 10.0,
        min_mz: float = 25.0,
        include_precursor_losses: bool = True,
        fg_gating: bool = True,
        max_serial_losses: int = 1,
        enforce_even_electron: bool = True,
        include_diagnostics: bool = True,
    ) -> None:
        adduct = ADDUCT_ALIASES.get(adduct, adduct)
        if adduct not in ADDUCTS:
            raise ValueError(
                f"Unsupported adduct {adduct!r}. Known: {sorted(set(ADDUCTS) | set(ADDUCT_ALIASES))}"
            )
        unknown = [l for l in neutral_losses if l not in LOSS_DEFS]
        if unknown:
            raise ValueError(f"Unknown neutral losses: {unknown}. Known: {sorted(LOSS_DEFS)}")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES: {smiles!r}")
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: m.GetNumAtoms())

        self.smiles = smiles
        self.adduct = adduct
        self.adduct_delta, self.charge = ADDUCTS[adduct]
        self.h_shifts = tuple(h_shifts)
        self.neutral_losses = tuple(neutral_losses)
        self.tolerance_ppm = tolerance_ppm
        self.min_mz = min_mz
        self.include_precursor_losses = include_precursor_losses
        self.fg_gating = fg_gating
        self.max_serial_losses = max_serial_losses
        self.enforce_even_electron = enforce_even_electron and adduct not in RADICAL_ADDUCTS
        self.include_diagnostics = include_diagnostics
        self.mol = mol
        self.neutral_mass = rdMolDescriptors.CalcExactMolWt(mol)
        self.precursor_mz = (self.neutral_mass + self.adduct_delta) / self.charge
        self.frag_carriers = FRAGMENT_CARRIERS.get(adduct, [(self.adduct_delta, self.charge)])
        self._fragments: list[TheoreticalFragment] | None = None
        self._composition_cache: dict[tuple[int, ...], tuple[float, int, int]] = {}
        self._mol_cache: dict[tuple[int, ...], Chem.Mol | None] = {}
        self._loss_options_cache: dict[tuple[int, ...], list[tuple[str, float]]] = {}

    def enumerate_fragments(self) -> list[TheoreticalFragment]:
        if self._fragments is not None:
            return self._fragments

        records: list[TheoreticalFragment] = []
        seen: set[tuple] = set()

        def add(side_atoms: tuple[int, ...], kind: str, cut_bonds: tuple[tuple[int, int], ...],
                h_shifts: tuple[int, ...] | None = None,
                losses: list[tuple[str, float]] | None = None,
                parity_check: bool = True) -> None:
            _, n_h, z_sum = self._side_composition(side_atoms)
            for h in (self.h_shifts if h_shifts is None else h_shifts):
                if parity_check and self.enforce_even_electron and (z_sum + n_h + h) % 2 != 0:
                    continue
                for loss_name, loss_mass in (self._loss_options(side_atoms) if losses is None else losses):
                    for carrier, c in self.frag_carriers:
                        mz = (self._side_mass(side_atoms) + h * H_MASS + carrier - loss_mass) / c
                        max_mz = (self.neutral_mass + carrier) / c + 2 * H_MASS + 0.1
                        if mz < self.min_mz or mz > max_mz:
                            continue
                        key = (round(mz, 4), side_atoms, h, loss_name)
                        if key in seen:
                            continue
                        seen.add(key)
                        records.append(
                            TheoreticalFragment(
                                mz=round(mz, 6),
                                side_atoms=side_atoms,
                                kind=kind,
                                cut_bonds=cut_bonds,
                                h_shift=h,
                                neutral_loss=loss_name,
                                smiles=self._side_smiles(side_atoms),
                            )
                        )

        n_atoms = self.mol.GetNumAtoms()
        if self.include_precursor_losses:
            add(tuple(range(n_atoms)), "precursor", ())

        for bond in self.mol.GetBonds():
            if bond.IsInRing():
                continue
            comps = self._components_excluding({bond.GetIdx()})
            if len(comps) == 2:
                cut = ((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),)
                for side in comps:
                    add(tuple(sorted(side)), "side_chain", cut)

        ring_bond_ids = [b.GetIdx() for b in self.mol.GetBonds() if b.IsInRing()]
        for b1, b2 in combinations(ring_bond_ids, 2):
            comps = self._components_excluding({b1, b2})
            if len(comps) == 2:
                cut = tuple(sorted((self._bond_key(b1), self._bond_key(b2))))
                for side in comps:
                    add(tuple(sorted(side)), "ring_fission", cut)

        if self.include_diagnostics:
            diag_h = (-2, -1, 0)
            no_loss = [("none", 0.0)]
            aromatic = {a.GetIdx() for a in self.mol.GetAtoms() if a.GetIsAromatic()}
            systems: list[set[int]] = []
            for atom_id in sorted(aromatic):
                if any(atom_id in s for s in systems):
                    continue
                comp = set()
                stack = [atom_id]
                while stack:
                    cur = stack.pop()
                    if cur in comp:
                        continue
                    comp.add(cur)
                    for nbr in self.mol.GetAtomWithIdx(cur).GetNeighbors():
                        if nbr.GetIdx() in aromatic:
                            stack.append(nbr.GetIdx())
                systems.append(comp)
            for sys_atoms in systems:
                if not (5 <= len(sys_atoms) <= 24):
                    continue
                exo = [b for b in self.mol.GetBonds()
                       if (b.GetBeginAtomIdx() in sys_atoms) != (b.GetEndAtomIdx() in sys_atoms)]

                def _diag_side(side: set[int]) -> None:
                    if not side or len(side) > 28:
                        return
                    ex = [b for b in self.mol.GetBonds()
                          if (b.GetBeginAtomIdx() in side) != (b.GetEndAtomIdx() in side)]
                    cut = tuple(sorted(self._bond_key(b.GetIdx()) for b in ex))
                    add(tuple(sorted(side)), "diagnostic", cut, diag_h, no_loss, False)

                _diag_side(set(sys_atoms))
                first_atoms = []
                for b in exo:
                    other = (b.GetEndAtomIdx() if b.GetBeginAtomIdx() in sys_atoms
                             else b.GetBeginAtomIdx())
                    if self.mol.GetAtomWithIdx(other).GetAtomicNum() in (6, 7, 8, 16):
                        first_atoms.append(other)
                        _diag_side(set(sys_atoms) | {other})
                if len(first_atoms) >= 2:
                    _diag_side(set(sys_atoms) | set(first_atoms))

            for atom in self.mol.GetAtoms():
                if atom.GetAtomicNum() != 7 or atom.GetIsAromatic():
                    continue
                skip = False
                for b in atom.GetBonds():
                    if b.GetBondTypeAsDouble() >= 2:
                        skip = True
                        break
                    other = b.GetOtherAtomIdx(atom.GetIdx())
                    other_atom = self.mol.GetAtomWithIdx(other)
                    if other_atom.GetAtomicNum() == 8:
                        skip = True
                        break
                    if other_atom.GetAtomicNum() == 6:
                        for b2 in other_atom.GetBonds():
                            if b2.GetBondTypeAsDouble() >= 2 and (
                                self.mol.GetAtomWithIdx(b2.GetBeginAtomIdx()).GetAtomicNum() == 8
                                or self.mol.GetAtomWithIdx(b2.GetEndAtomIdx()).GetAtomicNum() == 8
                            ):
                                skip = True
                                break
                        if skip:
                            break
                if skip:
                    continue
                carbons: list[int] = []
                stack = [(atom.GetIdx(), 0)]
                visited = {atom.GetIdx()}
                while stack and len(carbons) < 5:
                    cur, depth = stack.pop()
                    for nbr in self.mol.GetAtomWithIdx(cur).GetNeighbors():
                        nid = nbr.GetIdx()
                        if nid in visited:
                            continue
                        visited.add(nid)
                        if (nbr.GetAtomicNum() == 6 and not nbr.GetIsAromatic()
                                and not any(b.GetBondTypeAsDouble() >= 2
                                            and (b.GetBeginAtomIdx() == nid or b.GetEndAtomIdx() == nid)
                                            for b in nbr.GetBonds())):
                            carbons.append(nid)
                            if len(carbons) >= 5:
                                break
                            stack.append((nid, depth + 1))
                if len(carbons) >= 3:
                    side = tuple(sorted([atom.GetIdx()] + carbons[:5]))
                    exo = [b for b in self.mol.GetBonds()
                           if (b.GetBeginAtomIdx() in side) != (b.GetEndAtomIdx() in side)]
                    cut = tuple(sorted(self._bond_key(b.GetIdx()) for b in exo))
                    add(side, "diagnostic", cut, diag_h, no_loss)

        self._fragments = sorted(records, key=lambda f: f.mz)
        return self._fragments

    def theoretical_fragments_df(self) -> pd.DataFrame:
        frags = self.enumerate_fragments()
        return pd.DataFrame([f.__dict__ for f in frags])

    def match_peaks(
        self,
        mzs,
        intensities=None,
        tolerance_ppm: float | None = None,
        best_only: bool = True,
    ) -> pd.DataFrame:
        """Match observed peak m/z values against theoretical fragments."""
        tol = tolerance_ppm if tolerance_ppm is not None else self.tolerance_ppm
        frags = self.enumerate_fragments()
        theo = np.array([f.mz for f in frags])
        obs = np.asarray(mzs, dtype=float)

        rows = []
        for i, m in enumerate(obs):
            j = int(np.searchsorted(theo, m))
            for k in (j - 1, j):
                if 0 <= k < len(theo):
                    ppm = (theo[k] - m) / m * 1e6
                    if abs(ppm) <= tol:
                        f = frags[k]
                        orders = [self.mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() for a, b in f.cut_bonds]
                        rows.append(
                            {
                                "peak_index": i,
                                "peak_mz": m,
                                "theo_mz": f.mz,
                                "error_ppm": round(ppm, 2),
                                "kind": f.kind,
                                "h_shift": f.h_shift,
                                "neutral_loss": f.neutral_loss,
                                "cut_bonds": str(f.cut_bonds),
                                "side_atoms": f.side_atoms,
                                "fragment_smiles": f.smiles,
                                "frag_heavy_atoms": len(f.side_atoms),
                                "frag_heteroatoms": sum(
                                    1 for a in f.side_atoms if self.mol.GetAtomWithIdx(a).GetAtomicNum() != 6
                                ),
                                "cut_bond_max_order": max(orders) if orders else 0.0,
                                "cut_bond_aromatic": int(any(
                                    self.mol.GetBondBetweenAtoms(a, b).GetIsAromatic() for a, b in f.cut_bonds
                                )),
                            }
                        )
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["n_candidates_at_peak"] = df.groupby("peak_index")["peak_index"].transform("size")
        if intensities is not None:
            df["intensity"] = np.asarray(intensities, dtype=float)[df["peak_index"].to_numpy()]
        if best_only:
            df = df.loc[df["error_ppm"].abs().groupby(df["peak_index"]).idxmin()]
        return df.sort_values("peak_mz").reset_index(drop=True)

    def explained_intensity_fraction(self, mzs, intensities, tolerance_ppm: float | None = None) -> float:
        df = self.match_peaks(mzs, intensities, tolerance_ppm=tolerance_ppm)
        if df.empty:
            return 0.0
        return float(df["intensity"].sum() / np.asarray(intensities, dtype=float).sum())

    def _loss_options(self, side_atoms: tuple[int, ...]) -> list[tuple[str, float]]:
        if side_atoms in self._loss_options_cache:
            return self._loss_options_cache[side_atoms]

        if self.fg_gating:
            side_mol = self._side_mol(side_atoms)
            if side_mol is None:
                allowed = list(self.neutral_losses)
            else:
                allowed = [
                    l for l in self.neutral_losses
                    if (q := _smarts_query(LOSS_DEFS[l][1])) is not None and side_mol.HasSubstructMatch(q)
                ]
        else:
            allowed = list(self.neutral_losses)

        options: list[tuple[str, float]] = [("none", 0.0)]
        options += [(l, LOSS_DEFS[l][0]) for l in allowed]
        for depth in range(2, self.max_serial_losses + 1):
            for combo in combinations_with_replacement(allowed, depth):
                options.append(("+".join(combo), sum(LOSS_DEFS[l][0] for l in combo)))
        self._loss_options_cache[side_atoms] = options
        return options

    def _components_excluding(self, removed_bond_ids: set[int]) -> list[list[int]]:
        n = self.mol.GetNumAtoms()
        adjacency: list[list[int]] = [[] for _ in range(n)]
        for bond in self.mol.GetBonds():
            if bond.GetIdx() in removed_bond_ids:
                continue
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            adjacency[a].append(b)
            adjacency[b].append(a)

        seen = [False] * n
        components: list[list[int]] = []
        for start in range(n):
            if seen[start]:
                continue
            stack, comp = [start], []
            seen[start] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nb in adjacency[node]:
                    if not seen[nb]:
                        seen[nb] = True
                        stack.append(nb)
            components.append(comp)
        return components

    def _bond_key(self, bond_id: int) -> tuple[int, int]:
        bond = self.mol.GetBondWithIdx(bond_id)
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        return (a, b) if a < b else (b, a)

    def _side_composition(self, side_atoms: tuple[int, ...]) -> tuple[float, int, int]:
        """Return (monoisotopic mass, total H count, sum of atomic numbers) of a side."""
        if side_atoms not in self._composition_cache:
            mass, n_h, z_sum = 0.0, 0, 0
            for idx in side_atoms:
                atom = self.mol.GetAtomWithIdx(idx)
                m = MONOISOTOPIC_MASSES.get(atom.GetSymbol())
                if m is None:
                    m = atom.GetMass()
                h = atom.GetTotalNumHs(includeNeighbors=True)
                mass += m + h * H_MASS
                n_h += h
                z_sum += atom.GetAtomicNum()
            self._composition_cache[side_atoms] = (mass, n_h, z_sum)
        return self._composition_cache[side_atoms]

    def _side_mass(self, side_atoms: tuple[int, ...]) -> float:
        return self._side_composition(side_atoms)[0]

    def _side_mol(self, side_atoms: tuple[int, ...]) -> Chem.Mol | None:
        if side_atoms not in self._mol_cache:
            keep = set(side_atoms)
            rw = Chem.RWMol(self.mol)
            for idx in sorted(set(range(rw.GetNumAtoms())) - keep, reverse=True):
                rw.RemoveAtom(idx)
            mol = rw.GetMol()
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                try:
                    mol.UpdatePropertyCache(strict=False)
                    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                except Exception:
                    mol = None
            self._mol_cache[side_atoms] = mol
        return self._mol_cache[side_atoms]

    def _side_smiles(self, side_atoms: tuple[int, ...]) -> str:
        if side_atoms not in self._mol_cache:
            self._side_mol(side_atoms)
        mol = self._mol_cache[side_atoms]
        if mol is None:
            return ""
        try:
            return Chem.MolToSmiles(mol)
        except Exception:
            return ""
