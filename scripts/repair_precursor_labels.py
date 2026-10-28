"""Repair precursor/adduct labels in the training data (supervised, truth-aware).

Uses the truth SMILES to compute each structure's exact monoisotopic mass,
then fixes the label bugs quantified by scripts/diagnose_precursor_errors.py:

  * isotope misassignment: precursor is the M+1/M+2 isotope -> correct m/z
    by -1.00335/z / -2.00670/z (the "weak monoisotopic peak" scenario),
  * adduct swaps: true [M+Na]+/[M+K]+/[M+NH4]+/[M+Cl]- labeled [M+H]+/[M-H]-
    -> relabel the adduct,
  * in-source neutral losses (H2O, NH3): relabel to the loss-inclusive adduct,
  * labeled monomer that is really a dimer ion -> relabel to the [2M+...] form,
  * unsupervised garbage (non-finite/absurd precursor m/z) and unexplained
    mass mismatches >= --drop-da -> drop the row.

Small unexplained mismatches (0.02 Da .. drop threshold) are kept and flagged
`mass_mismatch` in `precursor_flag` (mostly genuine low-resolution instruments
such as LC-ESI-QQ, where the spectrum-truth association is still correct).

Writes a new parquet; the source file is never modified. Added columns:
  precursor_mz_raw   original value (set only for changed rows)
  adduct_raw         original label (set only for changed rows)
  repair_action      none | isotope_m1 | isotope_m2 | adduct_swap |
                     in_source_loss | dimer | drop:garbage
  precursor_err_da   post-repair error vs the truth mass (Da)
  precursor_flag     ok | mass_mismatch

Usage:
    uv run python scripts/repair_precursor_labels.py \
        [--data data/train_cut.parquet] [--out data/train_cut_repaired.parquet]
"""
import argparse
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from diagnose_precursor_errors import exact_masses, parse_adduct  # noqa: E402

ISOTOPE_OFFSETS = [("isotope_m1", 1.0033548), ("isotope_m2", 2.0067096)]
SWAPS = {  # current label -> {new label: err_da (z=1) if the true ion is `new`
    "[M+H]+": {"[M+Na]+": 22.989220 - 1.007276,
               "[M+K]+": 38.963158 - 1.007276,
               "[M+NH4]+": 18.033823 - 1.007276},
    "[M+Na]+": {"[M+H]+": -(22.989220 - 1.007276)},
    "[M+K]+": {"[M+H]+": -(38.963158 - 1.007276)},
    "[M-H]-": {"[M+Cl]-": 34.969402 + 1.007276},
    "[M+Cl]-": {"[M-H]-": -(34.969402 + 1.007276)},
}
LOSS_OFFSETS = {  # err_da (z=1) when the intact ion lost this before acquisition
    "H2O": 18.0105646,
    "NH3": 17.0265481,
}


def classify(err_da: float, z: int, adduct: str):
    """Best hypothesis for a single row. Returns (action, new_adduct, mz_corr)."""
    tol = 0.02 / z
    if abs(err_da) <= tol:
        return "none", adduct, 0.0
    for action, off in ISOTOPE_OFFSETS:
        if abs(err_da - off / z) <= tol:
            return action, adduct, -off / z
    for new_adduct, off in SWAPS.get(adduct, {}).items():
        if abs(err_da - off / z) <= tol:
            return "adduct_swap", new_adduct, 0.0
    for loss, off in LOSS_OFFSETS.items():
        if abs(err_da + off / z) <= tol:
            m = re.fullmatch(r"\[([^\]]*)\](\d*[+-])", adduct)
            if m is None:
                continue
            return "in_source_loss", f"[{m.group(1)}-{loss}]{m.group(2)}", 0.0
    return "mass_mismatch", adduct, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "train_cut.parquet"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "train_cut_repaired.parquet"))
    ap.add_argument("--drop-da", type=float, default=1.0,
                    help="drop rows whose unexplained |err| exceeds this (Da)")
    args = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(args.data)
    print(f"rows in : {pf.metadata.num_rows}  ({args.data})", flush=True)

    smiles_set = set()
    for batch in pf.iter_batches(batch_size=200_000, columns=["normalized_smiles"]):
        smiles_set.update(s for s in batch.column(0).to_pylist() if s is not None)
    masses, bad_smiles = exact_masses(sorted(smiles_set))
    print(f"structures: {len(masses)} ({bad_smiles} unparseable SMILES)", flush=True)

    schema = pf.schema_arrow
    names = schema.names
    i_adduct, i_prec, i_smiles = (names.index("adduct"), names.index("precursor_mz"),
                                  names.index("normalized_smiles"))
    f_prec, f_adduct = schema.field(i_prec), schema.field(i_adduct)
    new_fields = [
        pa.field("precursor_mz_raw", pa.float64()),
        pa.field("adduct_raw", schema.field(i_adduct).type),
        pa.field("repair_action", pa.string()),
        pa.field("precursor_err_da", pa.float64()),
        pa.field("precursor_flag", pa.string()),
    ]
    out_schema = pa.schema(list(schema) + new_fields)
    writer = pq.ParquetWriter(args.out, out_schema)

    cache: dict[str, object] = {}

    def parsed(label):
        if label not in cache:
            cache[label] = parse_adduct(label)
        return cache[label]

    n_in = n_out = 0
    action_counts: dict[str, int] = {}
    n_flag = 0
    for batch in pf.iter_batches(batch_size=100_000):
        b = len(batch)
        n_in += b
        adducts = batch.column(i_adduct).to_pylist()
        precs = batch.column(i_prec).to_pylist()
        smiles = batch.column(i_smiles).to_pylist()
        new_adduct = list(adducts)
        new_prec = list(precs)
        actions = [None] * b
        errs = [None] * b
        flags = [None] * b
        keep = [True] * b
        raw_prec = [None] * b
        raw_adduct = [None] * b
        for i in range(b):
            adduct, prec, smi = adducts[i], precs[i], smiles[i]
            M = masses.get(smi)
            p = parsed(adduct)
            if p is None or M is None or prec is None \
                    or not (10.0 < prec < 3000.0):
                actions[i] = "drop:unusable"
                keep[i] = False
                continue
            n, delta, z = p
            theo = (n * M + delta) / z
            err = prec - theo
            action, na, corr = classify(err, z, adduct)
            if action == "mass_mismatch":
                if adduct.startswith("[M"):
                    p2 = parsed("[2" + adduct[1:])
                    if p2 is not None:
                        n2, d2, z2 = p2
                        if abs(prec - (n2 * M + d2) / z2) <= 0.02:
                            action, na, corr = "dimer", "[2" + adduct[1:], 0.0
                if action == "mass_mismatch" and abs(err) >= args.drop_da:
                    actions[i] = "drop:garbage"
                    keep[i] = False
                    continue
            # error bookkeeping always against the post-repair label
            pn = parsed(na)
            nn, dn, zn = pn
            prec_fixed = prec + corr
            err_new = prec_fixed - (nn * M + dn) / zn
            new_adduct[i] = na
            new_prec[i] = prec_fixed
            errs[i] = err_new
            actions[i] = action
            flags[i] = "ok" if abs(err_new) <= 0.02 else "mass_mismatch"
            if action != "none":
                raw_prec[i] = prec
                raw_adduct[i] = adduct
        tbl = pa.Table.from_batches([batch], schema=schema)
        tbl = tbl.set_column(i_adduct, f_adduct, pa.array(new_adduct, f_adduct.type))
        tbl = tbl.set_column(i_prec, f_prec, pa.array(new_prec, f_prec.type))
        tbl = tbl.append_column(new_fields[0], pa.array(raw_prec, pa.float64()))
        tbl = tbl.append_column(new_fields[1], pa.array(raw_adduct, f_adduct.type))
        tbl = tbl.append_column(new_fields[2], pa.array(actions, pa.string()))
        tbl = tbl.append_column(new_fields[3], pa.array(errs, pa.float64()))
        tbl = tbl.append_column(new_fields[4], pa.array(flags, pa.string()))
        tbl = tbl.filter(pa.array(keep))
        writer.write_table(tbl)
        n_out += len(tbl)
        for i in range(b):
            if keep[i]:
                action_counts[actions[i]] = action_counts.get(actions[i], 0) + 1
                if flags[i] == "mass_mismatch":
                    n_flag += 1
        done = n_in / pf.metadata.num_rows * 100
        print(f"  {n_in} rows read ({done:.0f}%), {n_out} kept", flush=True)

    writer.close()
    print(f"\nrepair actions (kept rows):")
    for k, v in sorted(action_counts.items()):
        print(f"  {k:<20} {v}")
    print(f"\ndropped rows : {n_in - n_out}")
    print(f"kept rows    : {n_out}")
    print(f"flagged mass_mismatch: {n_flag}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
