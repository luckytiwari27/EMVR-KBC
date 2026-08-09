"""
macro_average_check.py -- diagnose whether the gap between your reproduction's
global metric and the LeSR paper's reported number is a macro/micro-average
mismatch, rather than a real reproduction problem.

Your saved run already has everything needed for this -- no extra API calls,
no rerun. It just recombines the per-relation *_test_ranks.pt files two
different ways:

  MICRO average (what lesr.py's own final printout gives you): pool every
    relation's ranks into one big list, then compute MR/MRR/Hit@k once.
    Relations with more test queries dominate this number.

  MACRO average (what a lot of papers, possibly including LeSR's Table 4,
    report instead): compute MR/MRR/Hit@k separately PER relation, then take
    the plain unweighted mean across relations. Every relation counts
    equally regardless of how many queries it has.

If your macro-average lands close to the paper's reported number, that
mismatch is most of your "gap" -- not a bug in your reproduction. If it's
still far off even after this check, the gap is likely real (LLM snapshot
drift, hyperparameters, or an actual bug) and is worth investigating further
before claiming a beat-LeSR result.

USAGE:
    python3 macro_average_check.py --run_dir runs/umls_vanilla --paper_mrr 0.722
"""
import argparse
import glob
import os

import numpy as np
import torch

from reasoner import compute_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="A completed --run_reasoner run directory.")
    p.add_argument("--paper_mrr", type=float, default=None,
                    help="The published paper's reported MRR for this dataset/config, "
                         "for a direct side-by-side comparison (e.g. 0.722 for LeSR "
                         "GPT-3.5 w/ WL on UMLS).")
    p.add_argument("--relation_names", default=None,
                    help="Optional path to a relations.dict-style file (id<TAB>name) "
                         "to print human-readable relation names instead of IDs.")
    return p.parse_args()


def relation_ids_in_run(run_dir):
    reasoner_dir = os.path.join(run_dir, "reasoner")
    ids = []
    for f in glob.glob(os.path.join(reasoner_dir, "*_test_ranks.pt")):
        rid = os.path.basename(f).replace("_test_ranks.pt", "")
        ids.append(rid)
    return sorted(ids, key=lambda x: int(x))


def load_relation_names(path):
    names = {}
    if path is None:
        return names
    with open(path) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                names[parts[0]] = parts[1]
    return names


if __name__ == "__main__":
    args = parse_args()
    reasoner_dir = os.path.join(args.run_dir, "reasoner")
    if not os.path.isdir(reasoner_dir):
        raise FileNotFoundError("{} has no reasoner/ subfolder -- is this a completed run?".format(args.run_dir))

    relation_names = load_relation_names(args.relation_names)
    relation_ids = relation_ids_in_run(args.run_dir)
    if not relation_ids:
        raise FileNotFoundError("No *_test_ranks.pt files found under {}".format(reasoner_dir))

    per_relation = []
    all_ranks = []
    for rid in relation_ids:
        ranks = torch.load(os.path.join(reasoner_dir, "{}_test_ranks.pt".format(rid)))
        if ranks.numel() == 0:
            continue
        mr, mrr, hit_ks = compute_metrics(ranks, k_vals=[1, 3, 10])
        per_relation.append({
            "rid": rid, "name": relation_names.get(rid, ""), "n": ranks.shape[0],
            "mr": mr, "mrr": mrr, "hit1": hit_ks[0], "hit3": hit_ks[1], "hit10": hit_ks[2],
        })
        all_ranks.append(ranks)

    # ---- MICRO average: pool everything, compute once (matches lesr.py's own printout) ----
    pooled = torch.cat(all_ranks)
    micro_mr, micro_mrr, micro_hits = compute_metrics(pooled, k_vals=[1, 3, 10])

    # ---- MACRO average: mean of the per-relation numbers, unweighted ----
    macro_mr = float(np.mean([r["mr"] for r in per_relation]))
    macro_mrr = float(np.mean([r["mrr"] for r in per_relation]))
    macro_hit1 = float(np.mean([r["hit1"] for r in per_relation]))
    macro_hit3 = float(np.mean([r["hit3"] for r in per_relation]))
    macro_hit10 = float(np.mean([r["hit10"] for r in per_relation]))

    W = 78
    print("=" * W)
    print("MACRO vs MICRO average check -- {}".format(args.run_dir))
    print("{} relations, {} total test queries".format(len(per_relation), pooled.shape[0]))
    print("=" * W)

    print("\n{:<12}{:>18}{:>22}".format("Metric", "MICRO (pooled)", "MACRO (per-rel mean)"))
    print("-" * W)
    print("{:<12}{:>18.4f}{:>22.4f}".format("MR", micro_mr, macro_mr))
    print("{:<12}{:>18.4f}{:>22.4f}".format("MRR", micro_mrr, macro_mrr))
    print("{:<12}{:>17.2f}%{:>21.2f}%".format("Hit@1", micro_hits[0] * 100, macro_hit1 * 100))
    print("{:<12}{:>17.2f}%{:>21.2f}%".format("Hit@3", micro_hits[1] * 100, macro_hit3 * 100))
    print("{:<12}{:>17.2f}%{:>21.2f}%".format("Hit@10", micro_hits[2] * 100, macro_hit10 * 100))

    if args.paper_mrr is not None:
        print("\n{:<28}{:.4f}".format("Paper's reported MRR:", args.paper_mrr))
        micro_gap = args.paper_mrr - micro_mrr
        macro_gap = args.paper_mrr - macro_mrr
        print("{:<28}{:+.4f}  ({:+.1f}%)".format("Gap vs MICRO:", micro_gap, micro_gap / args.paper_mrr * 100))
        print("{:<28}{:+.4f}  ({:+.1f}%)".format("Gap vs MACRO:", macro_gap, macro_gap / args.paper_mrr * 100))
        print()
        if abs(macro_gap) < abs(micro_gap) * 0.5:
            print("MACRO average is much closer to the paper's number than MICRO is.")
            print("This strongly suggests the paper reports a macro-average (or something")
            print("close to it), and your original 'gap' vs LeSR was mostly a metric")
            print("definition mismatch, not a reproduction problem.")
        elif abs(macro_gap) < 0.05:
            print("MACRO average lands close to the paper's number (within ~0.05 MRR).")
            print("Your reproduction is likely sound -- the averaging convention was the")
            print("main source of the discrepancy you saw before.")
        else:
            print("Neither MICRO nor MACRO average closes the gap to the paper's number.")
            print("This looks like a real reproduction discrepancy, not just an averaging")
            print("convention difference -- worth investigating (LLM snapshot drift,")
            print("hyperparameters, or a possible bug) before claiming a beat-LeSR result.")

    print("\n{:<8}{:<24}{:>8}{:>10}{:>10}".format("ID", "Relation", "N", "MRR", "Hit@10"))
    print("-" * W)
    for r in sorted(per_relation, key=lambda x: -x["n"]):
        print("{:<8}{:<24}{:>8}{:>10.4f}{:>9.1f}%".format(
            r["rid"], r["name"][:23], r["n"], r["mrr"], r["hit10"] * 100))
    print("\n(sorted by query count N, descending -- the relations at the top have the")
    print("most influence on the MICRO average; every relation counts equally in MACRO)")
    print("=" * W)