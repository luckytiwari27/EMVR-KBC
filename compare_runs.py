"""
compare_runs.py -- consolidated EMVR-KBC vs vanilla LeSR comparison report.

Loads two COMPLETED --run_reasoner runs (a vanilla run and an EMVR-KBC run
that shared the same extractor/proposer output) and prints two tables:

  1) the core accuracy + filtering table (LeSR vs EMVR-KBC side by side)
  2) the EMVR-only rule-analysis table (Average EV, Filtering Rate/Recall,
     EV-weight correlation, HCR, RCS/RQI where annotations exist)

Every number is read from artifacts lesr.py already writes to each run's
reasoner/ directory (*_ranks.pt, *_learned_weights.pt, *_rules.json,
*_ev_scores.json, *_emvr_stats.json) and recombined using the SAME
compute_metrics() function the live pipeline uses.

Multi-seed usage: pass a comma-separated list of run dirs to each of
--vanilla_run_dir / --emvr_run_dir (one dir per seed, same order). All
numbers are then reported as mean +/- std across seeds.

RCS/RQI: pass --wd15k_annotations <path to JSON {rule_text: [0/0.5/1,...]}>
if you have the Lv et al. 2021 WD15K human interpretability annotations.
Without it, RCS/RQI print as N/A -- never fabricated.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from reasoner import compute_metrics
from evidence import (
    filtering_recall, ev_weight_correlation, high_confidence_rule_ratio,
    rule_clarity_score, rule_quality_index,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vanilla_run_dir", required=True,
                    help="Comma-separated list of vanilla run dirs (one per seed).")
    p.add_argument("--emvr_run_dir", required=True,
                    help="Comma-separated list of EMVR-KBC run dirs (one per seed, "
                         "same order/seeds as --vanilla_run_dir).")
    p.add_argument("--dataset", required=True)
    p.add_argument("--hcr_threshold", type=float, default=0.5,
                    help="theta_hi: significance-weight threshold for 'high-confidence' rule.")
    p.add_argument("--wd15k_annotations", default=None,
                    help="Path to a JSON file of {rule_text: [0/0.5/1 scores]} from the "
                         "Lv et al. 2021 WD15K human interpretability annotations.")
    return p.parse_args()


def load_run_dirs(arg):
    dirs = [d.strip() for d in arg.split(",") if d.strip()]
    for d in dirs:
        if not os.path.isdir(os.path.join(d, "reasoner")):
            raise FileNotFoundError("{} has no reasoner/ subfolder -- is this a completed run?".format(d))
    return dirs


def relation_ids_in_run(run_dir):
    reasoner_dir = os.path.join(run_dir, "reasoner")
    ids = []
    for f in glob.glob(os.path.join(reasoner_dir, "*_test_ranks.pt")):
        rid = os.path.basename(f).replace("_test_ranks.pt", "")
        ids.append(rid)
    return sorted(ids, key=lambda x: int(x))


def load_ranks(run_dir, relation_ids):
    all_ranks = []
    for rid in relation_ids:
        fname = os.path.join(run_dir, "reasoner", "{}_test_ranks.pt".format(rid))
        if os.path.exists(fname):
            all_ranks.append(torch.load(fname))
    if not all_ranks:
        return None
    return torch.cat(all_ranks)


def accuracy_for_run(run_dir):
    relation_ids = relation_ids_in_run(run_dir)
    ranks = load_ranks(run_dir, relation_ids)
    if ranks is None:
        return None
    mr, mrr, hit_ks = compute_metrics(ranks, k_vals=[1, 3, 10])
    return {"n_query": ranks.shape[0], "mr": mr, "mrr": mrr,
            "hit1": hit_ks[0], "hit3": hit_ks[1], "hit10": hit_ks[2]}


def aggregate_seeds(per_seed_dicts, keys):
    out = {}
    for k in keys:
        vals = [d[k] for d in per_seed_dicts if d is not None and d.get(k) is not None]
        if not vals:
            out[k] = (None, None)
        else:
            out[k] = (float(np.mean(vals)), float(np.std(vals)) if len(vals) > 1 else 0.0)
    return out


def load_rule_texts(run_dir, rid):
    fname = os.path.join(run_dir, "reasoner", "{}_rules.json".format(rid))
    if not os.path.exists(fname):
        return []
    with open(fname) as f:
        rules = json.load(f)
    return [r[0] for r in rules]


def load_weights(run_dir, rid):
    fname = os.path.join(run_dir, "reasoner", "{}_learned_weights.pt".format(rid))
    if not os.path.exists(fname):
        return None
    w = torch.load(fname)
    if isinstance(w, dict):
        w = w["logical_weights"]
    return w.detach().cpu().numpy() if hasattr(w, "detach") else np.asarray(w)


def load_emvr_stats(run_dir, rid):
    fname = os.path.join(run_dir, "reasoner", "{}_emvr_stats.json".format(rid))
    if not os.path.exists(fname):
        return None
    with open(fname) as f:
        return json.load(f)


def load_ev_scores(run_dir, rid):
    fname = os.path.join(run_dir, "reasoner", "{}_ev_scores.json".format(rid))
    if not os.path.exists(fname):
        return None
    with open(fname) as f:
        return json.load(f)


def per_seed_emvr_analysis(vanilla_dir, emvr_dir, hcr_threshold):
    emvr_relations = relation_ids_in_run(emvr_dir)
    n_candidates_total, n_verified_total = 0, 0
    recall_hits, recall_total = 0, 0
    pooled_ev, pooled_w = [], []
    all_candidate_ev = []
    hcr_vanilla_vals, hcr_emvr_vals = [], []

    for rid in emvr_relations:
        emvr_stats = load_emvr_stats(emvr_dir, rid)
        if emvr_stats is not None:
            n_candidates_total += emvr_stats["n_candidates"]
            n_verified_total += emvr_stats["n_verified"]
            for row in emvr_stats.get("rows", []):
                if "ev_i" in row and row["ev_i"] is not None:
                    all_candidate_ev.append(row["ev_i"])

        emvr_rule_texts = load_rule_texts(emvr_dir, rid)
        vanilla_rule_texts = load_rule_texts(vanilla_dir, rid)
        vanilla_weights = load_weights(vanilla_dir, rid)
        emvr_weights = load_weights(emvr_dir, rid)

        if vanilla_weights is not None and len(vanilla_rule_texts) == len(vanilla_weights) - 1:
            vanilla_logical_w = vanilla_weights[:-1]
            vanilla_high_texts = [t for t, w in zip(vanilla_rule_texts, vanilla_logical_w) if w > hcr_threshold]
            if vanilla_high_texts:
                r = filtering_recall(emvr_rule_texts, vanilla_high_texts)
                if r is not None:
                    recall_hits += len(set(vanilla_high_texts) & set(emvr_rule_texts))
                    recall_total += len(vanilla_high_texts)

        if vanilla_weights is not None and len(vanilla_weights) > 1:
            hcr_vanilla_vals.append(high_confidence_rule_ratio(vanilla_weights[:-1], hcr_threshold))
        if emvr_weights is not None and len(emvr_weights) > 1:
            hcr_emvr_vals.append(high_confidence_rule_ratio(emvr_weights[:-1], hcr_threshold))

        ev_scores = load_ev_scores(emvr_dir, rid)
        if ev_scores is not None and emvr_weights is not None:
            n_logical = len(emvr_weights) - 1
            evs = ev_scores[:n_logical]
            ws = emvr_weights[:n_logical]
            if len(evs) == len(ws) and len(evs) > 0:
                pooled_ev.extend(evs)
                pooled_w.extend(ws.tolist())

    filt_rate = 1.0 - (n_verified_total / n_candidates_total) if n_candidates_total > 0 else None
    filt_recall = (recall_hits / recall_total) if recall_total > 0 else None
    corr = ev_weight_correlation(pooled_ev, pooled_w) if pooled_ev else {"pearson": None, "spearman": None}
    avg_ev = float(np.mean(all_candidate_ev)) if all_candidate_ev else None

    return {
        "n_candidates": n_candidates_total,
        "n_verified": n_verified_total,
        "filtering_rate": filt_rate,
        "filtering_recall": filt_recall,
        "filtering_recall_n": recall_total,
        "average_ev": avg_ev,
        "ev_weight_pearson": corr["pearson"],
        "ev_weight_spearman": corr["spearman"],
        "hcr_vanilla": float(np.mean(hcr_vanilla_vals)) if hcr_vanilla_vals else None,
        "hcr_emvr": float(np.mean(hcr_emvr_vals)) if hcr_emvr_vals else None,
    }


def fmt(mean_std, pct=False, decimals=4, already_pct=False):
    mean, std = mean_std
    if mean is None:
        return "N/A"
    if pct and not already_pct:
        mean = mean * 100
        std = (std * 100) if std is not None else None
    suffix = "%" if pct else ""
    if std is None or std == 0.0:
        return "{:.{d}f}{}".format(mean, suffix, d=decimals)
    return "{:.{d}f}+/-{:.{d}f}{}".format(mean, std, suffix, d=decimals)


def fmt_single(v, pct=False, decimals=4, already_pct=False):
    if v is None:
        return "N/A"
    if pct and not already_pct:
        v = v * 100
    return "{:.{d}f}{}".format(v, "%" if pct else "", d=decimals)


def fmt_int(mean_std):
    mean, std = mean_std
    if mean is None:
        return "N/A"
    if std is None or std == 0.0:
        return "{:.0f}".format(mean)
    return "{:.0f}+/-{:.0f}".format(mean, std)


if __name__ == "__main__":
    args = parse_args()
    vanilla_dirs = load_run_dirs(args.vanilla_run_dir)
    emvr_dirs = load_run_dirs(args.emvr_run_dir)
    if len(vanilla_dirs) != len(emvr_dirs):
        raise ValueError("Got {} vanilla run dirs but {} EMVR run dirs -- these must be "
                          "paired one-to-one, one pair per seed.".format(len(vanilla_dirs), len(emvr_dirs)))
    n_seeds = len(vanilla_dirs)

    wd15k_annotations = None
    if args.wd15k_annotations:
        with open(args.wd15k_annotations) as f:
            wd15k_annotations = json.load(f)

    vanilla_acc_per_seed = [accuracy_for_run(d) for d in vanilla_dirs]
    emvr_acc_per_seed = [accuracy_for_run(d) for d in emvr_dirs]
    vanilla_acc = aggregate_seeds(vanilla_acc_per_seed, ["mr", "mrr", "hit1", "hit3", "hit10"])
    emvr_acc = aggregate_seeds(emvr_acc_per_seed, ["mr", "mrr", "hit1", "hit3", "hit10"])

    per_seed = [per_seed_emvr_analysis(v, e, args.hcr_threshold) for v, e in zip(vanilla_dirs, emvr_dirs)]
    agg = aggregate_seeds(per_seed, [
        "n_candidates", "n_verified", "filtering_rate", "filtering_recall",
        "average_ev", "ev_weight_pearson", "ev_weight_spearman", "hcr_vanilla", "hcr_emvr",
    ])

    rcs_emvr, rqi_emvr = None, None
    if wd15k_annotations is not None:
        all_rcs = []
        for e_dir in emvr_dirs:
            for rid in relation_ids_in_run(e_dir):
                rule_texts = load_rule_texts(e_dir, rid)
                rcs = rule_clarity_score(rule_texts, wd15k_annotations)
                if rcs is not None:
                    all_rcs.append(rcs)
        if all_rcs:
            rcs_emvr = float(np.mean(all_rcs))
            hcr_mean, _ = agg["hcr_emvr"]
            rqi_emvr = rule_quality_index(hcr_mean, rcs_emvr)

    W = 70
    def line(): print("-" * W)
    seed_note = "{} seed{}".format(n_seeds, "s" if n_seeds > 1 else "")

    print("=" * W)
    print("{}  --  {}".format(args.dataset, seed_note))
    print("=" * W)

    print("\nCORE TABLE")
    line()
    print("{:<22}{:>22}{:>22}".format("Metric", "LeSR", "EMVR-KBC"))
    print("{:<22}{:>22}{:>22}".format("MR (down)", fmt(vanilla_acc["mr"], decimals=4), fmt(emvr_acc["mr"], decimals=4)))
    print("{:<22}{:>22}{:>22}".format("MRR (up)", fmt(vanilla_acc["mrr"], decimals=4), fmt(emvr_acc["mrr"], decimals=4)))
    print("{:<22}{:>22}{:>22}".format("Hits@1 (up)", fmt(vanilla_acc["hit1"], pct=True, decimals=2), fmt(emvr_acc["hit1"], pct=True, decimals=2)))
    print("{:<22}{:>22}{:>22}".format("Hits@3 (up)", fmt(vanilla_acc["hit3"], pct=True, decimals=2), fmt(emvr_acc["hit3"], pct=True, decimals=2)))
    print("{:<22}{:>22}{:>22}".format("Hits@10 (up)", fmt(vanilla_acc["hit10"], pct=True, decimals=2), fmt(emvr_acc["hit10"], pct=True, decimals=2)))
    print("{:<22}{:>22}{:>22}".format("# candidate rules", fmt_int(agg["n_candidates"]), fmt_int(agg["n_candidates"])))
    print("{:<22}{:>22}{:>22}".format("# verified rules", "--", fmt_int(agg["n_verified"])))
    print("{:<22}{:>22}{:>22}".format("Filtering rate", "--", fmt(agg["filtering_rate"], pct=True, decimals=1)))
    print("{:<22}{:>22}{:>22}".format("Filtering recall", "--", fmt(agg["filtering_recall"], pct=True, decimals=1)))

    if n_seeds == 1:
        print("\n[WARNING] Single-seed comparison -- no std available. Do not treat any")
        print("delta above as a confirmed result until run across multiple seeds.")

    print("\nEMVR ANALYSIS TABLE")
    line()
    print("{:<28}{}".format("Average EV", fmt(agg["average_ev"], decimals=3)))
    print("{:<28}{}".format("Filtering rate", fmt(agg["filtering_rate"], pct=True, decimals=1)))
    fr_n = sum(s["filtering_recall_n"] for s in per_seed)
    print("{:<28}{} (n={} vanilla high-weight rules checked, theta_hi={})".format(
        "Filtering recall", fmt(agg["filtering_recall"], pct=True, decimals=1), fr_n, args.hcr_threshold))
    print("{:<28}{}".format("EV->weight Pearson", fmt(agg["ev_weight_pearson"], decimals=3)))
    print("{:<28}{}".format("EV->weight Spearman", fmt(agg["ev_weight_spearman"], decimals=3)))
    print("{:<28}{}".format("HCR (theta_hi={})".format(args.hcr_threshold), fmt(agg["hcr_emvr"], pct=True, decimals=1, already_pct=True)))
    print("{:<28}{}".format("RCS", fmt_single(rcs_emvr, decimals=3)))
    print("{:<28}{}".format("RQI", fmt_single(rqi_emvr, decimals=2)))
    if rcs_emvr is None:
        print("\n[NOTE] RCS/RQI require the Lv et al. 2021 WD15K human interpretability")
        print("annotations (--wd15k_annotations). Only defined for WD15K -- N/A elsewhere.")

    print("\nVERDICT")
    line()
    v_mrr_mean, v_mrr_std = vanilla_acc["mrr"]
    e_mrr_mean, e_mrr_std = emvr_acc["mrr"]
    if v_mrr_mean is not None and e_mrr_mean is not None:
        gap = e_mrr_mean - v_mrr_mean
        combined_std = (v_mrr_std or 0) + (e_mrr_std or 0)
        if n_seeds == 1:
            print("Single seed only -- preliminary signal, not a validated result.")
            print("MRR delta: {:+.4f} ({:+.2f}%).".format(gap, gap / v_mrr_mean * 100))
        elif abs(gap) > combined_std:
            winner = "EMVR-KBC" if gap > 0 else "LeSR"
            print("{} wins on MRR: {:+.4f} ({:+.2f}%), gap exceeds combined std ({:.4f})".format(
                winner, gap, gap / v_mrr_mean * 100, combined_std))
            print("across {} seeds -- this is a real result, not noise.".format(n_seeds))
        else:
            print("MRR delta ({:+.4f}) is within combined std ({:.4f}) across {} seeds --".format(
                gap, combined_std, n_seeds))
            print("NOT a confirmed win either way.")
    print("=" * W)