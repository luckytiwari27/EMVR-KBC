"""
compare_emvr.py -- single-run (no seed sweep) EMVR-KBC vs LeSR report.

Replaces compare_runs.py. Differences that matter:

  * No seed averaging. ReasonerModel initialises raw_weights to torch.zeros,
    so with a shared extractor/proposer directory the vanilla reasoner run is
    deterministic and "3 seeds" was really 1 run reported three times. Seeds
    are therefore dropped rather than pretended.
  * Filtering Recall uses a scale-free high-weight definition (see
    emvr_metrics), so it produces a number instead of N/A.
  * EV->weight correlation is computed per relation and then aggregated.
  * Weight-vector layout (ReasonerModel vs ReasonerModelPlus) is detected
    rather than assumed.
  * A WEIGHT-LEARNING HEALTH CHECK runs first, because every downstream
    number is meaningless if the reasoner never left the uniform simplex.
  * Any arm may be omitted. With --vanilla_run_dir absent, the published
    LeSR numbers are used for the accuracy comparison; metrics that
    structurally require a vanilla run say so explicitly.

USAGE (all arms):
  python compare_emvr.py --dataset UMLs \
      --vanilla_run_dir runs/umls_wl_vanilla \
      --emvr_nowarm_run_dir runs/umls_wl_emvr_nowarm \
      --emvr_warm_run_dir runs/umls_wl_emvr_warm

USAGE (EMVR only, accuracy vs the published paper column):
  python compare_emvr.py --dataset UMLs \
      --emvr_warm_run_dir runs/umls_wl_emvr_warm
"""

import argparse
import json
import os
import sys

import numpy as np

import emvr_metrics as M


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   choices=["UMLs", "WN18RR", "FB15K", "WD15K", "ConceptNet"])
    p.add_argument("--vanilla_run_dir", default=None,
                   help="Vanilla LeSR reasoner run. Optional for accuracy, but REQUIRED "
                        "for Filtering Recall -- the weights of rules EMVR discarded only "
                        "exist in a run where those rules were actually grounded.")
    p.add_argument("--emvr_nowarm_run_dir", default=None,
                   help="EMVR with --use_emvr but WITHOUT --use_emvr_warmstart (isolates filtering).")
    p.add_argument("--emvr_warm_run_dir", default=None,
                   help="EMVR with --use_emvr AND --use_emvr_warmstart (full framework).")

    p.add_argument("--high_weight_mode", default="relative",
                   choices=["relative", "topk", "absolute"],
                   help="How a rule counts as high-significance. 'relative': w_i > kappa/n_r. "
                        "'topk': top frac*n_r rules. 'absolute': w_i > theta (legacy Eq.18; "
                        "essentially always empty under per-relation softmax).")
    p.add_argument("--kappa", type=float, default=3.0,
                   help="relative mode: multiple of the uniform weight 1/n_r.")
    p.add_argument("--topk_frac", type=float, default=0.2, help="topk mode: fraction of rules.")
    p.add_argument("--theta_hi", type=float, default=0.5, help="absolute mode threshold.")
    p.add_argument("--corr_min_rules", type=int, default=5,
                   help="Minimum verified rules for a relation to enter the correlation average.")
    p.add_argument("--sensitivity", action="store_true",
                   help="Also print Filtering Recall across a sweep of kappa values.")
    p.add_argument("--wd15k_annotations", default=None,
                   help="JSON {rule_text: [0/0.5/1, ...]} from Lv et al. 2021 (WD15K only).")
    p.add_argument("--show_lost_rules", type=int, default=5,
                   help="How many worst-recall relations to list in detail (0 to disable).")
    return p.parse_args()


W = 96
def line(ch="-"):
    print(ch * W)


def f(v, dec=4, pct=False, na="N/A"):
    if v is None:
        return na
    return "{:.{d}f}{}".format(v * 100 if pct else v, "%" if pct else "", d=dec)


def hw_kwargs(a):
    return dict(mode=a.high_weight_mode, kappa=a.kappa, frac=a.topk_frac, theta=a.theta_hi)


def describe_mode(a):
    if a.high_weight_mode == "relative":
        return "w_i > {:g} / n_r  (kappa x uniform weight)".format(a.kappa)
    if a.high_weight_mode == "topk":
        return "top {:.0%} of rules per relation".format(a.topk_frac)
    return "w_i > {:g}  (legacy fixed threshold)".format(a.theta_hi)


def main():
    a = parse_args()

    arms = []
    for label, d in [("LeSR (yours)", a.vanilla_run_dir),
                     ("EMVR no-warm", a.emvr_nowarm_run_dir),
                     ("EMVR warm", a.emvr_warm_run_dir)]:
        if d:
            arms.append((label, M.load_run(d, label)))
    if not arms:
        sys.exit("Give at least one of --vanilla_run_dir / --emvr_nowarm_run_dir / --emvr_warm_run_dir")

    runs = {label: run for label, run in arms}
    vanilla = runs.get("LeSR (yours)")
    emvr_arms = [(l, r) for l, r in arms if l != "LeSR (yours)"]

    annotations = None
    if a.wd15k_annotations:
        with open(a.wd15k_annotations) as fh:
            annotations = json.load(fh)

    print("=" * W)
    print("{}  --  EMVR-KBC report (single run per arm, no seed averaging)".format(a.dataset))
    print("=" * W)
    for label, run in arms:
        print("  {:<16} {:<52} model={}".format(label, run["run_dir"], run["model_variant"]))
        if run["misaligned"]:
            print("     [WARN] {} relation(s) with rule/weight length mismatch, excluded: {}".format(
                len(run["misaligned"]), run["misaligned"][:5]))
    print("  high-significance rule definition: {}".format(describe_mode(a)))

    # ------------------------------------------------------------------
    # 1. Weight-learning health check
    # ------------------------------------------------------------------
    print("\n1. WEIGHT-LEARNING HEALTH CHECK")
    line()
    print("{:<16}{:>10}{:>14}{:>12}{:>12}   {}".format(
        "Arm", "avg n_r", "w_max*n_r", "eff.frac", "alpha", "verdict"))
    line()
    inert = []
    for label, run in arms:
        h = M.weight_learning_health(run)
        if h is None:
            print("{:<16}{:>10}".format(label, "no data"))
            continue
        print("{:<16}{:>10.1f}{:>14.2f}{:>12.2f}{:>12}   {}".format(
            label, h["mean_n_rules"], h["mean_concentration"], h["mean_eff_frac"],
            f(h["mean_alpha"], 3), h["verdict"]))
        if h["mean_concentration"] < 1.5:
            inert.append(label)
    line()
    print("w_max*n_r = 1.00 means perfectly uniform weights, i.e. the base paper's")
    print("'w/o weight learning' ablation. Values >~1.5 mean weight learning is live.")
    if inert:
        print("\n[CRITICAL] {} still near-uniform. Raise --initial_lr / --num_epochs,".format(
            ", ".join(inert)))
        print("set --scheduler_gamma 1, and raise --early_stop_patience, then re-run.")
        print("Every number below is close to meaningless until this reads ACTIVE.")

    # ------------------------------------------------------------------
    # 2. Accuracy
    # ------------------------------------------------------------------
    print("\n2. CORE ACCURACY")
    line()
    paper = M.PAPER_REFERENCE[a.dataset]
    paper_nowl = M.PAPER_REFERENCE_NO_WL[a.dataset]
    acc = {label: M.accuracy_from_ranks(run["ranks"]) for label, run in arms}

    cols = ["Paper w/ WL", "Paper w/o WL"] + [l for l, _ in arms]
    header = "{:<14}".format("Metric") + "".join("{:>16}".format(c) for c in cols)
    print(header)
    line()

    def row(name, key, pct=False, dec=4):
        cells = []
        for src in (paper, paper_nowl):
            cells.append("{:.2f}%".format(src[key] * 100) if pct else "{:.4f}".format(src[key]))
        for label, _ in arms:
            v = acc[label][key] if acc[label] else None
            cells.append(f(v, dec=2 if pct else dec, pct=pct))
        print("{:<14}".format(name) + "".join("{:>16}".format(c) for c in cells))

    row("MR (down)", "mr")
    row("MRR (up)", "mrr")
    row("Hit@1 (up)", "hit1", pct=True)
    row("Hit@3 (up)", "hit3", pct=True)
    row("Hit@10 (up)", "hit10", pct=True)
    line()
    n_q = {l: (acc[l]["n_query"] if acc[l] else 0) for l, _ in arms}
    print("test queries per arm: " + ", ".join("{}={}".format(l, n) for l, n in n_q.items()))
    print("\n[Paper columns are fixed references from He et al. 2026 Tables V-VII (GPT-3.5).")
    print(" They are NOT recomputed from your runs. Only arms sharing your own rule pool")
    print(" are a controlled comparison; the paper columns are a sanity check. If your")
    print(" arms sit on the 'w/o WL' column, fix section 1 before reading anything else.]")
    if not vanilla:
        print("\n[NOTE] No --vanilla_run_dir given, so 'EMVR vs LeSR' below is against the")
        print("published column only. That is not a controlled comparison: it differs in")
        print("LLM snapshot, rule pool and reasoner settings, not just in EMVR.")

    # ------------------------------------------------------------------
    # 3. Verification / efficiency
    # ------------------------------------------------------------------
    print("\n3. RULE VERIFICATION AND EFFICIENCY")
    line()
    ds = M.DATASET_STATS[a.dataset]
    any_stats = False
    for label, run in emvr_arms:
        vs = M.verification_stats(run)
        if vs is None:
            print("{:<16} no EMVR stats on disk (was --emvr_save_stats set?)".format(label))
            continue
        any_stats = True
        fl = M.grounding_flops_saved(vs["n_candidates"], vs["n_verified"],
                                     ds["n_entities"], avg_degree=ds["avg_degree"])
        print("{:<16} candidates={}  verified={}  filtering rate={}".format(
            label, vs["n_candidates"], vs["n_verified"], f(vs["filtering_rate"], 1, pct=True)))
        print("{:<16} mean EV: all={}  kept={}  dropped={}   mean tau_r={}".format(
            "", f(vs["mean_ev_all"], 3), f(vs["mean_ev_kept"], 3),
            f(vs["mean_ev_dropped"], 3), f(vs["mean_tau_r"], 3)))
        print("{:<16} grounding ops saved (Eq.15-17) = {}".format(
            "", f(fl["pct_saved"] if fl else None, 1, pct=True)))
        if fl and fl["pct_saved"] < 0:
            print("{:<16} [negative: S*d = {:.0f} exceeds |E|^2 = {}, so the sampling pass costs".format(
                "", 500 * ds["avg_degree"], ds["n_entities"] ** 2))
            print("{:<16}  more than the grounding it avoids on this KB. This is proposal".format(""))
            print("{:<16}  Sec. X-D's worst case made concrete -- report it, do not hide it.".format(""))
    if not any_stats:
        print("(no EMVR arm with saved stats)")

    # ------------------------------------------------------------------
    # 4. Filtering Recall
    # ------------------------------------------------------------------
    print("\n4. FILTERING RECALL (Eq. 18, scale-free)")
    line()
    if vanilla is None:
        print("N/A -- structurally requires a vanilla LeSR run.")
        print("Filtering Recall asks: of the rules LeSR itself ended up trusting, how many")
        print("survived verification? The discarded rules are never grounded in an EMVR run,")
        print("so their would-be weights do not exist anywhere in it, and no published table")
        print("can supply them (the paper reports metrics, not per-rule weights).")
        print("Fix: run the vanilla reasoner once on the SAME extractor/proposer directory.")
        print("It costs no LLM API calls -- see run_emvr_wl.ps1.")
    else:
        for label, run in emvr_arms:
            fr = M.filtering_recall(vanilla, run, **hw_kwargs(a))
            if fr["micro"] is None:
                print("{:<16} N/A -- {}".format(label, fr["reason"]))
                continue
            print("{:<16} micro={}  macro={}   (n_high={} rules over {} relations)".format(
                label, f(fr["micro"], 1, pct=True), f(fr["macro"], 1, pct=True),
                fr["n_high_rules"], fr["n_relations"]))
            if a.show_lost_rules:
                worst = [r for r in fr["per_relation"] if r["recall"] < 1.0][:a.show_lost_rules]
                for r in worst:
                    print("    rel {:<5} recall={:.0%}  kept {}/{}  e.g. dropped: {}".format(
                        r["relation_id"], r["recall"], r["n_kept"], r["n_high"],
                        (r["lost_rules"][0][:70] + "...") if r["lost_rules"] else "-"))

        if a.sensitivity and a.high_weight_mode == "relative":
            print("\n  sensitivity to kappa (relative mode):")
            print("  {:<16}".format("kappa") + "".join("{:>12}".format(k) for k in [1.5, 2, 3, 5, 10]))
            for label, run in emvr_arms:
                cells = []
                for k in [1.5, 2, 3, 5, 10]:
                    fr = M.filtering_recall(vanilla, run, mode="relative", kappa=k)
                    cells.append("{} (n={})".format(f(fr["micro"], 0, pct=True), fr["n_high_rules"])
                                 if fr["micro"] is not None else "N/A")
                print("  {:<16}".format(label) + "".join("{:>12}".format(c) for c in cells))
        print("\nRecall near 1.0 means verification kept what LeSR would have valued. Note this")
        print("measures agreement with LeSR's own statistical weights, not ground-truth rule")
        print("correctness (proposal Sec. XIII-B).")

    # ------------------------------------------------------------------
    # 5. EV -> weight correlation
    # ------------------------------------------------------------------
    print("\n5. EV -> WEIGHT CORRELATION (per relation, then aggregated)")
    line()
    for label, run in emvr_arms:
        c = M.ev_weight_correlation(run, min_rules=a.corr_min_rules)
        if c["n_relations_used"] == 0:
            print("{:<16} N/A -- no relation had >= {} verified rules with EV scores saved "
                  "(need --emvr_save_stats)".format(label, a.corr_min_rules))
            continue
        sp, pe = c["spearman"], c["pearson"]
        print("{:<16} Spearman {} +/- {}   ({}/{} relations positive)".format(
            label, f(sp["mean"], 3), f(sp["std"], 3),
            int(round((sp["pct_positive"] or 0) / 100 * sp["n"])), sp["n"]))
        print("{:<16} Pearson  {} +/- {}   over {} rules in {} relations".format(
            "", f(pe["mean"], 3), f(pe["std"], 3), c["n_rules_used"], c["n_relations_used"]))
        print("{:<16} pooled (percentile-rank, scale-free): {}".format(
            "", f(c["pooled_rank_spearman"], 3)))
        print("{:<16} pooled RAW [old, broken -- mixes 1/n_r scales]: Pearson {}  Spearman {}".format(
            "", f(c["pooled_raw_pearson"], 3), f(c["pooled_raw_spearman"], 3)))
    print("\nThis is the test in proposal Sec. XIII-C.1. A clearly positive per-relation")
    print("correlation supports the warm-start design; near zero means EV_i does not")
    print("predict learned significance on this dataset, and any accuracy gain should be")
    print("attributed to filtering (compare the no-warm and warm arms above).")

    # ------------------------------------------------------------------
    # 6. Rule quality
    # ------------------------------------------------------------------
    print("\n6. RULE QUALITY")
    line()
    print("{:<16}{:>14}{:>14}{:>10}{:>10}".format("Arm", "HCR(scale-free)", "HCR(w>0.5)", "RCS", "RQI"))
    line()
    for label, run in arms:
        hcr = M.high_confidence_rule_ratio(run, **hw_kwargs(a))
        hcr_abs = M.high_confidence_rule_ratio(run, mode="absolute", theta=a.theta_hi)
        rcs = M.rule_clarity_score(run, annotations, **hw_kwargs(a))
        rqi = M.rule_quality_index(hcr, rcs)
        print("{:<16}{:>14}{:>14}{:>10}{:>10}".format(
            label, f(hcr, 1) + "%" if hcr is not None else "N/A",
            f(hcr_abs, 1) + "%" if hcr_abs is not None else "N/A",
            f(rcs, 3), f(rqi, 2)))
    if annotations is None:
        print("\n[RCS/RQI need the Lv et al. 2021 WD15K annotations (--wd15k_annotations);")
        print(" WD15K only, N/A elsewhere. Never fabricated.]")
    print("[HCR(w>0.5) is the legacy fixed-threshold version and is not comparable across")
    print(" arms: filtering changes n_r, which changes the weight scale. Use the scale-free")
    print(" column for any claim.]")

    # ------------------------------------------------------------------
    # 7. Ablation deltas
    # ------------------------------------------------------------------
    print("\n7. ABLATION DELTAS (MRR)")
    line()
    base_label, base = ("LeSR (yours)", acc.get("LeSR (yours)"))
    if base is None:
        base_label, base = "Paper w/ WL", {"mrr": paper["mrr"]}
        print("[baseline = published LeSR, not a controlled comparison]")
    for label in ["EMVR no-warm", "EMVR warm"]:
        if acc.get(label) is None:
            continue
        d = acc[label]["mrr"] - base["mrr"]
        print("{:<16} vs {:<14} {:+.4f} ({:+.2f}%)".format(
            label, base_label, d, 100 * d / base["mrr"]))
    if acc.get("EMVR no-warm") and acc.get("EMVR warm"):
        d = acc["EMVR warm"]["mrr"] - acc["EMVR no-warm"]["mrr"]
        print("{:<16} vs {:<14} {:+.4f} ({:+.2f}%)   <- the warm-start effect alone".format(
            "EMVR warm", "EMVR no-warm", d, 100 * d / acc["EMVR no-warm"]["mrr"]))
    else:
        print("[run both --emvr_nowarm_run_dir and --emvr_warm_run_dir to separate the")
        print(" filtering effect from the warm-start effect -- otherwise they are confounded.]")
    print("=" * W)


if __name__ == "__main__":
    main()