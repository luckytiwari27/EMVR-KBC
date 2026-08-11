"""
emvr_metrics.py -- corrected, scale-aware metrics for EMVR-KBC vs LeSR.

This module REPLACES the metric helpers previously used by compare_runs.py.
It fixes three concrete problems in the original implementation:

  (1) FILTERING RECALL was undefined in practice. Eq. (18) uses a fixed global
      threshold theta_hi (0.5) on w_i, but w_i is a per-relation softmax over
      (n_r + 1) entries, so its natural scale is 1/n_r and varies by relation.
      With ~25 rules/relation the maximum attainable weight is ~0.04, so the
      high-weight set was always empty -> denominator 0 -> "N/A".
      FIX: scale-free definitions -- "relative" (w_i > kappa / n_r) or
      "topk" (top ceil(frac * n_r) rules per relation). Both are invariant to
      how many rules a relation happens to have.

  (2) EV->WEIGHT CORRELATION was pooled across relations, mixing an absolute
      [0,1] scale (EV_i) with a per-relation 1/n_r scale (w_i). That mixing
      drives the correlation toward zero or negative regardless of the true
      per-relation relationship.
      FIX: compute per relation, then aggregate. Also provides a scale-free
      pooled variant (within-relation percentile ranks) and reports the old
      pooled-raw number explicitly labelled as the broken one, for contrast.

  (3) WEIGHT VECTOR ALIGNMENT. The old code always did weights[:-1] to strip
      the KGE slot. That is correct for ReasonerModel (softmax over n_rules+1)
      but WRONG for ReasonerModelPlus (softmax over n_rules only, alpha holds
      the embedding weight separately) -- it silently dropped a real rule and
      shifted every rule's weight by one position.
      FIX: detect the layout from len(weights) vs len(rule_texts), and report
      any relation whose files do not line up instead of skipping it silently.

Additionally provides a WEIGHT-LEARNING HEALTH CHECK, which detects the
failure mode where the reasoner's optimiser schedule is too short/too damped
for the softmax weights to move away from uniform (which reproduces the base
paper's "w/o weight learning" ablation rather than its main result).

No dependency on reasoner.py / evidence.py -- torch and numpy only, plus
scipy if available (optional, for tie-corrected Spearman).
"""

import glob
import json
import math
import os

import numpy as np
import torch

try:
    from scipy import stats as _scipy_stats
except Exception:  # scipy optional
    _scipy_stats = None


# ==========================================================================
# Published LeSR reference numbers (GPT-3.5 backbone, WITH weight learning),
# He et al. 2026, IEEE TASLP, Tables V-VII. Hardcoded reference only -- never
# computed from your runs. The "w/o WL" row is included because reproductions
# that land on it are diagnosing an inert reasoner, not an LLM/data problem.
# ==========================================================================

PAPER_REFERENCE = {
    "UMLs":       {"mr": 4.1,    "mrr": 0.764, "hit1": 0.682, "hit3": 0.815, "hit10": 0.918},
    "WN18RR":     {"mr": 1989.0, "mrr": 0.497, "hit1": 0.440, "hit3": 0.523, "hit10": 0.610},
    "FB15K":      {"mr": 124.3,  "mrr": 0.420, "hit1": 0.327, "hit3": 0.461, "hit10": 0.598},
    "WD15K":      {"mr": 95.0,   "mrr": 0.570, "hit1": 0.453, "hit3": 0.649, "hit10": 0.771},
    "ConceptNet": {"mr": 5866.5, "mrr": 0.345, "hit1": 0.227, "hit3": 0.419, "hit10": 0.567},
}

PAPER_REFERENCE_NO_WL = {
    "UMLs":       {"mr": 10.6,   "mrr": 0.364, "hit1": 0.224, "hit3": 0.406, "hit10": 0.677},
    "WN18RR":     {"mr": 1985.9, "mrr": 0.460, "hit1": 0.380, "hit3": 0.507, "hit10": 0.605},
    "FB15K":      {"mr": 128.9,  "mrr": 0.374, "hit1": 0.281, "hit3": 0.417, "hit10": 0.551},
    "WD15K":      {"mr": 98.4,   "mrr": 0.520, "hit1": 0.417, "hit3": 0.574, "hit10": 0.716},
    "ConceptNet": {"mr": 5879.6, "mrr": 0.190, "hit1": 0.098, "hit3": 0.205, "hit10": 0.407},
}


# ==========================================================================
# 1. Loading a completed run
# ==========================================================================

def relation_ids_in_run(run_dir):
    """Relation ids that produced test ranks in this run."""
    pattern = os.path.join(run_dir, "reasoner", "*_test_ranks.pt")
    ids = [os.path.basename(f).replace("_test_ranks.pt", "") for f in glob.glob(pattern)]
    return sorted(ids, key=lambda x: int(x))


def load_rule_texts(run_dir, rid):
    """_rules.json is JSON Lines (one JSON array per line), written by
    lesr.py::save_nested_list -- NOT a single JSON document."""
    fname = os.path.join(run_dir, "reasoner", "{}_rules.json".format(rid))
    if not os.path.exists(fname):
        return []
    rules = []
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if line:
                rules.append(json.loads(line))
    return [r[0] for r in rules]


def _load_weight_file(run_dir, rid):
    """Returns (weight_vector_np or None, alpha_float or None)."""
    fname = os.path.join(run_dir, "reasoner", "{}_learned_weights.pt".format(rid))
    if not os.path.exists(fname):
        return None, None
    obj = torch.load(fname, map_location="cpu")
    alpha = None
    if isinstance(obj, dict):
        alpha_obj = obj.get("logical_alpha", None)
        if alpha_obj is not None:
            alpha = float(alpha_obj) if not hasattr(alpha_obj, "item") else float(alpha_obj.item())
        obj = obj["logical_weights"]
    w = obj.detach().cpu().numpy() if hasattr(obj, "detach") else np.asarray(obj)
    return np.asarray(w, dtype=float), alpha


def _load_json(run_dir, rid, suffix):
    fname = os.path.join(run_dir, "reasoner", "{}_{}.json".format(rid, suffix))
    if not os.path.exists(fname):
        return None
    with open(fname) as f:
        return json.load(f)


def load_run(run_dir, label=""):
    """
    Loads one completed --run_reasoner directory into a dict:

      {
        "label", "run_dir",
        "relations": {rid: {rule_texts, w_logical, w_kge, alpha, ev, emvr_stats}},
        "ranks": concatenated test ranks tensor (or None),
        "misaligned": [(rid, n_rules, n_weights), ...],   # never silent
        "model_variant": "base" | "plus" | "unknown",
      }

    Weight-vector layout is detected, not assumed:
      len(w) == n_rules + 1  -> ReasonerModel      (last slot = KGE/embedding)
      len(w) == n_rules      -> ReasonerModelPlus  (embedding weight is alpha)
    """
    if not os.path.isdir(os.path.join(run_dir, "reasoner")):
        raise FileNotFoundError(
            "{} has no reasoner/ subfolder -- is this a completed --run_reasoner run?".format(run_dir))

    rids = relation_ids_in_run(run_dir)
    relations, misaligned, variants = {}, [], set()
    rank_chunks = []

    for rid in rids:
        rank_f = os.path.join(run_dir, "reasoner", "{}_test_ranks.pt".format(rid))
        if os.path.exists(rank_f):
            rank_chunks.append(torch.load(rank_f, map_location="cpu"))

        rule_texts = load_rule_texts(run_dir, rid)
        w, alpha = _load_weight_file(run_dir, rid)
        n = len(rule_texts)

        w_logical, w_kge = None, None
        if w is not None and n > 0:
            if len(w) == n + 1:
                w_logical, w_kge = w[:n], float(w[n])
                variants.add("base")
            elif len(w) == n:
                w_logical, w_kge = w, None
                variants.add("plus")
            else:
                misaligned.append((rid, n, len(w)))

        ev = _load_json(run_dir, rid, "ev_scores")
        if ev is not None and w_logical is not None and len(ev) != len(w_logical):
            # defensive: EV file written before keep_good_rules trimming drift
            ev = (list(ev) + [0.0] * len(w_logical))[:len(w_logical)]

        relations[rid] = {
            "rule_texts": rule_texts,
            "w_logical": w_logical,
            "w_kge": w_kge,
            "alpha": alpha,
            "ev": ev,
            "emvr_stats": _load_json(run_dir, rid, "emvr_stats"),
        }

    if len(variants) == 1:
        variant = variants.pop()
    elif len(variants) > 1:
        variant = "mixed(!)"
    else:
        variant = "unknown"

    return {
        "label": label or os.path.basename(os.path.normpath(run_dir)),
        "run_dir": run_dir,
        "relations": relations,
        "ranks": torch.cat(rank_chunks) if rank_chunks else None,
        "misaligned": misaligned,
        "model_variant": variant,
    }


# ==========================================================================
# 2. Accuracy
# ==========================================================================

def accuracy_from_ranks(ranks, k_vals=(1, 3, 10)):
    """Same computation as reasoner.compute_metrics, reimplemented so this
    module has no import-time dependency on the training code."""
    if ranks is None or len(ranks) == 0:
        return None
    r = ranks.float()
    out = {"n_query": int(r.shape[0]),
           "mr": float(torch.mean(r)),
           "mrr": float(torch.mean(1.0 / r))}
    for k in k_vals:
        out["hit{}".format(k)] = float(torch.mean((ranks <= k).float()))
    return out


# ==========================================================================
# 3. Weight-learning health check
# ==========================================================================

def weight_learning_health(run, min_rules=3):
    """
    Detects whether the softmax rule weights actually moved away from uniform.

    concentration = w_max * n_r   -- 1.0 means perfectly uniform (== the base
                                     paper's "w/o weight learning" ablation);
                                     larger means the reasoner committed to
                                     specific rules.
    eff_frac      = exp(H(w)) / n_r  -- effective fraction of rules in play;
                                     1.0 uniform, ->0 fully concentrated.

    A run whose mean concentration is below ~1.5 is, for practical purposes,
    NOT doing weight learning, whatever the flags said.
    """
    concentrations, eff_fracs, spreads, ns, alphas = [], [], [], [], []
    for rid, rec in run["relations"].items():
        w = rec["w_logical"]
        if w is None or len(w) < min_rules:
            continue
        n = len(w)
        p = np.clip(w / max(w.sum(), 1e-12), 1e-12, 1.0)
        concentrations.append(float(w.max()) * n)
        eff_fracs.append(float(np.exp(-np.sum(p * np.log(p)))) / n)
        spreads.append(float(w.max() / max(w.min(), 1e-12)))
        ns.append(n)
        if rec["alpha"] is not None:
            alphas.append(rec["alpha"])

    if not concentrations:
        return None

    mean_conc = float(np.mean(concentrations))
    if mean_conc < 1.15:
        verdict = "INERT -- weights are uniform; this reproduces the paper's 'w/o WL' row"
    elif mean_conc < 1.5:
        verdict = "WEAK -- weights barely moved; increase lr / epochs, disable LR decay"
    elif mean_conc < 4.0:
        verdict = "ACTIVE -- weight learning is working"
    else:
        verdict = "SHARP -- weights are highly concentrated; check for overfitting"

    return {
        "n_relations": len(concentrations),
        "mean_n_rules": float(np.mean(ns)),
        "mean_concentration": mean_conc,
        "max_concentration": float(np.max(concentrations)),
        "mean_eff_frac": float(np.mean(eff_fracs)),
        "mean_maxmin_ratio": float(np.mean(spreads)),
        "mean_alpha": float(np.mean(alphas)) if alphas else None,
        "verdict": verdict,
    }


# ==========================================================================
# 4. High-confidence rule sets (scale-free)
# ==========================================================================

def high_weight_indices(w, mode="relative", kappa=3.0, frac=0.2, theta=0.5):
    """
    Which rules of a relation count as 'high significance'?

      relative : w_i > kappa / n_r        (kappa x the uniform weight)
      topk     : top ceil(frac * n_r) rules by weight
      absolute : w_i > theta              (the original Eq. 18 definition;
                                           kept only for backwards comparison
                                           -- essentially always empty)
    """
    n = len(w)
    if n == 0:
        return np.array([], dtype=int)
    if mode == "relative":
        return np.where(w > (kappa / n))[0]
    if mode == "topk":
        k = max(1, int(math.ceil(frac * n)))
        return np.argsort(w)[-k:][::-1]
    if mode == "absolute":
        return np.where(w > theta)[0]
    raise ValueError("unknown mode {}".format(mode))


def high_confidence_rule_ratio(run, mode="relative", kappa=3.0, frac=0.2, theta=0.5):
    """HCR as a percentage, averaged over relations."""
    vals = []
    for rec in run["relations"].values():
        w = rec["w_logical"]
        if w is None or len(w) == 0:
            continue
        idx = high_weight_indices(w, mode, kappa, frac, theta)
        vals.append(100.0 * len(idx) / len(w))
    return float(np.mean(vals)) if vals else None


# ==========================================================================
# 5. Filtering Recall (Eq. 18, corrected)
# ==========================================================================

def filtering_recall(vanilla_run, emvr_run, mode="relative",
                     kappa=3.0, frac=0.2, theta=0.5):
    """
    Fraction of the rules that vanilla LeSR ended up trusting which survive
    EMVR-KBC's pre-grounding verification.

    Returns micro (pooled over rules), macro (mean of per-relation recalls),
    the denominator, and the per-relation breakdown, so a low number can be
    traced to specific relations instead of appearing as a bare "N/A".

    REQUIRES a vanilla run: the weights of rules EMVR discarded are only
    observable in a run where those rules were actually grounded and trained.
    There is no way to reconstruct them from the EMVR run alone.
    """
    hits_total, high_total = 0, 0
    per_relation, macro = [], []

    for rid, emvr_rec in emvr_run["relations"].items():
        van_rec = vanilla_run["relations"].get(rid)
        if van_rec is None or van_rec["w_logical"] is None:
            continue
        van_w, van_texts = van_rec["w_logical"], van_rec["rule_texts"]
        if len(van_texts) != len(van_w):
            continue

        idx = high_weight_indices(van_w, mode, kappa, frac, theta)
        if len(idx) == 0:
            continue

        high_texts = {van_texts[i] for i in idx}
        kept = high_texts & set(emvr_rec["rule_texts"])
        hits_total += len(kept)
        high_total += len(high_texts)
        rel_recall = len(kept) / len(high_texts)
        macro.append(rel_recall)
        per_relation.append({
            "relation_id": rid,
            "n_high": len(high_texts),
            "n_kept": len(kept),
            "recall": rel_recall,
            "lost_rules": sorted(high_texts - kept),
        })

    if high_total == 0:
        return {"micro": None, "macro": None, "n_high_rules": 0,
                "n_relations": 0, "per_relation": [],
                "reason": "no vanilla rule cleared the high-weight bar under mode='{}' "
                          "(if mode='absolute', this is expected -- see module docstring)".format(mode)}

    return {
        "micro": hits_total / high_total,
        "macro": float(np.mean(macro)),
        "n_high_rules": high_total,
        "n_relations": len(macro),
        "per_relation": sorted(per_relation, key=lambda d: d["recall"]),
        "reason": None,
    }


# ==========================================================================
# 6. EV -> weight correlation (corrected)
# ==========================================================================

def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    if _scipy_stats is not None:
        rho = _scipy_stats.spearmanr(x, y).correlation
        return None if (rho is None or np.isnan(rho)) else float(rho)
    xr, yr = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    if np.std(xr) == 0 or np.std(yr) == 0:
        return None
    return float(np.corrcoef(xr, yr)[0, 1])


def _pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _percentile_rank(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return np.zeros_like(v)
    return np.argsort(np.argsort(v)) / (len(v) - 1.0)


def ev_weight_correlation(run, min_rules=5):
    """
    Correlation between the pre-grounding evidence score EV_i and the final
    post-training significance weight w_i (the assumption Section XIII-C.1 of
    the proposal exists to test).

    Computed PER RELATION, then aggregated -- because w_i lives on a
    per-relation 1/n_r scale while EV_i is absolute, so pooling raw values
    across relations mixes scales and destroys the signal.

    Reports:
      per_relation_spearman / pearson : mean +/- std over relations, plus the
                                        share of relations with rho > 0
      pooled_rank_*                   : scale-free pooled version (values are
                                        converted to within-relation percentile
                                        ranks before pooling)
      pooled_raw_*                    : the OLD, broken pooled-raw number,
                                        reported only so the two can be
                                        compared side by side
    """
    rhos, rs, ns = [], [], []
    pooled_ev_rank, pooled_w_rank = [], []
    pooled_ev_raw, pooled_w_raw = [], []

    for rec in run["relations"].values():
        ev, w = rec["ev"], rec["w_logical"]
        if ev is None or w is None:
            continue
        m = min(len(ev), len(w))
        if m < min_rules:
            continue
        ev_v, w_v = np.asarray(ev[:m], float), np.asarray(w[:m], float)
        if np.std(ev_v) == 0 or np.std(w_v) == 0:
            continue

        rho, r = _spearman(ev_v, w_v), _pearson(ev_v, w_v)
        if rho is not None:
            rhos.append(rho)
        if r is not None:
            rs.append(r)
        ns.append(m)

        pooled_ev_rank.extend(_percentile_rank(ev_v).tolist())
        pooled_w_rank.extend(_percentile_rank(w_v).tolist())
        pooled_ev_raw.extend(ev_v.tolist())
        pooled_w_raw.extend(w_v.tolist())

    def agg(vals):
        if not vals:
            return {"mean": None, "std": None, "n": 0, "pct_positive": None}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "n": len(vals),
                "pct_positive": 100.0 * float(np.mean([v > 0 for v in vals]))}

    return {
        "spearman": agg(rhos),
        "pearson": agg(rs),
        "n_relations_used": len(ns),
        "n_rules_used": int(np.sum(ns)) if ns else 0,
        "min_rules": min_rules,
        "pooled_rank_spearman": _spearman(pooled_ev_rank, pooled_w_rank) if pooled_ev_rank else None,
        "pooled_raw_pearson": _pearson(pooled_ev_raw, pooled_w_raw) if pooled_ev_raw else None,
        "pooled_raw_spearman": _spearman(pooled_ev_raw, pooled_w_raw) if pooled_ev_raw else None,
    }


# ==========================================================================
# 7. Verification / efficiency stats
# ==========================================================================

def verification_stats(run):
    """Filtering rate, candidate/verified counts, mean EV, mean tau_r."""
    n_cand, n_ver = 0, 0
    all_ev, taus, kept_ev, dropped_ev = [], [], [], []

    for rec in run["relations"].values():
        st = rec["emvr_stats"]
        if st is None:
            continue
        n_cand += st.get("n_candidates", 0)
        n_ver += st.get("n_verified", 0)
        if st.get("tau_r") is not None:
            taus.append(st["tau_r"])
        for row in st.get("rows", []):
            ev = row.get("ev_i")
            if ev is None:
                continue
            all_ev.append(ev)
            (kept_ev if row.get("verified") else dropped_ev).append(ev)

    if n_cand == 0:
        return None
    return {
        "n_candidates": n_cand,
        "n_verified": n_ver,
        "filtering_rate": 1.0 - n_ver / n_cand,
        "mean_ev_all": float(np.mean(all_ev)) if all_ev else None,
        "mean_ev_kept": float(np.mean(kept_ev)) if kept_ev else None,
        "mean_ev_dropped": float(np.mean(dropped_ev)) if dropped_ev else None,
        "mean_tau_r": float(np.mean(taus)) if taus else None,
    }


def grounding_flops_saved(n_candidates, n_verified, n_entities,
                          sample_size=500, avg_degree=1.0):
    """Eq. (15)-(17): illustrative operation counts."""
    cost_lesr = n_candidates * (n_entities ** 2)
    cost_emvr = n_candidates * (sample_size * avg_degree) + n_verified * (n_entities ** 2)
    if cost_lesr == 0:
        return None
    return {"cost_lesr": cost_lesr, "cost_emvr": cost_emvr,
            "pct_saved": 1.0 - cost_emvr / cost_lesr}


DATASET_STATS = {  # Table I of the base paper, for the FLOPs estimate
    "UMLs":       {"n_entities": 135,   "avg_degree": 48.60},
    "WN18RR":     {"n_entities": 40943, "avg_degree": 2.27},
    "FB15K":      {"n_entities": 14541, "avg_degree": 21.33},
    "WD15K":      {"n_entities": 15812, "avg_degree": 11.16},
    "ConceptNet": {"n_entities": 78339, "avg_degree": 1.31},
}


# ==========================================================================
# 8. Rule interpretability (WD15K only)
# ==========================================================================

def rule_clarity_score(run, annotations, mode="relative", kappa=3.0, frac=0.2, theta=0.5):
    """Mean interpretability score over the run's high-confidence rules.
    Requires the Lv et al. 2021 WD15K annotations; returns None otherwise."""
    if annotations is None:
        return None
    scores = []
    for rec in run["relations"].values():
        w, texts = rec["w_logical"], rec["rule_texts"]
        if w is None or len(texts) != len(w):
            continue
        for i in high_weight_indices(w, mode, kappa, frac, theta):
            paths = annotations.get(texts[i])
            if paths:
                scores.append(float(np.mean(paths)))
    return float(np.mean(scores)) if scores else None


def rule_quality_index(hcr, rcs):
    """RQI = 2 * HCR * RCS / (HCR + RCS) * 100, base paper Section V-B.
    HCR is a percentage (0-100), RCS a fraction (0-1)."""
    if hcr is None or rcs is None:
        return None
    h = hcr / 100.0
    if h + rcs == 0:
        return None
    return 2.0 * h * rcs / (h + rcs) * 100.0