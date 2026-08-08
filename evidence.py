"""
evidence.py — EMVR-KBC: Evidence-guided Multi-stage Verification and Refinement for KBC

This module implements the *only* new component EMVR-KBC adds on top of LeSR:
a cheap, sampling-based evidence verification stage that sits between
`prepare_rules()` and `ground_rules_over_kg_semantic()` in the original pipeline.

Design goals (mirrors EMVR-KBC_Final_Proposal.pdf, Sections VI-X):
  1. Zero modification to reasoner.py / lesr.py class definitions.
  2. O(S·d) sampled evidence scoring per candidate rule instead of O(|E|^2) full
     grounding, with S fixed (default 500) regardless of |E| (Eq. 1, Eq. 16).
  3. Adaptive per-relation verification threshold tau_r (Eq. 6-7).
  4. Optional external corroboration via sentence-transformer + FAISS, only for
     very sparse relations (Eq. 4-5), reusing the same alignment idea LeSR
     already uses for relation-name mapping (no new alignment mechanism).
  5. A warm-start helper that produces logits for ReasonerModel.raw_weights
     equal to the pre-grounding evidence scores (Eq. 10) — the loss function
     itself (train_loop in reasoner.py) is untouched.
  6. Post-hoc metrics: Filtering Rate, Filtering Recall (Eq. 18), and
     EV-Weight correlation, so the efficiency/safety claims can be measured
     empirically per dataset rather than assumed.

Engineering note on internal evidence (Sec. VI-A of the proposal):
LeSR's own `ground_rules_over_kg_semantic()` computes C_i/A_i using dense-ish
sparse tensor products across the *entire* entity space for every rule type
(the eight tensor_logic_* cases). EMVR-KBC instead enumerates body-satisfying
(h,t) pairs via plain adjacency-dict BFS/chain-join, capped at S entity pairs,
which is the practical equivalent of the paper's "sparse BFS traversal" and
is what makes the verification stage cheap. The chain directions ("->"/"<-")
per rule_type below are copied verbatim from the rtemplate logic already
present in reasoner.py's `_prepare_rule_check_type_relax`, so no new rule
grammar is introduced.
"""

import math
import random
from collections import defaultdict

import numpy as np
import torch

from reasoner import prepare_rule_map_relations, RELATION_ID2Text_MAPPING_MODE  # reused, unchanged
from data import remove_wikidata_prefix

RAND_SEED = 5
random.seed(RAND_SEED)
np.random.seed(RAND_SEED)

# Direction table copied from reasoner.py::_prepare_rule_check_type_relax
# (rtemplate), so evidence.py never invents new rule-grammar semantics.
RULE_TYPE_BODY_DIRECTIONS = {
    "01": ["->"],
    "02": ["<-"],
    "11": ["->", "->"],
    "12": ["<-", "->"],
    "13": ["->", "<-"],
    "14": ["<-", "<-"],
    "21": ["->", "->", "->"],
    "22": ["->", "->", "<-"],
    "23": ["->", "<-", "->"],
    "24": ["<-", "->", "->"],
    "25": ["->", "<-", "<-"],
    "26": ["<-", "->", "<-"],
    "27": ["<-", "<-", "->"],
    "28": ["<-", "<-", "<-"],
}


# --------------------------------------------------------------------------
# 1. Cheap adjacency index (replaces LeSR's dense-ish |E|x|E| grounding for
#    the purpose of evidence sampling only; full grounding is untouched and
#    still runs, exactly once, on verified rules).
# --------------------------------------------------------------------------

def build_adjacency(train_arr):
    """
    train_arr: iterable of (h, r, t) integer-encoded triples (as produced by
               data.encode_kg_to_arr).
    Returns:
      fwd[r][h] = set(t)   # facts stored as (h, r, t)
      bwd[r][t] = set(h)   # same facts, indexed by tail
    """
    fwd = defaultdict(lambda: defaultdict(set))
    bwd = defaultdict(lambda: defaultdict(set))
    for h, r, t in train_arr:
        h, r, t = int(h), int(r), int(t)
        fwd[r][h].add(t)
        bwd[r][t].add(h)
    return fwd, bwd


def _hop_adjacency(fwd, bwd, rel_id, direction):
    """Return a dict x -> set(y) for a single chain hop."""
    if direction == "->":
        return fwd.get(rel_id, {})
    elif direction == "<-":
        return bwd.get(rel_id, {})
    else:
        raise ValueError("invalid direction {}".format(direction))


# --------------------------------------------------------------------------
# 2. Internal KB evidence: KBSample(phi_i)  (Eq. 1)
# --------------------------------------------------------------------------

def _enumerate_body_pairs(fwd, bwd, body_rel_ids, directions, max_frontier=20000):
    """
    Chain-joins the body relations hop by hop using plain dict adjacency
    (sparse BFS traversal — no |E|x|E| matrix is ever built).

    max_frontier bounds the number of (start) keys we keep expanding at each
    hop, protecting against combinatorial blow-up on very dense relations;
    this is a practical safeguard beyond what the proposal specifies and is
    documented here rather than silently applied.

    Returns: dict start_entity -> set(end_entity), i.e. the full census C_i
    (not yet subsampled to S=500 — subsampling happens in kb_sample_rule).
    """
    if len(body_rel_ids) == 0:
        return {}
    adj0 = _hop_adjacency(fwd, bwd, body_rel_ids[0], directions[0])
    frontier = {a: set(bs) for a, bs in adj0.items()}
    if len(frontier) > max_frontier:
        keep = set(random.sample(list(frontier.keys()), max_frontier))
        frontier = {a: b for a, b in frontier.items() if a in keep}
    for rel_id, direction in zip(body_rel_ids[1:], directions[1:]):
        adj = _hop_adjacency(fwd, bwd, rel_id, direction)
        new_frontier = defaultdict(set)
        for a, currents in frontier.items():
            for c in currents:
                if c in adj:
                    new_frontier[a].update(adj[c])
        frontier = dict(new_frontier)
        if len(frontier) > max_frontier:
            keep = set(random.sample(list(frontier.keys()), max_frontier))
            frontier = {a: b for a, b in frontier.items() if a in keep}
    return frontier


def kb_sample_rule(checked_rule, relation2id_dict, fwd, bwd, sample_size=500, verbose=False):
    """
    Implements Eq. (1): KBSample(phi_i) = |{(h,t) in S : A_i(h,t)>0}| / max(1, |{(h,t) in S : C_i(h,t)>0}|)

    checked_rule: [rule_text, rule_type, rule_cond (if_triplets), rule_head (then_triplet)]
                  exactly the format returned by reasoner.prepare_rule_check_type /
                  reasoner.prepare_rules.

    Returns dict with keys: kbsample, c_size (|C_i|, full census), a_size (|A_i| on sample),
                             sample_size_used, verifiable (bool)
    """
    rule_text, rule_type, rule_cond, rule_head = checked_rule
    relation2id_dict_clean = {remove_wikidata_prefix(k): v for k, v in relation2id_dict.items()}
    rel_ids = prepare_rule_map_relations(rule_cond, rule_head, relation2id_dict_clean, verbose=verbose)
    if rel_ids is None:
        return {"kbsample": 0.0, "c_size": 0, "a_size": 0, "sample_size_used": 0, "verifiable": False}

    body_rel_ids, head_rel_id = rel_ids[:-1], rel_ids[-1]
    directions = RULE_TYPE_BODY_DIRECTIONS.get(rule_type)
    if directions is None or len(directions) != len(body_rel_ids):
        return {"kbsample": 0.0, "c_size": 0, "a_size": 0, "sample_size_used": 0, "verifiable": False}

    frontier = _enumerate_body_pairs(fwd, bwd, body_rel_ids, directions)
    pairs = [(a, b) for a, bs in frontier.items() for b in bs]
    c_size = len(pairs)
    if c_size == 0:
        return {"kbsample": 0.0, "c_size": 0, "a_size": 0, "sample_size_used": 0, "verifiable": True}

    if c_size > sample_size:
        sample_pairs = random.sample(pairs, sample_size)
    else:
        sample_pairs = pairs  # census, matches Eq.(1)/(2) zero-sampling-error case

    head_fwd = fwd.get(head_rel_id, {})
    a_hits = sum(1 for (a, b) in sample_pairs if b in head_fwd.get(a, ()))
    kbsample = a_hits / max(1, len(sample_pairs))
    return {
        "kbsample": kbsample,
        "c_size": c_size,
        "a_size": a_hits,
        "sample_size_used": len(sample_pairs),
        "verifiable": True,
    }


# --------------------------------------------------------------------------
# 3. Optional external evidence: ExternalSim(phi_i)  (Eq. 4)
# --------------------------------------------------------------------------

class ExternalEvidenceIndex:
    """
    Thin wrapper around a sentence-transformer + FAISS flat index over a set
    of ConceptNet-style relation-chain pattern strings.

    Both dependencies are optional. If they are not installed, or no pattern
    file is supplied, `score()` returns 0.0 for every rule and EV_i falls
    back to KBSample alone (equivalent to beta=1.0 for that relation) —
    exactly the degradation the proposal accepts in Sec. XI-A.
    """

    def __init__(self, pattern_strings=None, model_name="BAAI/bge-large-en-v1.5"):
        self.available = False
        self.patterns = pattern_strings or []
        if len(self.patterns) == 0:
            return
        try:
            import faiss  # noqa: F401
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            embs = self.model.encode(self.patterns, normalize_embeddings=True)
            embs = np.asarray(embs, dtype="float32")
            self.index = faiss.IndexFlatIP(embs.shape[1])
            self.index.add(embs)
            self.available = True
        except Exception as e:  # pragma: no cover - optional path
            print("[evidence.py] external evidence disabled ({}). "
                  "Falling back to internal-only EV_i.".format(e))
            self.available = False

    def score(self, rule_relation_sequence_text):
        if not self.available:
            return 0.0
        q = self.model.encode([rule_relation_sequence_text], normalize_embeddings=True)
        q = np.asarray(q, dtype="float32")
        sims, _ = self.index.search(q, min(5, len(self.patterns)))
        return float(np.max(sims))


def rule_to_relation_sequence_text(checked_rule):
    """e.g. 'born_in -> located_in -> country' — used only as ExternalSim query text."""
    rule_text, rule_type, rule_cond, rule_head = checked_rule
    rels = [t[1] for t in rule_cond] + [rule_head[1]]
    return " -> ".join(r.replace("_", " ") for r in rels)


# --------------------------------------------------------------------------
# 4. Combine, adaptive threshold, verify  (Eq. 5-8)
# --------------------------------------------------------------------------

def compute_density(train_arr, n_entities, relation_id):
    n_facts_for_rel = sum(1 for (h, r, t) in train_arr if int(r) == relation_id)
    return n_facts_for_rel / max(1, n_entities ** 2)


def adaptive_threshold(density_r, tau_min=0.1, tau_max=0.5):
    return tau_min + (tau_max - tau_min) * density_r  # Eq. (7), corrected linear form


def combine_evidence(kbsample, external_sim, beta=0.7, density_r=None, external_density_gate=0.001):
    """Eq. (5). External term is only ever computed/used when density_r < gate."""
    if density_r is not None and density_r >= external_density_gate:
        return kbsample
    return beta * kbsample + (1 - beta) * external_sim


# --------------------------------------------------------------------------
# 5. Full per-relation verification pass (Sections VI-VII end to end)
# --------------------------------------------------------------------------

def verify_rules_for_relation(
    checked_rules,
    relation2id_dict,
    train_arr,
    n_entities,
    target_relation_id,
    fwd=None,
    bwd=None,
    sample_size=500,
    beta=0.7,
    tau_min=0.1,
    tau_max=0.5,
    external_density_gate=0.001,
    external_index=None,
    verbose=False,
):
    """
    Drop-in replacement for the "prepare_rules() -> ground_rules_over_kg_semantic()"
    gap identified in the proposal. Call this AFTER checked_rules is loaded/parsed
    and BEFORE ground_rules_over_kg_semantic() is called.

    Returns:
      verified_rules   : subset of checked_rules that passed verification (R_v)
      ev_scores        : list[float], EV_i for each rule in verified_rules, same order
      stats            : dict with density_r, tau_r, n_candidates, n_verified,
                          filtering_rate, per-rule diagnostic rows
    """
    if fwd is None or bwd is None:
        fwd, bwd = build_adjacency(train_arr)

    density_r = compute_density(train_arr, n_entities, target_relation_id)
    tau_r = adaptive_threshold(density_r, tau_min, tau_max)

    verified_rules, ev_scores, rows = [], [], []
    for checked_rule in checked_rules:
        kb_stats = kb_sample_rule(checked_rule, relation2id_dict, fwd, bwd,
                                   sample_size=sample_size, verbose=verbose)
        kbsample = kb_stats["kbsample"]
        ext_sim = 0.0
        if density_r < external_density_gate and external_index is not None and external_index.available:
            ext_sim = external_index.score(rule_to_relation_sequence_text(checked_rule))
        ev_i = combine_evidence(kbsample, ext_sim, beta=beta, density_r=density_r,
                                 external_density_gate=external_density_gate)
        verified = kb_stats["verifiable"] and (ev_i > tau_r)
        rows.append({
            "rule_text": checked_rule[0],
            "rule_type": checked_rule[1],
            "kbsample": kbsample,
            "ext_sim": ext_sim,
            "ev_i": ev_i,
            "c_size": kb_stats["c_size"],
            "a_size": kb_stats["a_size"],
            "verified": verified,
        })
        if verified:
            verified_rules.append(checked_rule)
            ev_scores.append(ev_i)

    n_candidates = len(checked_rules)
    n_verified = len(verified_rules)
    stats = {
        "density_r": density_r,
        "tau_r": tau_r,
        "n_candidates": n_candidates,
        "n_verified": n_verified,
        "filtering_rate": 1.0 - (n_verified / n_candidates) if n_candidates > 0 else 0.0,
        "rows": rows,
    }
    if verbose:
        print("[evidence.py] relation_id={}: density_r={:.5f} tau_r={:.3f} "
              "{}/{} rules verified ({:.1%} filtered before grounding)".format(
                  target_relation_id, density_r, tau_r, n_verified, n_candidates,
                  stats["filtering_rate"]))
    return verified_rules, ev_scores, stats


# --------------------------------------------------------------------------
# 6. Warm-start initialization for reasoner.ReasonerModel (Eq. 10)
#    NOTE: reasoner.py is not modified. We only set .data on the already-
#    constructed nn.Parameter, which softmaxes internally in forward().
# --------------------------------------------------------------------------

def compute_warmstart_logits(ev_scores, n_total_weights, kge_slot_value=0.0):
    """
    n_total_weights = n_logical_rules + 1 (the extra slot LeSR always appends
    for the KGE/embedding term; see lesr.py: `n_rules = n_rules + 1`).

    Because ReasonerModel.forward() applies softmax(raw_weights) itself,
    setting raw_weights := EV_i directly satisfies Eq. (10)
    (w_i^(0) = softmax(EV_i)) exactly, with no extra transform needed.
    The KGE slot is left at a neutral prior (default 0.0) since EV_i does not
    apply to the embedding term.
    """
    assert len(ev_scores) == n_total_weights - 1, (
        "expected {} verified-rule evidence scores (+1 KGE slot), got {}".format(
            n_total_weights - 1, len(ev_scores)))
    logits = torch.zeros(n_total_weights, dtype=torch.float32)
    logits[:len(ev_scores)] = torch.tensor(ev_scores, dtype=torch.float32)
    logits[-1] = kge_slot_value
    return logits


def apply_warmstart(model, ev_scores, kge_slot_value=0.0):
    """model: an already-constructed reasoner.ReasonerModel (or Plus variant)."""
    n_total = model.raw_weights.shape[0] + (1 if hasattr(model, "raw_alpha") else 0)
    if hasattr(model, "raw_alpha"):
        # ReasonerModelPlus: raw_weights covers only logical rules (num_rj - 1)
        logits = torch.tensor(ev_scores[:model.raw_weights.shape[0]], dtype=torch.float32)
        with torch.no_grad():
            model.raw_weights.copy_(logits.to(model.raw_weights.device))
    else:
        logits = compute_warmstart_logits(ev_scores, model.raw_weights.shape[0], kge_slot_value)
        with torch.no_grad():
            model.raw_weights.copy_(logits.to(model.raw_weights.device))
    return model


# --------------------------------------------------------------------------
# 7. Post-hoc evaluation metrics (Section XIII-B/C)
# --------------------------------------------------------------------------

def filtering_recall(verified_rule_texts, lesr_high_weight_rule_texts):
    """Eq. (18). Both args are sets/lists of rule_text strings."""
    top = set(lesr_high_weight_rule_texts)
    if len(top) == 0:
        return None
    kept = top & set(verified_rule_texts)
    return len(kept) / len(top)


def ev_weight_correlation(ev_scores, learned_weights):
    """
    Pearson r and Spearman rho between pre-grounding EV_i and the final,
    post-training significance weight w_i, across all verified rules.
    """
    ev = np.asarray(ev_scores, dtype=float)
    w = np.asarray(learned_weights, dtype=float)
    if len(ev) < 2 or np.std(ev) == 0 or np.std(w) == 0:
        return {"pearson": None, "spearman": None}
    pearson = float(np.corrcoef(ev, w)[0, 1])
    ev_rank = np.argsort(np.argsort(ev))
    w_rank = np.argsort(np.argsort(w))
    spearman = float(np.corrcoef(ev_rank, w_rank)[0, 1])
    return {"pearson": pearson, "spearman": spearman}


def grounding_flops_saved(n_candidates, n_verified, n_entities, sample_size=500, avg_degree=1.0):
    """Eq. (15)-(17), illustrative operation counts for reporting."""
    cost_lesr = n_candidates * (n_entities ** 2)
    cost_emvr = n_candidates * (sample_size * avg_degree) + n_verified * (n_entities ** 2)
    saved_pct = 1.0 - (cost_emvr / cost_lesr) if cost_lesr > 0 else 0.0
    return {"cost_lesr": cost_lesr, "cost_emvr": cost_emvr, "pct_saved": saved_pct}


# --------------------------------------------------------------------------
# Rule-quality metrics (Sec. VI-B / XIII-C of the base paper; matches He et
# al. 2026's definitions of HCR, RCS, RQI as used for the WD15K interpretability
# comparison, Table 8).
# --------------------------------------------------------------------------

def high_confidence_rule_ratio(learned_weights, threshold):
    """
    HCR = (# rules with post-training significance weight > threshold) / (# total learned rules).
    This is computable directly from a single trained ReasonerModel's weights,
    no external annotation needed. Returns a percentage (0-100), matching the
    base paper's Table 8 reporting convention.

    `learned_weights` should be the *logical-rule* weights only (i.e. excluding
    the trailing KGE-embedding slot) -- pass learned_weights[:-1] if you loaded
    the full softmax vector saved by lesr.py.
    threshold: theta_hi in the base paper. There is no single "correct" value;
    report whatever threshold you use alongside the number (0.5 is the
    illustrative value used in the EMVR-KBC proposal's Filtering Recall example).
    """
    w = np.asarray(learned_weights, dtype=float)
    if len(w) == 0:
        return 0.0
    n_high_conf = int(np.sum(w > threshold))
    return 100.0 * n_high_conf / len(w)


def rule_clarity_score(rule_texts, annotations):
    """
    RCS of a single rule = average interpretability score (0, 0.5, or 1) of its
    sampled real paths, as hand-labeled by human annotators (Lv et al. 2021).
    RCS of a model = average RCS across its high-confidence rules.

    This metric is NOT computable from your own run -- it requires the actual
    WD15K human interpretability annotation data (Lv et al., "Is multi-hop
    reasoning really explainable?", EMNLP 2021). It only applies to WD15K,
    since that's the only one of the five benchmark datasets with these
    annotations available.

    `annotations` must be a dict: {rule_text: [score, score, ...]} where each
    score is 0, 0.5, or 1, one entry per sampled path for that rule. If you
    don't have this file, pass annotations=None and this function returns
    None -- callers should report "N/A (requires WD15K human annotations)"
    rather than fabricating a number.
    """
    if annotations is None:
        return None
    per_rule_scores = []
    for rt in rule_texts:
        paths = annotations.get(rt)
        if paths:
            per_rule_scores.append(float(np.mean(paths)))
    if not per_rule_scores:
        return None
    return float(np.mean(per_rule_scores))


def rule_quality_index(hcr_pct, rcs):
    """
    RQI = harmonic mean of HCR (as a 0-100 percentage) and RCS (as a 0-1 score,
    rescaled to 0-100 for consistency with the base paper's Table 8 values,
    e.g. LeSR GPT-3.5: HCR=51.13, RCS=0.428, RQI=46.60).

    Returns None if RCS is unavailable (see rule_clarity_score) -- RQI cannot
    be honestly computed without it, and should be reported as N/A rather
    than silently substituting a proxy.
    """
    if rcs is None or hcr_pct is None or hcr_pct <= 0:
        return None
    rcs_pct = rcs * 100.0
    if rcs_pct <= 0:
        return None
    return 2 * hcr_pct * rcs_pct / (hcr_pct + rcs_pct)