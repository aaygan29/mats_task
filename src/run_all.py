"""End-to-end pipeline with checkpoints, three lenses, held-out fitting, and stats.

Stages: verify -> bake J (held-out corpus) + fit tuned lens (held-out) ->
detection (C0, L* chosen by BASELINE lenses only) -> steering (C1/C2/C3) ->
sanity gate + bootstrap stats -> figures + summary.

Usage:  python -m src.run_all --config configs/main.yaml [--force] [--out DIR]
"""
import argparse
import hashlib
import json
import os
import sys
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import common, lenses, experiment, plots


def auto_layers(n_layers, k=8):
    return sorted(set(int(round(x)) for x in np.linspace(1, n_layers - 1, k)))


def bootstrap_ci(vals, nboot=3000, seed=0):
    vals = np.asarray(vals, float)
    if len(vals) < 2:
        return [None, None]
    rng = np.random.default_rng(seed)
    means = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(nboot)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def paired_perm_p(diffs, nboot=10000, seed=0):
    """Two-sided sign-flip permutation test that mean(diffs) != 0."""
    diffs = np.asarray(diffs, float)
    if len(diffs) < 2:
        return None
    rng = np.random.default_rng(seed)
    obs = abs(diffs.mean())
    cnt = sum(abs((diffs * rng.choice([-1, 1], len(diffs))).mean()) >= obs for _ in range(nboot))
    return float((cnt + 1) / (nboot + 1))


def slope_ci(xs, ys, nboot=3000, seed=0):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(xs) < 3:
        return None, [None, None]
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(nboot):
        idx = rng.choice(len(xs), len(xs), replace=True)
        if len(np.unique(xs[idx])) < 2:
            continue
        slopes.append(np.polyfit(xs[idx], ys[idx], 1)[0])
    return float(np.polyfit(xs, ys, 1)[0]), [float(np.percentile(slopes, 2.5)),
                                             float(np.percentile(slopes, 97.5))]


def sanity_gate(records, chosen_alpha):
    """Cheap structural checks on the metric's own controls. Catches the run-1 class of
    bugs (non-zero-centered metric, collapsed coherence) BEFORE trusting anything."""
    coh = [r for r in records if r.get("coherent")]
    rnd = [r["toward_Ap"] for r in coh if r["method"] == "random" and r["alpha"] == chosen_alpha]
    con = [r for r in coh if r["mode"] == "concept" and r["alpha"] == chosen_alpha]
    checks = {
        "random_control_mean_near_zero": bool(abs(np.mean(rnd)) < 0.05) if rnd else False,
        "has_coherent_concept_trials": bool(len(con) > 0),
        "random_control_mean": float(np.mean(rnd)) if rnd else None,
        "n_coherent_concept": int(len(con)),
    }
    checks["PASS"] = bool(checks["random_control_mean_near_zero"]
                          and checks["has_coherent_concept_trials"])
    print(f"[sanity] {checks}", flush=True)
    return checks


def main(config_path, out_dir=None, force=False):
    cfg = yaml.safe_load(open(config_path))
    repo = common.REPO
    out_dir = out_dir or repo
    rdir = os.path.join(out_dir, "results"); fdir = os.path.join(out_dir, "figures")
    cdir = os.path.join(out_dir, "checkpoints")
    for d in (rdir, fdir, cdir):
        os.makedirs(d, exist_ok=True)
    tag = f" [{cfg['model_name'].split('/')[-1]}]"

    model, tok, device, dtype = common.load_model(cfg["model_name"])
    items_all = common.load_prompts(cfg.get("prompts_path"))

    # ---- verify ----
    verify = common.answers_zero_shot(model, tok, items_all, device)
    json.dump(verify, open(os.path.join(rdir, "verify.json"), "w"), indent=2)
    keep_rank = cfg.get("keep_rank", 10 ** 9)
    keep_ids = {v["id"] for v in verify if v["answer_rank"] <= keep_rank}
    items = [it for it in items_all if it["id"] in keep_ids]
    print(f"[run] kept {len(items)}/{len(items_all)} items (answer_rank<={keep_rank})", flush=True)

    word2id, cand_ids, id2idx = common.label_token_ids(tok, items)
    layers = cfg.get("layers") or auto_layers(model.config.num_hidden_layers,
                                              cfg.get("n_layers_probe", 8))

    # ---- held-out corpus for baking J and fitting tuned lens ----
    corpus_path = os.path.join(repo, "data", "bake_corpus.json")
    corpus_txt = json.load(open(corpus_path))["sentences"]
    corpus = [tok(s, return_tensors="pt").to(device)
              for s in corpus_txt[:cfg.get("bake_max_prompts", len(corpus_txt))]]

    cfg_hash = hashlib.md5(json.dumps({"m": cfg["model_name"], "cand": cand_ids,
        "layers": layers, "corpus": len(corpus)}, sort_keys=True).encode()).hexdigest()[:8]
    jpath = os.path.join(cdir, f"jvecs_{cfg_hash}.pt")
    if os.path.exists(jpath) and not force:
        print("[run] loading cached jvecs", flush=True)
        J = torch.load(jpath, map_location=device)["J"].to(device)
    else:
        J = lenses.bake_jvecs(model, tok, corpus, cand_ids, layers,
                              positions=cfg.get("bake_positions", "last"))
        torch.save({"J": J.cpu(), "hash": cfg_hash}, jpath)

    # tuned lens per layer (fit on the SAME held-out corpus)
    print("[run] fitting tuned lens per layer (held-out)...", flush=True)
    tuned = {l: lenses.fit_tuned_lens(model, corpus, cand_ids, l,
                                      ridge=cfg.get("tuned_ridge", 100.0)) for l in layers}

    # ---- detection ----
    per_prompt = experiment.run_detection(model, tok, items, cand_ids, id2idx, layers,
                                          device, J, tuned)
    summ = experiment.summarize_detection(per_prompt, layers)
    # L* chosen by the BASELINE lenses only (not J-Lens) -> no forking-path toward J-Lens
    cand_layers = [l for l in layers if l >= 1]
    Lstar = max(cand_layers, key=lambda l: 0.5 * (summ["logit"][l] + summ["tuned"][l]))
    det_out = {"summary": {m: {str(l): summ[m][l] for l in layers} for m in experiment.METHODS},
               "Lstar": Lstar, "Lstar_rule": "argmax mean(logit,tuned) MRR",
               "per_prompt": [rec for _, _, _, rec in per_prompt]}
    json.dump(det_out, open(os.path.join(rdir, "detection.json"), "w"), indent=2)
    print(f"[run] L*={Lstar}  MRR jlens={summ['jlens'][Lstar]:.3f} "
          f"logit={summ['logit'][Lstar]:.3f} tuned={summ['tuned'][Lstar]:.3f}", flush=True)

    # ---- steering ----
    W_tuned = tuned[Lstar][0]
    records = experiment.run_steering(
        model, tok, items, id2idx, J, layers, Lstar, device, W_tuned,
        methods=tuple(cfg.get("methods", ["jlens", "logit", "tuned"])),
        alphas=tuple(cfg.get("alphas", [0.5, 1.0, 2.0, 4.0])))
    json.dump(records, open(os.path.join(rdir, "steering.json"), "w"), indent=2)

    chosen_alpha = min(r["alpha"] for r in records)  # gentlest perturbation, a-priori rule
    sanity = sanity_gate(records, chosen_alpha)

    # ---- figures ----
    plots.fig1_detection(det_out["summary"], layers, Lstar,
                         os.path.join(fdir, "fig1_detection.png"), tag)
    plots.fig2_causal_vs_linsim(records, os.path.join(fdir, "fig2_causal_vs_linsim.png"), tag, chosen_alpha)
    plots.fig3_confound(records, os.path.join(fdir, "fig3_confound.png"), tag, chosen_alpha)
    plots.fig4_controls(records, os.path.join(fdir, "fig4_controls.png"), tag, chosen_alpha)
    plots.fig5_alpha_curve(records, os.path.join(fdir, "fig5_alpha_curve.png"), tag)

    summary = build_summary(cfg, summ, Lstar, records, items, chosen_alpha, sanity)
    json.dump(summary, open(os.path.join(rdir, "summary.json"), "w"), indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary["headline"], indent=2))
    return summary


def _paired(records, alpha, m1, m2, field="toward_Ap"):
    """Per-item paired values for m1 vs m2 on the intersection of coherent concept trials."""
    d = {}
    for r in records:
        if r["mode"] == "concept" and r["alpha"] == alpha and r.get("coherent") \
                and r["method"] in (m1, m2):
            d.setdefault(r["id"], {})[r["method"]] = (r["linsim"], r[field])
    xs, a, b = [], [], []
    for v in d.values():
        if m1 in v and m2 in v:
            xs.append(v[m1][0]); a.append(v[m1][1]); b.append(v[m2][1])
    return xs, a, b


def build_summary(cfg, summ, Lstar, records, items, alpha, sanity):
    coh = [r for r in records if r.get("coherent")]
    def mean(m, mode, sel=lambda r: True):
        e = [r["toward_Ap"] for r in coh if r["method"] == m and r["mode"] == mode
             and r["alpha"] == alpha and sel(r)]
        return float(np.mean(e)) if e else None

    xs_jl, jl, lo = _paired(records, alpha, "jlens", "logit")
    _, jl2, tu = _paired(records, alpha, "jlens", "tuned")
    diff_jl_lo = list(np.array(jl) - np.array(lo)) if jl else []
    diff_jl_tu = list(np.array(jl2) - np.array(tu)) if jl2 else []
    slope, slope_ci_ = slope_ci(xs_jl, diff_jl_lo) if diff_jl_lo else (None, [None, None])

    headline = {
        "model": cfg["model_name"], "Lstar": Lstar, "chosen_alpha": alpha,
        "n_items_kept": len(items), "n_paired_jlens_logit": len(jl),
        "metric": "toward_Ap = P(counterfactual answer) moved (coherent trials)",
        "sanity_gate": sanity,
        "C0_detection_MRR_at_Lstar": {m: summ[m][Lstar] for m in experiment.METHODS},
        "C1_jlens_concept_mean": mean("jlens", "concept"),
        "C1_jlens_concept_CI": bootstrap_ci([r["toward_Ap"] for r in coh
            if r["method"] == "jlens" and r["mode"] == "concept" and r["alpha"] == alpha]),
        "C1_random_control_mean": mean("random", "random"),
        "C2_means": {"jlens": float(np.mean(jl)) if jl else None,
                     "logit": float(np.mean(lo)) if lo else None,
                     "tuned": float(np.mean(tu)) if tu else None},
        "C2_jlens_minus_logit_CI": bootstrap_ci(diff_jl_lo),
        "C2_jlens_minus_logit_paired_p": paired_perm_p(diff_jl_lo),
        "C2_jlens_minus_tuned_paired_p": paired_perm_p(diff_jl_tu),
        "C3_advantage_slope": slope, "C3_slope_CI": slope_ci_,
        "dominance_answer_swap_jlens": mean("jlens", "answer"),
        "token_push_jlens_concept": mean("jlens", "concept",
            lambda r: True) and float(np.mean([r["push_Ip"] for r in coh
            if r["method"] == "jlens" and r["mode"] == "concept" and r["alpha"] == alpha])),
    }
    return {"headline": headline, "chosen_alpha": alpha}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    main(args.config, out_dir=args.out, force=args.force)
