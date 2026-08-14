"""End-to-end pipeline with checkpoints.

Stages (each checkpointed; re-run resumes):
  0. verify prompts zero-shot   -> results/verify.json   (drops bad items)
  1. bake J-Lens jvecs          -> checkpoints/jvecs.pt
  2. detection sweep (C0)       -> results/detection.json (picks L*)
  3. causal steering (C1/C2/C3) -> results/steering.json
  4. figures + summary          -> figures/*.png, results/summary.json

Usage:  python -m src.run_all --config configs/smoke.yaml [--force] [--out DIR]
"""
import argparse
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

    # ---- stage 0: verify ----
    vpath = os.path.join(rdir, "verify.json")
    verify = common.answers_zero_shot(model, tok, items_all, device)
    json.dump(verify, open(vpath, "w"), indent=2)
    keep_rank = cfg.get("keep_rank", 10 ** 9)
    keep_ids = {v["id"] for v in verify if v["answer_rank"] <= keep_rank}
    items = [it for it in items_all if it["id"] in keep_ids]
    print(f"[run] kept {len(items)}/{len(items_all)} items (answer_rank<={keep_rank})", flush=True)

    word2id, cand_ids, id2idx = common.label_token_ids(tok, items)
    layers = cfg.get("layers") or auto_layers(model.config.num_hidden_layers,
                                              cfg.get("n_layers_probe", 8))

    # ---- stage 1: bake jvecs ----
    jpath = os.path.join(cdir, "jvecs.pt")
    if os.path.exists(jpath) and not force:
        print("[run] loading cached jvecs", flush=True)
        blob = torch.load(jpath, map_location=device)
        J, cand_ids, layers = blob["J"], blob["cand_ids"], blob["layers"]
        id2idx = {tid: i for i, tid in enumerate(cand_ids)}
    else:
        corpus = [tok(it["prompt"], return_tensors="pt").to(device)
                  for it in items[:cfg.get("bake_max_prompts", len(items))]]
        J = lenses.bake_jvecs(model, tok, corpus, cand_ids, layers,
                              positions=cfg.get("bake_positions", "last"))
        torch.save({"J": J.cpu(), "cand_ids": cand_ids, "layers": layers}, jpath)
        J = J.to(device)

    # ---- stage 2: detection ----
    dpath = os.path.join(rdir, "detection.json")
    per_prompt = experiment.run_detection(model, tok, items, cand_ids, id2idx, layers, device)
    experiment.detection_add_jlens(per_prompt, J, layers, cand_ids)
    summ = experiment.summarize_detection(per_prompt, layers)
    cand_layers = [l for l in layers if l >= 1]
    Lstar = max(cand_layers, key=lambda l: summ["jlens"][l])
    det_out = {"summary": {m: {str(l): summ[m][l] for l in layers} for m in summ},
               "Lstar": Lstar,
               "per_prompt": [rec for _, _, _, rec in per_prompt]}
    json.dump(det_out, open(dpath, "w"), indent=2)
    print(f"[run] L* = {Lstar}  jlens_MRR={summ['jlens'][Lstar]:.3f} "
          f"logit_MRR={summ['logit'][Lstar]:.3f}", flush=True)

    # ---- optional probe baseline at L* ----
    P = None
    methods = list(cfg.get("methods", ["jlens", "logit"]))
    if "probe" in methods:
        with torch.no_grad():
            resids = [model(**tok(it["prompt"], return_tensors="pt").to(device),
                            output_hidden_states=True).hidden_states[Lstar][0, -1].float()
                      for it in items]
            gmean = torch.stack(resids).mean(0)
        enc_by_cid = {}
        for it in items:
            cid = common.first_token_id(tok, it["entity"])
            enc_by_cid.setdefault(cid, []).append(tok(it["prompt"], return_tensors="pt").to(device))
        P, _ = lenses.probe_vecs(model, cand_ids, id2idx, enc_by_cid, Lstar, gmean)

    # ---- stage 3: steering ----
    spath = os.path.join(rdir, "steering.json")
    records = experiment.run_steering(
        model, tok, items, id2idx, J, layers, Lstar, device,
        methods=tuple(m for m in methods if m != "probe") + (("probe",) if P is not None else ()),
        alphas=tuple(cfg.get("alphas", [2.0, 4.0, 8.0])), P=P, seed=cfg.get("seed", 0))
    json.dump(records, open(spath, "w"), indent=2)

    # ---- stage 4: figures + summary ----
    plots.fig1_detection(det_out["summary"], layers, Lstar,
                         os.path.join(fdir, "fig1_detection.png"), tag)
    plots.fig2_causal_vs_linsim(records, os.path.join(fdir, "fig2_causal_vs_linsim.png"), tag)
    plots.fig3_confound(records, os.path.join(fdir, "fig3_confound.png"), tag)
    plots.fig4_controls(records, os.path.join(fdir, "fig4_controls.png"), tag)

    summary = build_summary(cfg, summ, Lstar, records, items, layers)
    json.dump(summary, open(os.path.join(rdir, "summary.json"), "w"), indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary["headline"], indent=2))
    return summary


def build_summary(cfg, summ, Lstar, records, items, layers):
    a = plots._concept_best_alpha(records)
    def mean(m, mode, sel=lambda r: True):
        e = [r["effect"] for r in records if r["method"] == m and r["mode"] == mode
             and r["alpha"] == a and sel(r)]
        return float(np.mean(e)) if e else None
    lin = sorted(r["linsim"] for r in records if r["mode"] == "concept" and r["method"] == "jlens")
    med = lin[len(lin)//2] if lin else 0.0
    # C3 slope: (jlens-logit) effect vs linsim
    by_id = {}
    for r in records:
        if r["mode"] == "concept" and r["alpha"] == a:
            by_id.setdefault(r["id"], {})[r["method"]] = (r["linsim"], r["effect"])
    xs = [d["jlens"][0] for d in by_id.values() if "jlens" in d and "logit" in d]
    ys = [d["jlens"][1] - d["logit"][1] for d in by_id.values() if "jlens" in d and "logit" in d]
    slope = float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 2 else None
    headline = {
        "model": cfg["model_name"], "Lstar": Lstar, "best_alpha": a,
        "n_items": len(items),
        "C0_detection_MRR_at_Lstar": {"jlens": summ["jlens"][Lstar], "logit": summ["logit"][Lstar]},
        "C1_jlens_concept_effect_mean": mean("jlens", "concept"),
        "C1_random_control_mean": mean("random", "random"),
        "C2_jlens_vs_logit_concept": {"jlens": mean("jlens", "concept"), "logit": mean("logit", "concept")},
        "C3_lowlinsim": {
            "jlens": mean("jlens", "concept", lambda r: r["linsim"] <= med),
            "logit": mean("logit", "concept", lambda r: r["linsim"] <= med)},
        "C3_advantage_slope_vs_linsim": slope,
        "dominance_control_answer_swap_jlens": mean("jlens", "answer"),
    }
    return {"headline": headline, "best_alpha": a, "linsim_median": med}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    main(args.config, out_dir=args.out, force=args.force)
