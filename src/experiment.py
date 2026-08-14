"""Detection (C0) and causal steering (C1/C2/C3), across three lenses:
J-Lens (single-token Jacobian), logit lens (network-blind), tuned lens (learned-linear,
network-aware). Tuned lens is the baseline that makes C0 a fair test.
"""
import math
import torch
from src.common import first_token_id, unembedding
from src.lenses import jlens_scores, logitlens_scores, tuned_scores

METHODS = ("jlens", "logit", "tuned")


@torch.no_grad()
def last_resid(model, enc, layers):
    hs = model(**enc, output_hidden_states=True).hidden_states
    return {l: hs[l][0, -1].detach().float() for l in layers}


def _rank_of(scores, target_idx):
    order = torch.argsort(scores, descending=True)
    return int((order == target_idx).nonzero().item()) + 1  # 1 = best


@torch.no_grad()
def run_detection(model, tok, items, cand_ids, id2idx, layers, device, J, tuned):
    """Per item/layer/lens: rank of the true entity among candidates.
    J: [len(layers), n_cand, d]; tuned: {layer: (W,b)}."""
    lidx = {l: i for i, l in enumerate(layers)}
    per_prompt = []
    for it in items:
        enc = tok(it["prompt"], return_tensors="pt").to(device)
        resid = last_resid(model, enc, layers)
        ent_idx = id2idx[first_token_id(tok, it["entity"])]
        rec = {"id": it["id"], "family": it["family"], "entity": it["entity"]}
        for l in layers:
            rec[f"jlens_L{l}"] = _rank_of(jlens_scores(resid[l], J[lidx[l]]), ent_idx)
            rec[f"logit_L{l}"] = _rank_of(logitlens_scores(resid[l], model, cand_ids), ent_idx)
            W, b = tuned[l]
            rec[f"tuned_L{l}"] = _rank_of(tuned_scores(resid[l], W, b), ent_idx)
        per_prompt.append((it, resid, ent_idx, rec))
    return per_prompt


def summarize_detection(per_prompt, layers):
    out = {m: {} for m in METHODS}
    n = len(per_prompt)
    for l in layers:
        for m in METHODS:
            out[m][l] = sum(1.0 / rec[f"{m}_L{l}"] for _, _, _, rec in per_prompt) / n
    return out


def add_hook(model, module_idx, pos, vec):
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        h[:, pos, :] += vec.to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    return model.model.layers[module_idx].register_forward_hook(hook)


@torch.no_grad()
def scored_last(model, enc, ids):
    lp = torch.log_softmax(model(**enc).logits[0, -1].float(), dim=-1)
    p = lp.exp()
    ent = float(-(p * lp).sum())
    return {k: float(lp[v]) for k, v in ids.items()}, {
        "entropy": ent, "top_id": int(lp.argmax()), "max_p": float(p.max())}


def method_vec(method, cid, id2idx, J_Lstar, U, W_tuned):
    if method == "jlens":
        return J_Lstar[id2idx[cid]]
    if method == "logit":
        return U[cid].float()
    if method == "tuned":
        return W_tuned[id2idx[cid]]
    raise ValueError(method)


@torch.no_grad()
def run_steering(model, tok, items, id2idx, J, layers, Lstar, device, W_tuned,
                 methods=("jlens", "logit", "tuned"), alphas=(0.5, 1.0, 2.0, 4.0),
                 rand_seeds=(0, 1, 2, 3, 4)):
    """Causal flip effect (toward_Ap = prob mass moved onto A'), with controls.
    random control is averaged over several seeds for a variance estimate.
    Guards against degenerate (zero-norm) steering directions -> NaN."""
    U = unembedding(model)
    lidx = {l: i for i, l in enumerate(layers)}
    J_Lstar = J[lidx[Lstar]]
    module_idx = Lstar - 1
    records = []
    for it in items:
        enc = tok(it["prompt"], return_tensors="pt").to(device)
        pos = enc.input_ids.shape[1] - 1
        I = first_token_id(tok, it["entity"]); Ip = first_token_id(tok, it["cf_entity"])
        A = first_token_id(tok, it["answer"]); Ap = first_token_id(tok, it["cf_answer"])
        # skip items whose concept/answer collide on the first token (would NaN or be meaningless)
        if I == Ip or A == Ap:
            print(f"[steer] SKIP {it['id']} (first-token collision I/Ip or A/Ap)", flush=True)
            continue
        ids = {"A": A, "Ap": Ap, "I": I, "Ip": Ip}
        base, base_coh = scored_last(model, enc, ids)
        linsim = float(torch.cosine_similarity(U[A].float(), U[I].float(), dim=0))
        resid_norm = float(model(**enc, output_hidden_states=True)
                           .hidden_states[Lstar][0, pos].float().norm())

        def direction(method, frm, to):
            v = method_vec(method, to, id2idx, J_Lstar, U, W_tuned) \
                - method_vec(method, frm, id2idx, J_Lstar, U, W_tuned)
            n = v.norm()
            return None if float(n) < 1e-8 else v / n

        def measure(unit, scale):
            h = add_hook(model, module_idx, pos, scale * unit)
            try:
                st, coh = scored_last(model, enc, ids)
            finally:
                h.remove()
            coherent = coh["entropy"] >= 0.3 * base_coh["entropy"] and coh["max_p"] <= 0.999
            return {
                "toward_Ap": math.exp(st["Ap"]) - math.exp(base["Ap"]),
                "push_Ip": math.exp(st["Ip"]) - math.exp(base["Ip"]),
                "flip": bool(st["Ap"] > st["A"]),
                "p_Ap_base": math.exp(base["Ap"]), "p_Ap_steer": math.exp(st["Ap"]),
                "p_A_base": math.exp(base["A"]), "p_A_steer": math.exp(st["A"]),
                "coherent": bool(coherent), "entropy_steer": coh["entropy"],
                "entropy_base": base_coh["entropy"]}

        for alpha in alphas:
            scale = alpha * resid_norm
            common = {"id": it["id"], "family": it["family"], "linsim": linsim, "alpha": alpha}
            for m in methods:
                for mode, frm, to in (("concept", I, Ip), ("answer", A, Ap)):
                    u = direction(m, frm, to)
                    if u is None:
                        continue
                    records.append({**common, "method": m, "mode": mode, **measure(u, scale)})
            # random control, averaged over seeds (matched norm)
            for s in rand_seeds:
                g = torch.Generator(device="cpu").manual_seed(1000 * s + int(alpha * 10))
                u = torch.nn.functional.normalize(
                    torch.randn(model.config.hidden_size, generator=g), dim=0).to(device)
                records.append({**common, "method": "random", "mode": "random",
                                "seed": s, **measure(u, scale)})
        print(f"[steer] {it['id']} done", flush=True)
    return records
