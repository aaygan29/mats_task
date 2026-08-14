"""Detection (C0) and causal steering (C1/C2/C3).

Detection: on the last prompt token, how well does each lens rank the TRUE hidden
entity among the candidate set, across layers? -> picks the working layer L*.

Steering: at L*, inject (concept_to - concept_from) using each method's direction and
measure how much the FINAL answer flips (A -> A'). The headline covariate is linsim =
cos(U[answer], U[entity]): J-Lens is only interesting if it flips the answer even when
linsim is low (answer not linearly readable from entity).
"""
import math
import torch
from src.common import first_token_id, unembedding
from src.lenses import jlens_scores, logitlens_scores


@torch.no_grad()
def last_resid(model, enc, layers):
    hs = model(**enc, output_hidden_states=True).hidden_states
    return {l: hs[l][0, -1].detach().float() for l in layers}


def _rank_of(scores, target_idx):
    order = torch.argsort(scores, descending=True)
    return int((order == target_idx).nonzero().item()) + 1  # 1 = best


@torch.no_grad()
def run_detection(model, tok, items, cand_ids, id2idx, layers, device):
    """Per layer, per lens: mean reciprocal rank (MRR) of the true entity."""
    per_prompt = []
    for it in items:
        enc = tok(it["prompt"], return_tensors="pt").to(device)
        resid = last_resid(model, enc, layers)
        ent_idx = id2idx[first_token_id(tok, it["entity"])]
        # bake jvecs must be provided via closure; here we only do logit-lens which
        # needs no bake. J-Lens detection is filled in run_all (needs J tensor).
        rec = {"id": it["id"], "family": it["family"], "entity": it["entity"]}
        for l in layers:
            s = logitlens_scores(resid[l], model, cand_ids)
            rec[f"logit_L{l}"] = _rank_of(s, ent_idx)
        per_prompt.append((it, resid, ent_idx, rec))
    return per_prompt


def detection_add_jlens(per_prompt, J, layers, cand_ids):
    """Fill in J-Lens ranks using baked J [len(layers), n_cand, d]."""
    lidx = {l: i for i, l in enumerate(layers)}
    for it, resid, ent_idx, rec in per_prompt:
        for l in layers:
            s = jlens_scores(resid[l], J[lidx[l]])
            rec[f"jlens_L{l}"] = _rank_of(s, ent_idx)
    return per_prompt


def summarize_detection(per_prompt, layers):
    """MRR per (method, layer). Returns dict method -> {layer: mrr}."""
    out = {"jlens": {}, "logit": {}}
    n = len(per_prompt)
    for l in layers:
        for m in ("jlens", "logit"):
            key = f"{m}_L{l}"
            mrr = sum(1.0 / rec[key] for _, _, _, rec in per_prompt) / n
            out[m][l] = mrr
    return out


def add_hook(model, module_idx, pos, vec):
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        h[:, pos, :] += vec.to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    return model.model.layers[module_idx].register_forward_hook(hook)


@torch.no_grad()
def logprobs_at_last(model, enc, ids):
    lp = torch.log_softmax(model(**enc).logits[0, -1].float(), dim=-1)
    return {k: float(lp[v]) for k, v in ids.items()}


def method_vec(method, cid, id2idx, J_Lidx, U, P):
    if method == "jlens":
        return J_Lidx[id2idx[cid]]
    if method == "logit":
        return U[cid].float()
    if method == "probe":
        return P[id2idx[cid]]
    raise ValueError(method)


@torch.no_grad()
def run_steering(model, tok, items, id2idx, J, layers, Lstar, device,
                 methods=("jlens", "logit"), alphas=(2.0, 4.0, 8.0),
                 P=None, seed=0):
    """For each prompt/method/mode/alpha: causal flip effect on the answer.

    mode: 'concept' (entity->cf_entity, the real test), 'answer' (answer->cf_answer,
    Neel's fig-15 dominance control), 'random' (matched-norm random vector).
    Effect = [lp(A') - lp(A)]_steered - [lp(A') - lp(A)]_baseline.
    Also records whether steering merely pushed the cf_entity token (mediation control).
    """
    U = unembedding(model)
    lidx = {l: i for i, l in enumerate(layers)}
    J_Lstar = J[lidx[Lstar]]
    module_idx = Lstar - 1  # hidden_states[L] == output of decoder layer L-1
    g = torch.Generator(device="cpu").manual_seed(seed)
    records = []
    for it in items:
        enc = tok(it["prompt"], return_tensors="pt").to(device)
        pos = enc.input_ids.shape[1] - 1
        I = first_token_id(tok, it["entity"]); Ip = first_token_id(tok, it["cf_entity"])
        A = first_token_id(tok, it["answer"]); Ap = first_token_id(tok, it["cf_answer"])
        ids = {"A": A, "Ap": Ap, "I": I, "Ip": Ip}
        base = logprobs_at_last(model, enc, ids)
        base_gap = base["Ap"] - base["A"]
        linsim = float(torch.cosine_similarity(U[A].float(), U[I].float(), dim=0))
        resid_norm = float(model(**enc, output_hidden_states=True)
                           .hidden_states[Lstar][0, pos].float().norm())
        rand_unit = torch.nn.functional.normalize(
            torch.randn(model.config.hidden_size, generator=g), dim=0).to(device)

        def direction(method, frm, to):
            v = method_vec(method, to, id2idx, J_Lstar, U, P) \
                - method_vec(method, frm, id2idx, J_Lstar, U, P)
            return torch.nn.functional.normalize(v, dim=0)

        for alpha in alphas:
            scale = alpha * resid_norm
            trials = []
            for m in methods:
                trials.append((m, "concept", direction(m, I, Ip)))
                trials.append((m, "answer", direction(m, A, Ap)))
            trials.append(("random", "random", rand_unit))
            for m, mode, unit in trials:
                h = add_hook(model, module_idx, pos, scale * unit)
                try:
                    st = logprobs_at_last(model, enc, ids)
                finally:
                    h.remove()
                effect = (st["Ap"] - st["A"]) - base_gap
                records.append({
                    "id": it["id"], "family": it["family"], "linsim": linsim,
                    "method": m, "mode": mode, "alpha": alpha, "effect": effect,
                    "steered_gap": st["Ap"] - st["A"], "base_gap": base_gap,
                    "delta_Ap": st["Ap"] - base["Ap"], "delta_A": st["A"] - base["A"],
                    # mediation control: did we move the answer A' MORE than the raw
                    # entity token I'? If delta_Ap >> delta_Ip, it's real multi-hop.
                    "delta_Ip": st["Ip"] - base["Ip"],
                })
        print(f"[steer] {it['id']} done", flush=True)
    return records
