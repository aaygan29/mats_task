"""The three concept-direction extractors and their readers.

For a concept token c and layer L we produce a direction in residual-stream space:

  * logit-lens : U[c]            (the unembedding row; the cheap baseline)
  * J-Lens     : mean_pos d logit_c / d h_L      (single-token Jacobian, averaged)
  * diff-mean  : mean(h_L | about c) - mean(h_L)  (classic probe baseline)

J-Lens here is the *single-token* variant Neel names in his review ("even J-Lens
computed from single token Jacobians is better than logit lens in earlier layers").
It is the honest cheap-to-bake version; multi-token / future-token Jacobians are the
documented extension. The direction doubles as (a) a reader (dot with h_L to score a
concept) and (b) a steering vector (add to h_L to inject the concept).
"""
import torch
from src.common import unembedding, final_norm


def get_hidden_and_logits(model, enc):
    """Forward pass WITH grad. Returns (hidden_states tuple, logits[seq,V])."""
    out = model(**enc, output_hidden_states=True)
    return out.hidden_states, out.logits[0]


def bake_jvecs(model, tok, corpus_encs, cand_ids, layers, positions="last", skip=4):
    """Average single-token Jacobian d logit_c / d h_L over a small corpus.

    Returns J : tensor [len(layers), len(cand_ids), d_model] on the model device.
    corpus_encs: list of tokenizer encodings (already on device).
    """
    device = model.device
    d = model.config.hidden_size
    J = torch.zeros(len(layers), len(cand_ids), d, device=device, dtype=torch.float32)
    count = 0
    for pi, enc in enumerate(corpus_encs):
        hs, logits = get_hidden_and_logits(model, enc)
        seq = logits.shape[0]
        pos_list = [seq - 1] if positions == "last" else list(range(skip, seq))
        hs_sel = [hs[l] for l in layers]
        for pos in pos_list:
            lp = torch.log_softmax(logits[pos].float(), dim=-1)
            for ci, cid in enumerate(cand_ids):
                grads = torch.autograd.grad(lp[cid], hs_sel, retain_graph=True)
                for li, g in enumerate(grads):
                    J[li, ci] += g[0, pos].float()
            count += 1
        del hs, logits
        print(f"[bake] prompt {pi+1}/{len(corpus_encs)} done (pos={len(pos_list)})", flush=True)
    J /= max(count, 1)
    return J


def logit_vecs(model, cand_ids):
    """U[cand_ids] : [n_cand, d] -- logit-lens directions in residual space."""
    U = unembedding(model)
    return U[torch.tensor(cand_ids, device=U.device)].float().detach()


@torch.no_grad()
def probe_vecs(model, cand_ids, id2idx, entity_encs_by_cid, layer, global_mean):
    """Diff-of-means probe direction per candidate at `layer`.
    entity_encs_by_cid: {cand_id: [encs where that concept is the entity]}.
    Candidates with no positive examples get a zero vector (availability=False)."""
    d = model.config.hidden_size
    P = torch.zeros(len(id2idx), d, device=model.device, dtype=torch.float32)
    avail = [False] * len(id2idx)
    for cid, encs in entity_encs_by_cid.items():
        if cid not in id2idx or not encs:
            continue
        acc = torch.zeros(d, device=model.device, dtype=torch.float32)
        for enc in encs:
            hs = model(**enc, output_hidden_states=True).hidden_states
            acc += hs[layer][0, -1].float()
        P[id2idx[cid]] = acc / len(encs) - global_mean
        avail[id2idx[cid]] = True
    return P, avail


# ---- readers: score a candidate set from a residual vector ----

def jlens_scores(resid_L, J_layer):
    """resid_L [d], J_layer [n_cand, d] -> [n_cand] concept scores."""
    return (J_layer @ resid_L.float())


def logitlens_scores(resid_L, model, cand_ids):
    """Proper logit-lens read: final-norm then unembed, restricted to candidates."""
    fn = final_norm(model)(resid_L.unsqueeze(0)).squeeze(0).float()
    U = unembedding(model)
    return fn @ U[torch.tensor(cand_ids, device=U.device)].float().T


# ---- tuned lens (learned-linear baseline) ----
# The critical baseline: logit lens IGNORES layers L..final; J-Lens uses the true
# downstream Jacobian. A fair test needs a LEARNED-LINEAR map at L, fit on held-out
# data to predict the model's own final candidate logits from h_L. If J-Lens does not
# beat this, its detection win is just "downstream layers add information", not a
# property of the Jacobian. W[c] doubles as a tuned-lens steering direction.

@torch.no_grad()
def fit_tuned_lens(model, corpus_encs, cand_ids, layer, ridge=100.0, skip=4):
    """Ridge-fit W[n_cand,d], b[n_cand] : h_L -> model's true final candidate logits.
    Fit on the HELD-OUT corpus only (never the eval items)."""
    d = model.config.hidden_size
    cid = torch.tensor(cand_ids, device=model.device)
    Xs, Ys = [], []
    for enc in corpus_encs:
        out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states[layer][0]            # [seq, d]
        yl = out.logits[0][:, cid].float()          # [seq, n_cand] final logits at candidates
        Xs.append(hs[skip:].float()); Ys.append(yl[skip:])
    X = torch.cat(Xs); Y = torch.cat(Ys)            # [N,d], [N,n_cand]
    xm, ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - xm, Y - ym
    A = Xc.T @ Xc + ridge * torch.eye(d, device=X.device)
    W = torch.linalg.solve(A, Xc.T @ Yc).T          # [n_cand, d]
    b = ym - W @ xm
    return W.detach(), b.detach()


def tuned_scores(resid_L, W, b):
    return W @ resid_L.float() + b
