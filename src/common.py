"""Shared utilities: device/model loading, tokenization, forward passes, scoring.

The whole project asks one question: when we edit the model's internal guess of a
hidden intermediate entity, does the final answer change as J-Lens predicts, and does
J-Lens beat cheap baselines *even when the answer is not linearly readable from the
entity*? This module is the plumbing shared by detection and steering.
"""
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device):
    # bf16 on GPU; float32 elsewhere (MPS autograd is fragile in fp16).
    return torch.bfloat16 if device == "cuda" else torch.float32


def load_model(model_name, device=None, dtype=None):
    device = device or pick_device()
    dtype = dtype or pick_dtype(device)
    print(f"[common] loading {model_name} on {device} ({dtype}) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    # NOTE: we deliberately do NOT freeze params. We never optimise them, but leaving
    # requires_grad=True keeps the autograd graph alive so we can take gradients of a
    # logit w.r.t. intermediate hidden states (the J-Lens Jacobian).
    print(f"[common] loaded. n_layers={model.config.num_hidden_layers} "
          f"d_model={model.config.hidden_size} vocab={model.config.vocab_size}", flush=True)
    return model, tok, device, dtype


def load_prompts(path=None):
    path = path or os.path.join(REPO, "data", "prompts.json")
    with open(path) as f:
        data = json.load(f)
    return data["items"]


def first_token_id(tok, word, with_space=True):
    """First content token id of `word` (with a leading space, as it appears mid-text)."""
    s = (" " + word) if with_space else word
    ids = tok.encode(s, add_special_tokens=False)
    return ids[0]


def label_token_ids(tok, items):
    """Union of first-token ids for every entity/answer/cf_entity/cf_answer -> candidate set."""
    words = set()
    for it in items:
        for k in ("entity", "answer", "cf_entity", "cf_answer"):
            words.add(it[k])
    word2id = {w: first_token_id(tok, w) for w in sorted(words)}
    # warn loudly if two DISTINCT target words collide on their first token: detection
    # ranking and flip scoring cannot then tell them apart.
    byid = {}
    for w, tid in word2id.items():
        byid.setdefault(tid, []).append(w)
    collisions = {tid: ws for tid, ws in byid.items() if len(ws) > 1}
    if collisions:
        print(f"[common] WARNING first-token collisions among target labels: "
              f"{[ws for ws in collisions.values()]}", flush=True)
    ids = sorted(set(word2id.values()))
    id2idx = {tid: i for i, tid in enumerate(ids)}
    return word2id, ids, id2idx


@torch.no_grad()
def answers_zero_shot(model, tok, items, device, topk=5):
    """Verify each prompt: is the intended answer the argmax next token (or in top-k)?
    Returns list of dicts with the model's actual top tokens. Used to DROP bad items."""
    out = []
    for it in items:
        enc = tok(it["prompt"], return_tensors="pt").to(device)
        logits = model(**enc).logits[0, -1]
        top = torch.topk(logits, topk).indices.tolist()
        top_str = [tok.decode([t]).strip() for t in top]
        ans_id = first_token_id(tok, it["answer"])
        rank = (logits.argsort(descending=True) == ans_id).nonzero().item() + 1
        out.append({"id": it["id"], "answer": it["answer"], "top": top_str,
                    "answer_rank": int(rank), "answer_is_top1": top[0] == ans_id})
    return out


def prompt_last_pos(tok, prompt, device):
    enc = tok(prompt, return_tensors="pt").to(device)
    return enc, enc.input_ids.shape[1] - 1


def unembedding(model):
    """U : [vocab, d_model] (the lm_head weight)."""
    return model.get_output_embeddings().weight


def final_norm(model):
    """The model's final RMSNorm/LayerNorm module (applied before the unembedding)."""
    return model.model.norm
