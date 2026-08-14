# Write-up skeleton (YOU fill this in, in your own voice)

> Neel rejects LLM-written applications ("eerily similar... a significant negative signal").
> This is a SKELETON only: section order, what goes where, and bracketed prompts. Write the
> actual prose yourself. Numbers in [[double brackets]] come from results/summary.json after
> the Qwen3-4B run; paste and interpret them yourself.

---

## Executive summary (target ~1 page, max 3 / 600 words, include figures)

### What problem am I trying to solve? (and why it's interesting)
- [ ] One or two sentences: J-Lens claims to read a model's intermediate "working memory".
      Neel's review says it's useful for hypothesis *generation* but unproven for *validation*,
      and flags the multi-hop linear-relatedness confound. I test whether J-Lens is a causal
      auditing signal or that linear artifact.
- [ ] Why you personally find it interesting (one honest sentence, your voice).

### High-level takeaways (bullets)
- [ ] Headline: at low linsim, J-Lens [[does / does not]] beat logit lens (slope =
      [[C3_advantage_slope_vs_linsim]]).
- [ ] Detection: J-Lens MRR [[jlens]] vs logit [[logit]] at L*=[[Lstar]].
- [ ] The most surprising number you saw.
- [ ] The biggest limitation, stated plainly.

### Key experiment 1 - Detection (Fig 1)
- [ ] One paragraph: what it was, what you found, why it supports the takeaway.
- [ ] Insert figures/fig1_detection.png

### Key experiment 2 - Causal flip vs linearity (Fig 2, the headline)
- [ ] One paragraph. Insert figures/fig2_causal_vs_linsim.png

### Key experiment 3 - The confound test (Fig 3)
- [ ] One paragraph: the decisive slope. Insert figures/fig3_confound.png
- [ ] Controls (Fig 4): concept-swap vs answer-swap vs random.

---

## Randomly selected raw examples (put these right after the exec summary)
> Neel: "randomly selected, not cherry-picked". Pick ~5 steering records with a fixed rule
> (e.g. first 5 by id), paste the prompt, entity->cf_entity, answer/cf_answer, and the effect,
> and say in one line whether the flip looked real to your eye.
- [ ] example 1 ...
- [ ] example 2 ...

---

## Method (enough to reproduce without reading code)
- [ ] Model, prompt set + how linsim is measured, the 3 extractors, steering protocol, metric.
- [ ] What "J-Lens" means here (single-token Jacobian) and why that scoping is honest.

## What I verified myself (Neel weights this heavily)
- [ ] "I read N raw steering records and confirmed flips were real moves to A', not token-push."
- [ ] "I re-ran with seed=1; random control stayed ~0; signs stable."
- [ ] "I recomputed the headline slope by hand from steering.json."

## Limitations and next steps
- [ ] Single-token J-Lens only; candidate-set detection; same-position steering; small N.
- [ ] Natural extension: multi-token / future-token Jacobian; cross-position patching.

## Time log
- [ ] Toggl screenshot; rough breakdown across explore / understand / distill.
