"""Binoculars score (Hans et al., 2024, ICML) -- the Tier-2 model-based
detector RESEARCH_PLAN.md pre-registers for Experiment 1: "Whether it
actually does that for ELL writers is experiment 1."

Reimplemented directly from the reference implementation
(github.com/ahans30/Binoculars, `binoculars/metrics.py` and
`binoculars/detector.py`, fetched and read in full rather than trusted from
memory, since a sign or log/exp error here would be silent and easy to
miss) rather than from a paraphrase of the paper. Two functions map
directly onto that source:

- `token_cross_entropy` is `binoculars/metrics.py::perplexity` (the name
  in the original is misleading -- it returns the mean per-token
  cross-entropy in nats, not an exponentiated perplexity).
- `cross_model_entropy` is `binoculars/metrics.py::entropy`.

`binoculars_score` matches `binoculars/detector.py::Binoculars.compute_score`
exactly: `ppl = token_cross_entropy(performer_logits, ...)`,
`x_ppl = cross_model_entropy(p_logits=observer_logits, q_logits=performer_logits, ...)`,
`score = ppl / x_ppl`. Lower score = more likely AI-generated (matches the
reference implementation's convention).

Model substitution: the original paper uses Falcon-7B (observer, base) /
Falcon-7B-Instruct (performer, instruction-tuned). This study substitutes
a same-tokenizer-family base/instruct pair small enough for the 6GB-VRAM
GPU available (`RESEARCH_PLAN.md` explicitly permits this: "Substitute
smaller models if VRAM is tight and record which") -- see
scripts/experiment1b_tier2_ell_gate.py for which pair and why.
"""
import torch

_ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
_softmax_fn = torch.nn.Softmax(dim=-1)


def token_cross_entropy(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean per-token cross-entropy (nats) of the actual next tokens under
    `logits`, masked to real (non-padding) positions. Equivalent to
    `binoculars/metrics.py::perplexity` (non-median branch)."""
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = input_ids[..., 1:].contiguous()
    shifted_mask = attention_mask[..., 1:].contiguous()
    ce = _ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels)
    return (ce * shifted_mask).sum(1) / shifted_mask.sum(1)


def cross_model_entropy(p_logits: torch.Tensor, q_logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy between softmax(p_logits) (the reference distribution)
    and q_logits (what's being evaluated against it), averaged over real
    token positions. Equivalent to `binoculars/metrics.py::entropy`
    (non-median, non-sampled branch), but takes an explicit attention mask
    rather than re-deriving one from a pad token id, since this project's
    tokenizer call already produces one."""
    batch, seq_len, vocab_size = q_logits.shape
    p_proba = _softmax_fn(p_logits).reshape(-1, vocab_size)
    q_scores = q_logits.reshape(-1, vocab_size)
    ce = _ce_loss_fn(input=q_scores, target=p_proba).view(batch, seq_len)
    mask = attention_mask.to(ce.dtype)
    return (ce * mask).sum(1) / mask.sum(1)


def binoculars_score(
    observer_logits: torch.Tensor, performer_logits: torch.Tensor,
    input_ids: torch.Tensor, attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Lower score = more likely AI-generated (matches the reference
    implementation's convention: `predict()` flags scores below a
    threshold as AI-generated)."""
    ppl = token_cross_entropy(performer_logits, input_ids, attention_mask)
    x_ppl = cross_model_entropy(observer_logits, performer_logits, attention_mask)
    return ppl / x_ppl
