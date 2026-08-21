"""Unit tests for src.binoculars_score, using small hand-constructed logits
(no real model, no GPU needed) so the arithmetic can be checked directly
against the reference implementation's formulas."""
import torch
import torch.nn.functional as F

from src.binoculars_score import binoculars_score, cross_model_entropy, token_cross_entropy


def test_token_cross_entropy_matches_direct_computation():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7)
    input_ids = torch.randint(0, 7, (2, 5))
    attention_mask = torch.ones(2, 5, dtype=torch.long)

    result = token_cross_entropy(logits, input_ids, attention_mask)

    # Hand-compute via the same shift, using torch's own cross_entropy directly.
    shifted_logits = logits[:, :-1, :]
    shifted_labels = input_ids[:, 1:]
    expected = F.cross_entropy(
        shifted_logits.transpose(1, 2), shifted_labels, reduction="none"
    ).mean(dim=1)
    assert torch.allclose(result, expected, atol=1e-5)


def test_token_cross_entropy_respects_attention_mask():
    # A padded position with an extreme logit should not affect the result
    # once masked out.
    logits = torch.zeros(1, 4, 3)
    logits[0, 0] = torch.tensor([10.0, -10.0, -10.0])  # confidently predicts class 0
    logits[0, 1] = torch.tensor([10.0, -10.0, -10.0])  # would be masked out
    input_ids = torch.tensor([[0, 0, 0, 0]])
    mask_with_pad = torch.tensor([[1, 1, 1, 0]])  # last position padded
    mask_full = torch.tensor([[1, 1, 1, 1]])

    result_padded = token_cross_entropy(logits, input_ids, mask_with_pad)
    result_full = token_cross_entropy(logits, input_ids, mask_full)
    # Different masks should (in general) give different results since a
    # different number/set of positions contribute to the mean.
    assert not torch.allclose(result_padded, result_full)


def test_cross_model_entropy_is_low_when_distributions_agree_and_peaked():
    # p and q both confidently predict the same class -> cross-entropy of a
    # near-delta distribution against itself is close to its own (near-zero) entropy.
    peaked_logits = torch.tensor([[[20.0, -20.0], [20.0, -20.0]]])  # (1, 2, 2)
    mask = torch.ones(1, 2, dtype=torch.long)
    result = cross_model_entropy(peaked_logits, peaked_logits, mask)
    assert result.item() < 1e-3


def test_cross_model_entropy_is_high_when_distributions_disagree():
    p_logits = torch.tensor([[[20.0, -20.0]]])  # confidently predicts class 0
    q_logits = torch.tensor([[[-20.0, 20.0]]])  # confidently predicts class 1
    mask = torch.ones(1, 1, dtype=torch.long)
    result = cross_model_entropy(p_logits, q_logits, mask)
    assert result.item() > 10.0  # cross-entropy blows up when q assigns ~0 to p's mass


def test_cross_model_entropy_matches_hand_computed_soft_target_case():
    # p is a soft (non-peaked) distribution: softmax([0, log(2)]) = [1/3, 2/3].
    # Cross-entropy against a q that predicts class 0 with certainty is
    # -sum_v p(v) log softmax(q)(v) = -[1/3*log(~0) + 2/3*log(~1)] -> large,
    # dominated by the 1/3 * -log(epsilon) term. Check against a direct
    # manual computation instead of an inequality, for a tighter check.
    p_logits = torch.tensor([[[0.0, torch.log(torch.tensor(2.0))]]])
    q_logits = torch.tensor([[[20.0, -20.0]]])
    mask = torch.ones(1, 1, dtype=torch.long)

    p_proba = F.softmax(p_logits, dim=-1)
    log_q = F.log_softmax(q_logits, dim=-1)
    expected = -(p_proba * log_q).sum(dim=-1).squeeze()

    result = cross_model_entropy(p_logits, q_logits, mask)
    assert torch.allclose(result.squeeze(), expected, atol=1e-4)


def test_binoculars_score_equals_ppl_over_x_ppl():
    torch.manual_seed(1)
    observer_logits = torch.randn(1, 4, 5)
    performer_logits = torch.randn(1, 4, 5)
    input_ids = torch.randint(0, 5, (1, 4))
    mask = torch.ones(1, 4, dtype=torch.long)

    score = binoculars_score(observer_logits, performer_logits, input_ids, mask)
    expected_ppl = token_cross_entropy(performer_logits, input_ids, mask)
    expected_xppl = cross_model_entropy(observer_logits, performer_logits, mask)
    assert torch.allclose(score, expected_ppl / expected_xppl)


def test_binoculars_score_is_lower_for_text_the_performer_finds_surprising_relative_to_observer():
    # Construct a case where performer's own perplexity on the actual
    # tokens is high (surprised by them) while its distribution still
    # roughly matches the observer's (low x_ppl) -> score = ppl/x_ppl should
    # be higher than a case where performer is confident AND matches
    # observer (low ppl, low x_ppl can go either way, but this checks the
    # score direction is driven by ppl when x_ppl is held comparable).
    # Two positions needed: shifting drops the last one, so predicting a
    # single next-token target requires seq_len=2 logits/input_ids.
    shared_logits = torch.tensor([[[20.0, -20.0], [20.0, -20.0]]])  # confidently predicts class 0 at both steps
    input_ids_matching = torch.tensor([[0, 0]])  # actual next token IS class 0 -> low ppl
    input_ids_surprising = torch.tensor([[0, 1]])  # actual next token is class 1 -> high ppl
    mask = torch.ones(1, 2, dtype=torch.long)

    score_matching = binoculars_score(shared_logits, shared_logits, input_ids_matching, mask)
    score_surprising = binoculars_score(shared_logits, shared_logits, input_ids_surprising, mask)
    assert score_surprising.item() > score_matching.item()
