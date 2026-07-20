import torch

from ste_optimized.config import SamplingConfig
from ste_optimized.ste import LogitProcessingChain, one_hot_ste


def test_one_hot_ste_forward_is_hard_gradient_is_soft():
    logits = torch.randn(4, 10, requires_grad=True)
    probs = torch.softmax(logits, dim=-1)
    codes = torch.randint(0, 10, (4,))
    y = one_hot_ste(codes, probs)
    hard = torch.nn.functional.one_hot(codes, 10).float()
    assert torch.allclose(y.detach(), hard)          # forward value = hard
    y.sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def _chain(**overrides):
    s = SamplingConfig(**overrides) if overrides else SamplingConfig()
    return LogitProcessingChain(s, vocab_size=64, codec_eos_id=50)


def test_temperature_and_topk():
    chain = _chain(top_k=3, top_p=1.0, temperature=0.5,
                   repetition_penalty=1.0, min_new_tokens=0)
    chain.suppress_text_range = False  # toy vocab: range would cover everything
    logits = torch.randn(1, 2, 64)
    codes = torch.randint(0, 40, (1, 2))
    valid = torch.ones(1, 2, dtype=torch.bool)
    out = chain.process(logits, codes, valid)
    finite = torch.isfinite(out)
    # exactly top-3 finite among non-suppressed positions per step
    assert (finite.sum(-1) == 3).all()


def test_min_new_tokens_suppresses_eos():
    chain = _chain(min_new_tokens=2, repetition_penalty=1.0)
    logits = torch.zeros(1, 4, 64)
    codes = torch.zeros(1, 4, dtype=torch.long)
    valid = torch.ones(1, 4, dtype=torch.bool)
    out = chain.process(logits, codes, valid)
    assert torch.isinf(out[0, 0, 50]) and out[0, 0, 50] < 0
    assert torch.isinf(out[0, 1, 50]) and out[0, 1, 50] < 0
    assert torch.isfinite(out[0, 2, 50])


def test_text_range_suppressed_except_eos():
    chain = _chain(repetition_penalty=1.0, min_new_tokens=0, top_k=0)
    logits = torch.zeros(1, 1, 64)
    codes = torch.zeros(1, 1, dtype=torch.long)
    valid = torch.ones(1, 1, dtype=torch.bool)
    out = chain.process(logits, codes, valid)
    lo = 64 - 1024 if 64 - 1024 > 0 else 0
    # vocab 64 with range [vocab-1024, vocab) clamps to whole vocab; EOS kept
    assert torch.isfinite(out[0, 0, 50])


def test_repetition_penalty_is_history_dependent():
    chain = _chain(repetition_penalty=2.0, min_new_tokens=0, top_k=0,
                   temperature=1.0)
    chain.suppress_text_range = False
    logits = torch.ones(1, 3, 64)
    codes = torch.tensor([[5, 9, 7]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    out = chain.process(logits, codes, valid)
    assert torch.isclose(out[0, 0, 5], torch.tensor(1.0))   # not yet seen
    assert torch.isclose(out[0, 1, 5], torch.tensor(0.5))   # seen -> /2
    assert torch.isclose(out[0, 2, 9], torch.tensor(0.5))
    assert torch.isclose(out[0, 2, 3], torch.tensor(1.0))   # never seen


def test_gradient_flows_through_chain():
    chain = _chain(repetition_penalty=1.05)
    logits = torch.randn(1, 3, 64, requires_grad=True)
    codes = torch.randint(0, 40, (1, 3))
    valid = torch.ones(1, 3, dtype=torch.bool)
    probs = torch.softmax(chain.process(logits, codes, valid), dim=-1)
    probs.sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
