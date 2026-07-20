import torch

from ste_optimized.codec import _reference_one_hot


def test_reference_one_hot_accepts_inference_tensor_outside_inference_mode():
    with torch.inference_mode():
        ref_codes = torch.tensor([[0, 3], [9, -2]])

    encoded = _reference_one_hot(
        ref_codes, torch.device("cpu"), vocab_size=8, dtype=torch.float32
    )

    assert encoded.shape == (2, 2, 8)
    assert encoded[0, 0, 0] == 1
    assert encoded[0, 1, 3] == 1
    assert encoded[1, 0, 7] == 1
    assert encoded[1, 1, 0] == 1
