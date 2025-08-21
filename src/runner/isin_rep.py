import torch

def torch_isin(elements: torch.Tensor, test_elements: torch.Tensor) -> torch.Tensor:
    """
    Compatibility wrapper for torch.isin().
    Returns a boolean tensor indicating whether each element in `elements` is in `test_elements`.

    Works in PyTorch < 1.10 where torch.isin is not available.
    """
    if hasattr(torch, 'isin'):
        return torch.isin(elements, test_elements)
    else:
        # Broadcasted comparison and reduction
        return (elements[..., None] == test_elements).any(-1)

# Optional: monkey-patch torch for uniform usage across your codebase
if not hasattr(torch, 'isin'):
    torch.isin = torch_isin
