import torch


def pad_tensor_to_rank_size(tensor: torch.Tensor, rank_size: int) -> torch.Tensor:
    """
    verl DP Proto requires the batch size to be divisible by the DP size.
    This function pads the tensor to be divisible by the DP size using last row of the tensor.
    """
    pad_size = (rank_size - tensor.shape[0] % rank_size) % rank_size
    if pad_size == 0:
        return tensor
    else:
        last_row = tensor[-1, :]
        padded_tensor = torch.cat([tensor, last_row.repeat(pad_size, 1)], dim=0)
        return padded_tensor


def pad_tensor_batch_dim_with_zeros(tensor: torch.Tensor, multiple: int) -> torch.Tensor:
    """Pad dim 0 to the next multiple of ``multiple`` with zeros (no contribution under loss masks)."""
    pad_size = (multiple - tensor.shape[0] % multiple) % multiple
    if pad_size == 0:
        return tensor
    if tensor.is_nested:
        raise NotImplementedError("pad_tensor_batch_dim_with_zeros does not support nested tensors")
    tail = tensor.shape[1:]
    zeros = torch.zeros((pad_size, *tail), dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, zeros], dim=0)


def truncate_tensor_to_rank_size(tensor: torch.Tensor, rank_size: int) -> torch.Tensor:
    """
    Truncate the tensor along the first dimension so its size is divisible by rank_size (DP size).
    """
    size = tensor.shape[0]
    truncated_size = (size // rank_size) * rank_size
    if truncated_size == 0:
        raise ValueError(
            f"Cannot truncate tensor with size {size} to be divisible by rank_size {rank_size}; "
            "would result in empty tensor."
        )
    if truncated_size == size:
        return tensor
    return tensor[:truncated_size]
