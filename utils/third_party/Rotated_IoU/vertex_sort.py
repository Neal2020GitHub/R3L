import torch

MAX_NUM_VERT_IDX = 9
INTERSECTION_OFFSET = 8
EPSILON = 1e-8


def sort_vertices(vertices: torch.Tensor, mask: torch.Tensor, num_valid: torch.Tensor) -> torch.Tensor:
    """Sort convex polygon vertices with vectorized PyTorch operations.

    Args:
        vertices: (B, N, M, 2) centroid-normalized candidate vertices.
        mask: (B, N, M) bool mask for valid candidates.
        num_valid: (B, N) count of valid candidates.

    Returns:
        (B, N, 9) long indices matching the original reference behavior:
        sorted valid vertices, first vertex repeated, then padded with an
        invalid intersection index.
    """
    if vertices.ndim != 4 or vertices.shape[-1] != 2:
        raise ValueError(f"vertices must have shape (B, N, M, 2), got {tuple(vertices.shape)}")
    if mask.shape != vertices.shape[:-1]:
        raise ValueError(f"mask must have shape {tuple(vertices.shape[:-1])}, got {tuple(mask.shape)}")
    if num_valid.shape != vertices.shape[:2]:
        raise ValueError(f"num_valid must have shape {tuple(vertices.shape[:2])}, got {tuple(num_valid.shape)}")

    B, N, M, _ = vertices.shape
    if M < MAX_NUM_VERT_IDX:
        raise ValueError(f"vertices must include at least {MAX_NUM_VERT_IDX} candidates, got {M}")

    device = vertices.device
    mask = mask.bool()
    num_valid = num_valid.to(device=device, dtype=torch.long)

    # Match the CUDA contract: padding uses any invalid intersection candidate
    # (indices 8..M-1), whose original vertex is zero and has zero gradient.
    invalid_intersections = ~mask[..., INTERSECTION_OFFSET:]
    has_invalid = invalid_intersections.any(dim=-1)
    first_invalid = invalid_intersections.to(torch.long).argmax(dim=-1) + INTERSECTION_OFFSET
    pad = torch.where(has_invalid, first_invalid, torch.full_like(first_invalid, M - 1))

    idx = pad.unsqueeze(-1).expand(B, N, MAX_NUM_VERT_IDX).clone()

    valid_polygon = num_valid >= 3
    angles = torch.atan2(vertices[..., 1], vertices[..., 0])
    angles = torch.where(angles < 0, angles + 2 * torch.pi, angles)
    scores = torch.where(mask & valid_polygon.unsqueeze(-1), angles, torch.full_like(angles, float("inf")))
    order = torch.argsort(scores, dim=-1, stable=True)[..., : MAX_NUM_VERT_IDX - 1]

    slots = torch.arange(MAX_NUM_VERT_IDX - 1, device=device).view(1, 1, -1)
    take = valid_polygon.unsqueeze(-1) & (slots < num_valid.clamp(max=MAX_NUM_VERT_IDX - 1).unsqueeze(-1))
    idx[..., : MAX_NUM_VERT_IDX - 1] = torch.where(take, order, idx[..., : MAX_NUM_VERT_IDX - 1])

    # Close the polygon by duplicating the first sorted vertex at position num_valid.
    close_pos = num_valid.clamp(min=0, max=MAX_NUM_VERT_IDX - 1).unsqueeze(-1)
    close_val = torch.where(valid_polygon.unsqueeze(-1), idx[..., 0:1], pad.unsqueeze(-1))
    idx.scatter_(2, close_pos, close_val)

    # Reference special-case: identical boxes may produce duplicated corner
    # candidates. The original kernel cycles through the first box's corners
    # instead of taking adjacent duplicates from both boxes.
    corner_delta = torch.abs(vertices[..., :4, :] - vertices[..., 4:INTERSECTION_OFFSET, :]).amax(dim=-1)
    duplicated_corners = mask[..., :4] & mask[..., 4:INTERSECTION_OFFSET] & (corner_delta < EPSILON)
    duplicate_count = duplicated_corners.sum(dim=-1)
    corner_only = (mask[..., :INTERSECTION_OFFSET].sum(dim=-1).to(num_valid.dtype) == num_valid)
    duplicated_pair = corner_only & (num_valid == 2 * duplicate_count)

    first_corner_scores = torch.where(mask[..., :4], angles[..., :4], torch.full_like(angles[..., :4], float("inf")))
    first_corner_order = torch.argsort(first_corner_scores, dim=-1, stable=True)

    full_duplicate_pattern = torch.cat(
        [
            first_corner_order,
            first_corner_order[..., :1],
            pad.unsqueeze(-1).expand(B, N, 4),
        ],
        dim=-1,
    )
    partial_duplicate_pattern = torch.cat(
        [
            first_corner_order[..., :3],
            first_corner_order[..., :3],
            first_corner_order[..., :1],
            pad.unsqueeze(-1).expand(B, N, 2),
        ],
        dim=-1,
    )

    full_duplicate = duplicated_pair & (duplicate_count == 4)
    partial_duplicate = duplicated_pair & (duplicate_count == 3)
    idx = torch.where(full_duplicate.unsqueeze(-1), full_duplicate_pattern, idx)
    idx = torch.where(partial_duplicate.unsqueeze(-1), partial_duplicate_pattern, idx)
    return idx.long()


if __name__ == "__main__":
    v = torch.rand([8, 1024, 24, 2]).float()
    m = torch.rand([8, 1024, 24]) > 0.8
    nv = torch.sum(m.int(), dim=-1).int()
    print(sort_vertices(v, m, nv).shape)


sort_v = sort_vertices
