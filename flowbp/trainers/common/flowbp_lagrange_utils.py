
"""Shared FlowBP-Lagrange connector utilities.

These helpers are backend-agnostic and are used by the SD3.5 FlowBP-Lagrange
trainer after removing the FLUX trainer modules.
"""

from __future__ import annotations

import os

import torch


# Buffer for accumulating j-k index samples (saved to CSV at end of training)
_jk_histogram_buffer: dict[str, list[float]] = {
    "k_idx": [],
    "j_idx": [],
    "jk_gap": [],
}


def save_jk_sampling_csv(output_dir: str) -> str | None:
    """Save accumulated j/k sampling data to CSV. Call at end of training."""
    import csv as _csv

    if not _jk_histogram_buffer["k_idx"]:
        return None
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "jk_sampling.csv")
    with open(csv_path, "w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(["step", "k_idx", "j_idx", "jk_gap", "seg_a", "seg_b", "seg_c"])
        n = len(_jk_histogram_buffer["k_idx"])
        seg_a = _jk_histogram_buffer.get("seg_a", [0.0] * n)
        seg_b = _jk_histogram_buffer.get("seg_b", [0.0] * n)
        seg_c = _jk_histogram_buffer.get("seg_c", [0.0] * n)
        for i in range(n):
            writer.writerow([
                i + 1,
                _jk_histogram_buffer["k_idx"][i],
                _jk_histogram_buffer["j_idx"][i],
                _jk_histogram_buffer["jk_gap"][i],
                seg_a[i] if i < len(seg_a) else 0.0,
                seg_b[i] if i < len(seg_b) else 0.0,
                seg_c[i] if i < len(seg_c) else 0.0,
            ])
    return csv_path


def _select_interval_support_indices(
    start_idx: int,
    target_idx: int,
    num_velocities: int,
    order: int = 4,
) -> list[int]:
    """Select approximately even velocity supports in [start_idx, target_idx)."""
    assert start_idx < target_idx, (start_idx, target_idx)
    assert 0 <= start_idx < num_velocities

    order = max(1, int(order))
    last_idx = min(target_idx - 1, num_velocities - 1)
    interval = list(range(start_idx, last_idx + 1))
    if len(interval) <= order:
        return interval

    positions = (
        torch.linspace(0, len(interval) - 1, steps=order)
        .round()
        .long()
        .tolist()
    )
    support = [interval[p] for p in positions]
    support[0] = start_idx

    dedup: list[int] = []
    for idx in support:
        if idx not in dedup:
            dedup.append(idx)
    return dedup


def _select_extrapolation_support_indices(
    start_idx: int,
    target_idx: int,
    num_velocities: int,
    order: int = 3,
) -> list[int]:
    """Pick the `order` velocity supports ending at start_idx (inclusive)."""
    assert 0 <= start_idx < num_velocities, (start_idx, num_velocities)
    assert target_idx > start_idx, (start_idx, target_idx)
    order = max(1, int(order))
    effective_order = min(order, start_idx + 1)
    return list(range(start_idx - effective_order + 1, start_idx + 1))


def _select_support_indices(
    weight_scheme: str,
    start_idx: int,
    target_idx: int,
    num_velocities: int,
    order: int,
) -> list[int]:
    """Dispatcher: pick supports based on the weight scheme."""
    scheme = str(weight_scheme).lower()
    if scheme == "adams_bashforth":
        return _select_extrapolation_support_indices(
            start_idx=start_idx,
            target_idx=target_idx,
            num_velocities=num_velocities,
            order=order,
        )
    return _select_interval_support_indices(
        start_idx=start_idx,
        target_idx=target_idx,
        num_velocities=num_velocities,
        order=order,
    )


def _select_active_support_indices(
    support_indices: list[int],
    start_idx: int,
    mode: str = "start",
    max_active: int | None = None,
) -> list[int]:
    """
    Select which support indices should participate in backward.
    support_indices is the list returned by _select_interval_support_indices.
    start_idx is always the first differentiable endpoint of this interval.
    """
    mode = str(mode).lower()
    if start_idx not in support_indices and mode != "none":
        raise ValueError(
            f"start_idx={start_idx} must be present in support_indices={support_indices}"
        )

    if mode == "none":
        selected: list[int] = []
    elif mode == "start":
        selected = [start_idx]
    elif mode == "midpoint":
        if len(support_indices) <= 1:
            selected = [start_idx]
        else:
            selected = support_indices[:2]
    elif mode == "all":
        selected = list(support_indices)
    else:
        raise ValueError(
            "Invalid flowbp_lagrange_grad_support_mode="
            f"{mode!r}; expected one of 'none', 'start', 'midpoint', or 'all'."
        )

    support_set = set(support_indices)
    ordered: list[int] = []
    for idx in selected:
        if idx in support_set and idx not in ordered:
            ordered.append(idx)
    if max_active is not None:
        max_active = int(max_active)
    if max_active is not None and max_active > 0:
        ordered = ordered[:max_active]
    return ordered


def _scale_nonstart_gradient_forward_value(
    tensor: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    Keep forward value unchanged but scale backward gradient by scale.
    If scale=1, return tensor.
    If scale=0, return tensor.detach().
    Otherwise return tensor.detach() + scale * (tensor - tensor.detach()).
    """
    if scale == 1:
        return tensor
    if scale == 0:
        return tensor.detach()
    detached = tensor.detach()
    return detached + float(scale) * (tensor - detached)


def _lagrange_integral_weights(
    sigma_points: torch.Tensor,
    sigma_start: torch.Tensor,
    sigma_target: torch.Tensor,
    device,
    dtype,
) -> torch.Tensor:
    """Weights for integrating the Lagrange interpolant over sigma."""
    points64 = sigma_points.detach().to(device=device, dtype=torch.float64)
    start64 = torch.as_tensor(sigma_start, device=device, dtype=torch.float64)
    target64 = torch.as_tensor(sigma_target, device=device, dtype=torch.float64)

    weights = []
    for i in range(points64.numel()):
        coeff = torch.ones(1, device=device, dtype=torch.float64)
        denom = torch.ones((), device=device, dtype=torch.float64)
        for j in range(points64.numel()):
            if j == i:
                continue
            new_coeff = torch.zeros(
                coeff.numel() + 1,
                device=device,
                dtype=torch.float64,
            )
            new_coeff[:-1] += -points64[j] * coeff
            new_coeff[1:] += coeff
            coeff = new_coeff
            denom = denom * (points64[i] - points64[j])

        coeff = coeff / denom
        integral = torch.zeros((), device=device, dtype=torch.float64)
        for power, c in enumerate(coeff):
            integral = integral + c / float(power + 1) * (
                target64 ** (power + 1) - start64 ** (power + 1)
            )
        weights.append(integral)

    return torch.stack(weights).to(device=device, dtype=dtype)


def _uniform_integral_weights(
    num_points: int,
    sigma_start: torch.Tensor,
    sigma_target: torch.Tensor,
    device,
    dtype,
) -> torch.Tensor:
    """Uniform velocity weights that preserve the interval integral length."""
    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}")
    start = torch.as_tensor(sigma_start, device=device, dtype=torch.float64)
    target = torch.as_tensor(sigma_target, device=device, dtype=torch.float64)
    weight = (target - start) / float(num_points)
    return torch.full((num_points,), weight.item(), device=device, dtype=dtype)


def _lagrange_quadrature_predict(
    x_start,
    current_v,
    cached_velocities,
    sigmas,
    start_idx,
    target_idx,
    order=4,
    detach_history=True,
    support_indices=None,
    active_velocities=None,
    grad_rescale=0.0,
    weight_scheme="lagrange",
):
    """FlowBP-Lagrange interval quadrature connector via Lagrange or uniform weights."""
    assert target_idx > start_idx, (start_idx, target_idx)
    assert len(sigmas) == len(cached_velocities) + 1
    assert 0 <= start_idx < len(cached_velocities)
    assert 0 < target_idx < len(sigmas)

    device = x_start.device
    dtype = torch.float32
    weight_scheme = str(weight_scheme).lower()
    if support_indices is None:
        support_indices = _select_support_indices(
            weight_scheme=weight_scheme,
            start_idx=start_idx,
            target_idx=target_idx,
            num_velocities=len(cached_velocities),
            order=order,
        )
    else:
        support_indices = list(support_indices)
    assert target_idx not in support_indices, (target_idx, support_indices)
    assert all(0 <= idx < len(cached_velocities) for idx in support_indices)

    use_start_velocity_gradient = active_velocities is None
    if active_velocities is None:
        active_velocities = {}
    active_support_indices = [
        idx for idx in support_indices if idx in active_velocities
    ]
    sigma_points = sigmas[support_indices].to(device=device, dtype=torch.float64)
    sigma_start = sigmas[start_idx].to(device=device, dtype=torch.float64)
    sigma_target = sigmas[target_idx].to(device=device, dtype=torch.float64)
    if weight_scheme in ("lagrange", "adams_bashforth"):
        weights = _lagrange_integral_weights(
            sigma_points=sigma_points,
            sigma_start=sigma_start,
            sigma_target=sigma_target,
            device=device,
            dtype=dtype,
        )
    elif weight_scheme == "uniform":
        weights = _uniform_integral_weights(
            num_points=len(support_indices),
            sigma_start=sigma_start,
            sigma_target=sigma_target,
            device=device,
            dtype=dtype,
        )
    else:
        raise ValueError(
            f"Invalid flowbp_lagrange_weight_scheme={weight_scheme!r}; "
            "expected 'lagrange', 'uniform', or 'adams_bashforth'."
        )
    start_pos = support_indices.index(start_idx)

    grad_rescale_factor = 1.0
    if grad_rescale > 0 and active_support_indices:
        total_abs_weight = weights.abs().sum()
        idx_to_pos = {idx: pos for pos, idx in enumerate(support_indices)}
        active_abs_weight = sum(
            weights[idx_to_pos[idx]].abs() for idx in active_support_indices
        )
        if active_abs_weight > 1e-8:
            full_factor = (total_abs_weight / active_abs_weight).item()
            grad_rescale_factor = 1.0 + grad_rescale * (full_factor - 1.0)

    delta = torch.zeros_like(x_start, dtype=torch.float32)
    for weight, idx in zip(weights, support_indices):
        if idx in active_velocities:
            velocity = active_velocities[idx]
            if grad_rescale_factor != 1.0:
                v_detached = velocity.detach()
                velocity = v_detached + grad_rescale_factor * (velocity - v_detached)
        elif idx == start_idx and use_start_velocity_gradient:
            velocity = current_v
            if grad_rescale_factor != 1.0:
                v_detached = velocity.detach()
                velocity = v_detached + grad_rescale_factor * (velocity - v_detached)
        else:
            velocity = cached_velocities[idx]
            if detach_history:
                velocity = velocity.detach()
        delta = delta + weight * velocity.float()

    info = {
        "support_indices": support_indices,
        "active_support_indices": active_support_indices,
        "weights": weights.detach(),
        "start_weight": weights[start_pos].detach(),
        "weight_abs_sum": weights.abs().sum().detach(),
        "grad_rescale_factor": torch.tensor(
            grad_rescale_factor, device=device, dtype=torch.float32
        ),
    }
    return x_start.float() + delta, info
