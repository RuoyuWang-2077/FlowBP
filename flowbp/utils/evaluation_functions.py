"""Backward-compatible online evaluation wrapper."""

from flowbp.eval.runners.online import run_online_flux_eval


def flux_evaluation(*args, **kwargs):
    return run_online_flux_eval(*args, **kwargs)
