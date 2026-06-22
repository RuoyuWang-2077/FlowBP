"""Evaluation utilities for FlowBP.

Heavy generation dependencies such as diffusers are imported by the concrete
runner modules, not at package import time.
"""

__all__: list[str] = []
