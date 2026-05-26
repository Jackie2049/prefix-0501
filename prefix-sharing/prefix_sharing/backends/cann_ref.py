"""CANN/NPU reference backend placeholder.

This backend intentionally avoids CUDA-specific assumptions. It reuses the
torch reference path when torch_npu is installed and otherwise remains an
optional test target.
"""

from __future__ import annotations

from prefix_sharing.backends.torch_ref import TorchReferenceBackend


class CannReferenceBackend(TorchReferenceBackend):
    pass
