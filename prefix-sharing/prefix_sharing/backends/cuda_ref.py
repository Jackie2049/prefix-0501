"""CUDA reference backend placeholder.

Phase 1 keeps CUDA correctness behind the same PyTorch implementation when
torch+CUDA are available. High-performance TE/flash-attn kernels are phase 2.
"""

from __future__ import annotations

from prefix_sharing.backends.torch_ref import TorchReferenceBackend


class CudaReferenceBackend(TorchReferenceBackend):
    pass
