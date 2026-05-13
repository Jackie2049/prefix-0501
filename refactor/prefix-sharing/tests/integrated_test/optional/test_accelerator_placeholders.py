import pytest


def test_cann_optional_dependency_placeholder():
    torch_npu = pytest.importorskip("torch_npu")
    assert torch_npu is not None


def test_real_verl_megatron_optional_dependency_placeholder():
    verl = pytest.importorskip("verl")
    megatron = pytest.importorskip("megatron")
    assert verl is not None
    assert megatron is not None
