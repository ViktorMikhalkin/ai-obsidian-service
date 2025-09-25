import importlib
import pytest


def _available(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


def _has_faiss_cpu() -> bool:
    return _available("faiss")


def _has_faiss_gpu() -> bool:
    if not _available("faiss"):
        return False
    try:
        faiss = importlib.import_module("faiss")
        return hasattr(faiss, "StandardGpuResources")
    except Exception:
        return False


def _has_sentence_transformers() -> bool:
    return _available("sentence_transformers")


def _has_cuda() -> bool:
    if not _available("torch"):
        return False
    try:
        torch = importlib.import_module("torch")
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    faiss_ok = _has_faiss_cpu()
    st_ok = _has_sentence_transformers()
    faiss_gpu_ok = _has_faiss_gpu()
    cuda_ok = _has_cuda()

    for item in items:
        if "faiss" in item.keywords and not (faiss_ok and st_ok):
            item.add_marker(pytest.mark.skip(reason="FAISS and/or sentence-transformers not installed"))
        if "integration_cpu" in item.keywords and not (faiss_ok and st_ok):
            item.add_marker(pytest.mark.skip(reason="integration_cpu requires FAISS (CPU) and sentence-transformers"))
        if "integration_gpu" in item.keywords and not (faiss_gpu_ok and st_ok and cuda_ok):
            item.add_marker(pytest.mark.skip(reason="integration_gpu requires FAISS (GPU), sentence-transformers and CUDA"))
