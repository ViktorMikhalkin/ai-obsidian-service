from __future__ import annotations

import os
import pytest


# --- Minimal, safe defaults for local/unit runs --------------------------------
# Don't override if user/CI already set these.
os.environ.setdefault("AIOBS_TEST_MODE", "1")            # test-safe paths/dirs
os.environ.setdefault("VECTOR_STORE_BACKEND", "memory")  # unit tests use memory
# Keep Ollama off for unit tests
os.environ.setdefault("OLLAMA_BASE_URL", "")
os.environ.setdefault("OLLAMA_MODEL", "")


# --- Capability probes ----------------------------------------------------------
def _has_sentence_transformers() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False


def _has_faiss_cpu() -> bool:
    try:
        import faiss  # type: ignore
        # tiny smoke to ensure the native lib is really loadable
        _ = faiss.IndexFlatL2(2)
        return True
    except Exception:
        return False


def _has_faiss_gpu() -> bool:
    """
    Best-effort GPU probe: FAISS with GPU bindings + a basic resource allocation.
    We don't fail hard if drivers are absent—just return False.
    """
    try:
        import faiss  # type: ignore
        if not hasattr(faiss, "StandardGpuResources"):
            return False
        res = faiss.StandardGpuResources()  # may raise if no CUDA/runtime
        del res
        return True
    except Exception:
        return False


def _has_torch_cuda() -> bool:
    try:
        import torch  # noqa: F401
        return bool(torch.cuda.is_available())  # type: ignore[attr-defined]
    except Exception:
        return False


HAS_ST = _has_sentence_transformers()
HAS_FAISS_CPU = _has_faiss_cpu()
HAS_FAISS_GPU = _has_faiss_gpu()
HAS_TORCH_CUDA = _has_torch_cuda()


# --- Marker-based skipping policy ------------------------------------------------
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """
    Respect project markers declared in pyproject.toml:
      - unit (default)
      - integration_cpu
      - integration_gpu
      - requires_faiss
      - requires_st
      - e2e
      - faiss
    """
    skip_faiss_cpu = pytest.mark.skip(reason="FAISS (CPU) not available")
    skip_faiss_gpu = pytest.mark.skip(reason="FAISS (GPU) / CUDA not available")
    skip_st = pytest.mark.skip(reason="sentence-transformers not available")
    skip_cuda = pytest.mark.skip(reason="CUDA not available")

    for item in items:
        # CPU FAISS–dependent tests
        if ("integration_cpu" in item.keywords) or ("faiss" in item.keywords) or ("requires_faiss" in item.keywords):
            if not HAS_FAISS_CPU:
                item.add_marker(skip_faiss_cpu)

        # GPU / CUDA dependent tests
        if "integration_gpu" in item.keywords:
            if not (HAS_FAISS_GPU and HAS_TORCH_CUDA):
                item.add_marker(skip_faiss_gpu if not HAS_FAISS_GPU else skip_cuda)

        # sentence-transformers dependent tests
        if "requires_st" in item.keywords:
            if not HAS_ST:
                item.add_marker(skip_st)


# --- Optional: make env visible in -s logs (handy for debugging) ----------------
def pytest_report_header(config: pytest.Config) -> str:
    return (
        "Env: "
        f"VECTOR_STORE_BACKEND={os.environ.get('VECTOR_STORE_BACKEND')} | "
        f"AIOBS_TEST_MODE={os.environ.get('AIOBS_TEST_MODE')} | "
        f"FAISS_CPU={HAS_FAISS_CPU} FAISS_GPU={HAS_FAISS_GPU} | "
        f"CUDA={HAS_TORCH_CUDA} | ST={HAS_ST}"
    )