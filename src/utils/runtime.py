from __future__ import annotations

import logging
import os
import platform
from typing import Any

import torch


LOGGER = logging.getLogger(__name__)


def detect_system_memory_gb() -> float | None:
    try:
        total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return total_bytes / (1024**3)
    except (AttributeError, ValueError, OSError):
        return None


def is_macos() -> bool:
    return platform.system().lower() == "darwin"


def has_mps() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def has_cuda() -> bool:
    return torch.cuda.is_available()


def recommend_model_name(config: dict[str, Any]) -> str:
    explicit = config.get("model_name")
    if explicit and explicit != "auto":
        return explicit

    memory_gb = detect_system_memory_gb() or 0
    if is_macos():
        if memory_gb <= 12:
            return "Qwen/Qwen2.5-0.5B-Instruct"
        if memory_gb <= 24:
            return "Qwen/Qwen2.5-1.5B-Instruct"
        return "Qwen/Qwen2.5-3B-Instruct"

    return config.get("default_model_name", "Qwen/Qwen2.5-3B-Instruct")


def resolve_adapter_mode(config: dict[str, Any]) -> str:
    requested = str(config.get("adapter_mode", "auto")).lower()
    if requested in {"lora", "qlora"}:
        if requested == "qlora" and not supports_qlora():
            LOGGER.warning("QLoRA requested, but 4-bit bitsandbytes is not supported here. Falling back to LoRA.")
            return "lora"
        return requested

    return "qlora" if supports_qlora() else "lora"


def supports_qlora() -> bool:
    if not has_cuda():
        return False
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_device(config: dict[str, Any]) -> str:
    requested = config.get("device_map", "auto")
    if requested != "auto":
        return str(requested)
    if has_cuda():
        return "auto"
    if has_mps():
        return "mps"
    return "cpu"


def accelerator_report(config: dict[str, Any]) -> dict[str, Any]:
    selected_device = resolve_device(config)
    report = {
        "selected_device": selected_device,
        "cuda_available": has_cuda(),
        "mps_built": bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_built(),
        "mps_available": has_mps(),
        "torch_version": torch.__version__,
        "python_arch": platform.machine(),
    }

    if selected_device == "cpu":
        if report["mps_built"] and not report["mps_available"]:
            report["reason"] = "MPS is built into this PyTorch install, but not available at runtime."
        elif not report["mps_built"] and is_macos():
            report["reason"] = "This PyTorch install was built without MPS support."
        else:
            report["reason"] = "No GPU backend is available in the current runtime."
    elif selected_device == "mps":
        report["reason"] = "Using Apple Metal Performance Shaders."
    else:
        report["reason"] = "Using CUDA acceleration."

    return report


def log_accelerator_report(logger: logging.Logger, config: dict[str, Any], *, context: str) -> None:
    report = accelerator_report(config)
    logger.info(
        "%s accelerator: device=%s cuda_available=%s mps_built=%s mps_available=%s torch=%s arch=%s",
        context,
        report["selected_device"],
        report["cuda_available"],
        report["mps_built"],
        report["mps_available"],
        report["torch_version"],
        report["python_arch"],
    )
    logger.info("%s accelerator detail: %s", context, report["reason"])
