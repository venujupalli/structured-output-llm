from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.utils.config import load_yaml


@dataclass
class ModelLoaderConfig:
    model_name: str
    cache_dir: Optional[str] = None
    load_in_4bit: bool = False
    device_map: str = "auto"
    trust_remote_code: bool = False
    max_retries: int = 3
    retry_backoff_seconds: int = 2

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "ModelLoaderConfig":
        data: Dict[str, Any] = load_yaml(config_path)
        return cls(
            model_name=data["model_name"],
            cache_dir=data.get("cache_dir"),
            load_in_4bit=bool(data.get("load_in_4bit", False)),
            device_map=str(data.get("device_map", "auto")),
            trust_remote_code=bool(data.get("trust_remote_code", False)),
            max_retries=int(data.get("max_retries", 3)),
            retry_backoff_seconds=int(data.get("retry_backoff_seconds", 2)),
        )


class ModelLoader:
    def __init__(self, config: ModelLoaderConfig, logger: Optional[logging.Logger] = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def load(self):
        cache_dir = self._resolve_cache_dir(self.config.cache_dir)
        model_source = self._ensure_model_available(cache_dir)

        tokenizer = self._load_tokenizer(model_source, cache_dir)
        model = self._load_model(model_source, cache_dir)

        self.logger.info("Model and tokenizer loaded successfully from %s", model_source)
        return model, tokenizer

    def _resolve_cache_dir(self, configured_cache_dir: Optional[str]) -> Path:
        env_cache_dir = os.getenv("TRANSFORMERS_CACHE") or os.getenv("HF_HOME")
        selected = configured_cache_dir or env_cache_dir
        if selected is None:
            selected = str(Path.home() / ".cache" / "huggingface")

        cache_dir = Path(selected).expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Using model cache directory: %s", cache_dir)
        return cache_dir

    def _ensure_model_available(self, cache_dir: Path) -> str:
        model_name = self.config.model_name

        try:
            local_snapshot = snapshot_download(
                repo_id=model_name,
                cache_dir=str(cache_dir),
                local_files_only=True,
                resume_download=True,
            )
            self.logger.info("Model already exists in cache: %s", local_snapshot)
            return local_snapshot
        except LocalEntryNotFoundError:
            self.logger.info("Model not found in local cache. Downloading %s", model_name)
        except Exception as exc:
            self.logger.warning("Local cache check failed for %s: %s", model_name, exc)

        last_error: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                self.logger.info(
                    "Downloading model snapshot (attempt %d/%d): %s",
                    attempt,
                    self.config.max_retries,
                    model_name,
                )
                snapshot_path = snapshot_download(
                    repo_id=model_name,
                    cache_dir=str(cache_dir),
                    local_files_only=False,
                    resume_download=True,
                )
                self.logger.info("Download completed: %s", snapshot_path)
                return snapshot_path
            except (HfHubHTTPError, LocalEntryNotFoundError, OSError, RuntimeError) as exc:
                last_error = exc
                self.logger.exception("Download failed on attempt %d: %s", attempt, exc)
                self._cleanup_possible_corrupted_cache(cache_dir)
                if attempt < self.config.max_retries:
                    sleep_seconds = self.config.retry_backoff_seconds * attempt
                    self.logger.info("Retrying download in %d seconds", sleep_seconds)
                    time.sleep(sleep_seconds)

        raise RuntimeError(
            f"Failed to download model '{model_name}' after {self.config.max_retries} attempts"
        ) from last_error

    def _cleanup_possible_corrupted_cache(self, cache_dir: Path) -> None:
        repo_cache_dir = cache_dir / self._repo_cache_folder_name(self.config.model_name)
        if repo_cache_dir.exists():
            self.logger.warning("Removing potentially corrupted cache: %s", repo_cache_dir)
            shutil.rmtree(repo_cache_dir, ignore_errors=True)

    def _load_tokenizer(self, model_source: str, cache_dir: Path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                cache_dir=str(cache_dir),
                trust_remote_code=self.config.trust_remote_code,
                local_files_only=True,
                use_fast=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception:
            self.logger.exception("Tokenizer loading failed for source: %s", model_source)
            raise

    def _load_model(self, model_source: str, cache_dir: Path):
        kwargs: Dict[str, Any] = {
            "cache_dir": str(cache_dir),
            "device_map": self.config.device_map,
            "trust_remote_code": self.config.trust_remote_code,
            "local_files_only": True,
        }

        if self.config.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        try:
            return AutoModelForCausalLM.from_pretrained(model_source, **kwargs)
        except Exception:
            self.logger.exception("Model loading failed for source: %s", model_source)
            raise

    @staticmethod
    def _repo_cache_folder_name(model_name: str) -> str:
        return "models--" + model_name.replace("/", "--")


def load_model_and_tokenizer(config_path: str | Path) -> Tuple[Any, Any]:
    config = ModelLoaderConfig.from_yaml(config_path)
    return ModelLoader(config).load()
