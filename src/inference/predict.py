from __future__ import annotations

import argparse
import logging

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.logging_utils import setup_logging
from src.utils.runtime import has_mps, log_accelerator_report


LOGGER = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    log_accelerator_report(
        LOGGER,
        {"device_map": "mps" if has_mps() else "cpu"},
        context="Inference",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
    model_kwargs = {"device_map": {"": "mps"}} if has_mps() else {"device_map": "auto"}
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model = PeftModel.from_pretrained(model, args.adapter_path)

    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=256)
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
