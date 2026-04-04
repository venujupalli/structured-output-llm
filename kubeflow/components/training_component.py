from kfp import dsl


@dsl.component(base_image="your-registry/structured-output-llm:latest")
def training_component(
    model_config_path: str,
    training_config_path: str,
    output_model_path: dsl.OutputPath(str),
):
    import json
    import os
    import subprocess

    cmd = [
        "/app/docker/entrypoint.sh",
        "train",
        "--model-config",
        model_config_path,
        "--training-config",
        training_config_path,
    ]
    subprocess.run(cmd, check=True, env=os.environ.copy())

    training_output_dir = os.environ.get("TRAINING_OUTPUT_DIR", "/mnt/output/qwen25-structured/adapter")
    with open(output_model_path, "w", encoding="utf-8") as f:
        json.dump({"adapter_path": training_output_dir}, f)
