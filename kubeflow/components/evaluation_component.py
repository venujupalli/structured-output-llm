from kfp import dsl


@dsl.component(base_image="your-registry/structured-output-llm:latest")
def evaluation_component(
    model_config_path: str,
    training_config_path: str,
    schema_config_path: str,
    model_artifact: dsl.InputPath(str),
    eval_report_path: dsl.OutputPath(str),
):
    import json
    import os
    import subprocess

    with open(model_artifact, "r", encoding="utf-8") as f:
        model_info = json.load(f)

    env = os.environ.copy()
    env["TRAINING_OUTPUT_DIR"] = model_info["adapter_path"]

    cmd = [
        "/app/docker/entrypoint.sh",
        "evaluate",
        "--model-config",
        model_config_path,
        "--training-config",
        training_config_path,
        "--schema-config",
        schema_config_path,
        "--output-report",
        eval_report_path,
    ]
    subprocess.run(cmd, check=True, env=env)
