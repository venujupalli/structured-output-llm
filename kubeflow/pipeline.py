from kfp import compiler, dsl

from kubeflow.components.evaluation_component import evaluation_component
from kubeflow.components.training_component import training_component


@dsl.component(base_image="python:3.11")
def data_prep_component(dataset_path: str) -> str:
    return dataset_path


@dsl.pipeline(name="structured-output-qlora-pipeline")
def structured_output_pipeline(
    dataset_path: str = "/mnt/data/train.json",
    model_config_path: str = "/app/configs/model_config.yaml",
    training_config_path: str = "/app/configs/training_config.yaml",
    schema_config_path: str = "/app/configs/schema_config.yaml",
):
    prepared_data = data_prep_component(dataset_path=dataset_path)

    train_task = training_component(
        model_config_path=model_config_path,
        training_config_path=training_config_path,
    )
    train_task.set_caching_options(True)
    train_task.after(prepared_data)

    eval_task = evaluation_component(
        model_config_path=model_config_path,
        training_config_path=training_config_path,
        schema_config_path=schema_config_path,
        model_artifact=train_task.outputs["output_model_path"],
    )
    eval_task.set_caching_options(True)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=structured_output_pipeline,
        package_path="kubeflow/structured_output_pipeline.yaml",
    )
