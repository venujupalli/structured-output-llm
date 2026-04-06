# User Guide: Cluster Setup, Training Jobs, and Evaluation Jobs

This guide explains how to prepare a Kubernetes cluster and run training/evaluation workloads for this project.

## 1) Prerequisites

Make sure you have:

- A Kubernetes cluster you can access with `kubectl`.
- Docker installed (for building/pushing your image).
- Permissions to create namespaces, PVCs, and Jobs in the cluster.

Project defaults used by `make`:

- `NAMESPACE=llm-training`
- `IMAGE_NAME=structured-output-llm`
- `DOCKER_TAG=latest`

You can override any of these per command, for example:

```bash
make setup NAMESPACE=my-ml-namespace
```

## 2) Build and Push the Container Image

The Kubernetes manifests reference an image that you must publish to a registry your cluster can pull from.

1. Build the image:

   ```bash
   make build IMAGE_NAME=<registry>/<repo>/structured-output-llm DOCKER_TAG=<tag>
   ```

2. Push the image if your cluster cannot read your local Docker images:

   ```bash
   make push IMAGE_NAME=<registry>/<repo>/structured-output-llm DOCKER_TAG=<tag>
   ```

For Docker Desktop Kubernetes, the default local image `structured-output-llm:latest` can be used directly after a local build.

## 3) Prepare Kubernetes Manifests

Before deploying, validate these key settings:

### Storage

`k8s/pvc.yaml` defines one PVC:

- `llm-models-pvc` (20Gi, ReadWriteOnce)

Update storage class / access mode if your cluster requires different settings.

### Training Job

`k8s/training-job.yaml` uses:

- Job name: `llm-training-job`
- Command: `/app/docker/entrypoint.sh train ...`
- Config/data mounts:
  - repo-bundled synthetic data from `/app/data/synthetic/*.jsonl`
  - `/models` from `llm-models-pvc`
  - `/config` from ConfigMap `llm-pipeline-config`
- Uses `configs/model_config.k8s.yaml` and `configs/training_config.k8s.yaml` for a lightweight Kubernetes smoke deployment.

### Evaluation Job

`k8s/evaluation-job.yaml` uses:

- Job name: `llm-evaluation-job`
- Command: `/app/docker/entrypoint.sh evaluate ...`
- Mounted volumes are `/models` and `/config`

### ConfigMap dependency

The `k8s/kustomization.yaml` bundle includes the `llm-pipeline-config` ConfigMap automatically when you run the `make` targets below.

## 4) Set Up the Cluster Namespace

Initialize cluster access and namespace:

```bash
make setup NAMESPACE=<namespace>
```

Apply persistent volumes:

```bash
make apply-pvc NAMESPACE=<namespace>
```

## 5) Deploy Training and Evaluation Jobs

Apply manifests:

```bash
make deploy-training NAMESPACE=<namespace>
make deploy-eval NAMESPACE=<namespace>
```

Or deploy all resources at once:

```bash
make deploy-all NAMESPACE=<namespace>
```

`deploy-all` now applies storage, config, and the training job. Run evaluation after training completes:

```bash
make deploy-eval NAMESPACE=<namespace>
```

If you want to evaluate the base model without any adapter, use:

```bash
make deploy-eval-base NAMESPACE=<namespace>
```

## 6) Run / Re-run Jobs Safely

The helper targets `run-training` and `run-eval` delete jobs by name before re-applying manifests.

Because the manifests use names `llm-training-job` and `llm-evaluation-job`, run these commands with matching overrides:

```bash
make run-training NAMESPACE=<namespace> TRAINING_JOB_NAME=llm-training-job
make run-eval NAMESPACE=<namespace> EVAL_JOB_NAME=llm-evaluation-job
```

Run both:

```bash
make run-all NAMESPACE=<namespace> \
  TRAINING_JOB_NAME=llm-training-job \
  EVAL_JOB_NAME=llm-evaluation-job
```

## 7) Monitor Job Progress

Check current pods and jobs:

```bash
make status NAMESPACE=<namespace>
```

Stream training logs:

```bash
make logs-training NAMESPACE=<namespace> TRAINING_JOB_NAME=llm-training-job
```

Stream evaluation logs:

```bash
make logs-eval NAMESPACE=<namespace> EVAL_JOB_NAME=llm-evaluation-job
```

Inspect failures/details:

```bash
make describe-training NAMESPACE=<namespace> TRAINING_JOB_NAME=llm-training-job
make describe-eval NAMESPACE=<namespace> EVAL_JOB_NAME=llm-evaluation-job
```

## 8) Cleanup

Delete jobs and related pods:

```bash
make clean NAMESPACE=<namespace> \
  TRAINING_JOB_NAME=llm-training-job \
  EVAL_JOB_NAME=llm-evaluation-job
```

Delete jobs/pods and PVCs:

```bash
make clean-all NAMESPACE=<namespace> \
  TRAINING_JOB_NAME=llm-training-job \
  EVAL_JOB_NAME=llm-evaluation-job
```

## 9) Optional: Local Training/Evaluation Script Entry Points

If you want to test outside Kubernetes first:

```bash
uv sync
bash scripts/run_training_local.sh
bash scripts/run_evaluation_local.sh
```

The local training and evaluation scripts use `uv run` with Mac-friendly defaults under `configs/model_config.local.yaml` and `configs/training_config.local.yaml`.

You can choose a local training preset with `TRAINING_PRESET`:

```bash
TRAINING_PRESET=speed bash scripts/run_training_local.sh
TRAINING_PRESET=balanced bash scripts/run_training_local.sh
TRAINING_PRESET=quality bash scripts/run_training_local.sh
```

`balanced` is the default preset when `TRAINING_PRESET` is not set.

### Local model/adaptor defaults

For local MacBook runs, `configs/model_config.local.yaml` now uses `model_name: auto` and `adapter_mode: auto`.

- On macOS, the project picks a Qwen2.5 Instruct model tier based on available system memory:
  - up to 12 GB: `Qwen/Qwen2.5-0.5B-Instruct`
  - 12 GB to 24 GB: `Qwen/Qwen2.5-1.5B-Instruct`
  - above 24 GB: `Qwen/Qwen2.5-3B-Instruct`
- On Apple Silicon / non-CUDA local setups, the project uses LoRA by default.
- On CUDA setups with `bitsandbytes` available, `adapter_mode: auto` upgrades to QLoRA automatically.

This avoids the unsupported 4-bit `bitsandbytes` path on most MacBook environments while still letting GPU environments use QLoRA.

### Local UI

To launch a minimal Streamlit dashboard for local training and evaluation:

```bash
UV_CACHE_DIR=.cache/uv uv sync
bash scripts/run_ui.sh
```

The dashboard lets you:

- trigger local training
- run local evaluation on the default validation set
- run evaluation on the golden set at `data/golden/golden.jsonl`
- tail recent logs for each job
- inspect the latest local and golden evaluation reports
