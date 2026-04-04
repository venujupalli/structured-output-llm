# User Guide: Cluster Setup, Training Jobs, and Evaluation Jobs

This guide explains how to prepare a Kubernetes cluster and run training/evaluation workloads for this project.

## 1) Prerequisites

Make sure you have:

- A Kubernetes cluster you can access with `kubectl`.
- At least one GPU-capable node (the jobs request `nvidia.com/gpu: "1"`).
- Docker installed (for building/pushing your image).
- Permissions to create namespaces, PVCs, and Jobs in the cluster.

Project defaults used by `make`:

- `NAMESPACE=llm-training`
- `IMAGE_NAME=llm-training`
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

2. Push the image:

   ```bash
   make push IMAGE_NAME=<registry>/<repo>/structured-output-llm DOCKER_TAG=<tag>
   ```

3. Update both manifests to the same image/tag:

- `k8s/training-job.yaml` → `spec.template.spec.containers[0].image`
- `k8s/evaluation-job.yaml` → `spec.template.spec.containers[0].image`

## 3) Prepare Kubernetes Manifests

Before deploying, validate these key settings:

### Storage

`k8s/pvc.yaml` defines two PVCs:

- `llm-dataset-pvc` (100Gi, ReadWriteMany)
- `llm-models-pvc` (200Gi, ReadWriteMany)

Update storage class / access mode if your cluster requires different settings.

### Training Job

`k8s/training-job.yaml` uses:

- Job name: `llm-training-job`
- Command: `python src/training/train.py`
- Config/data mounts:
  - `/data` from `llm-dataset-pvc`
  - `/models` from `llm-models-pvc`
  - `/config` from ConfigMap `llm-pipeline-config`
- GPU node selector and toleration:
  - `nodeSelector: { accelerator: nvidia }`
  - toleration for key `nvidia.com/gpu`

### Evaluation Job

`k8s/evaluation-job.yaml` uses:

- Job name: `llm-evaluation-job`
- Command currently set to: `python src/evaluation/evaluator.py`
- Mounted volumes are the same as training (`/data`, `/models`, `/config`)

> Note: The repository’s module entry point is `src/evaluation/evaluate.py`. If your image does not include `src/evaluation/evaluator.py`, update the command in the evaluation manifest before deployment.

### ConfigMap dependency

Both jobs mount a ConfigMap named `llm-pipeline-config`. Create it before running jobs.

Example:

```bash
kubectl create configmap llm-pipeline-config \
  -n <namespace> \
  --from-file=configs/training_config.yaml \
  --from-file=configs/model_config.yaml \
  --from-file=configs/schema_config.yaml \
  --dry-run=client -o yaml | kubectl apply -f -
```

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
bash scripts/run_training.sh
bash scripts/run_evaluation.sh
```

These scripts read config paths from environment variables and use defaults under `configs/`.
