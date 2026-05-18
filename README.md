# Structured Output LLM Platform

Production-style LLM fine-tuning and evaluation platform for generating reliable structured JSON outputs using QLoRA, Kubernetes, Kubeflow, and MLflow.

Overview

Large Language Models (LLMs) frequently generate malformed or inconsistent structured outputs, making enterprise automation workflows unreliable.

This project focuses on improving structured JSON generation reliability through:

- QLoRA-based fine-tuning
- Schema-aware evaluation
- JSON validation pipelines
- Experiment tracking with MLflow
- Kubernetes-native orchestration
- Kubeflow pipeline automation
- Synthetic and golden dataset evaluation

The platform is designed to simulate production-grade structured-output workflows commonly used in:

- E-commerce automation
- AI-powered APIs
- Document extraction systems
- Enterprise AI pipelines
- Workflow orchestration systems

---

## Key Features

### Fine-Tuning Pipeline
- QLoRA-based fine-tuning for efficient LLM adaptation
- PEFT + BitsAndBytes integration
- Config-driven training workflows
- GPU-aware runtime handling

### Evaluation Framework
- JSON Parse Success validation
- Schema Validation Accuracy
- Required Field Completeness analysis
- Golden dataset benchmarking
- Structured-output consistency evaluation

### MLOps & Infrastructure
- Kubernetes-native deployment workflows
- Kubeflow pipeline orchestration
- MLflow experiment tracking
- Dockerized execution environments
- Configurable Makefile-based automation

### Data Pipeline
- Synthetic dataset generation
- Golden validation datasets
- Structured schema definitions
- Automated preprocessing workflows

---

## Repository Structure

```text
configs/        -> Training, schema, and runtime configs
data/           -> Synthetic and golden datasets
docker/         -> Docker environments
k8s/            -> Kubernetes deployment specs
kubeflow/       -> Kubeflow pipelines
scripts/        -> Automation scripts
src/            -> Core application logic
```

---

## System Architecture

```text
Synthetic / Golden Datasets
            ↓
     Preprocessing Pipeline
            ↓
      QLoRA Fine-Tuning
            ↓
      MLflow Tracking
            ↓
    Structured Evaluation
            ↓
 Schema Validation Engine
            ↓
 Kubernetes / Kubeflow
            ↓
     Structured Inference
```

---

## Training Workflow

1. Generate synthetic and golden datasets
2. Preprocess structured training examples
3. Fine-tune Qwen 2.5 Instruct models using QLoRA
4. Track experiments with MLflow
5. Run evaluation pipelines against validation datasets
6. Validate outputs against predefined schemas
7. Deploy workflows using Kubernetes and Kubeflow

---

## Evaluation Metrics

The platform evaluates structured-output quality using:

| Metric | Description |
|--------|-------------|
| JSON Parse Success | Measures whether outputs can be parsed correctly |
| Schema Validation Accuracy | Validates schema compliance |
| Required Field Completeness | Ensures mandatory fields exist |
| Invalid Output Rate | Tracks malformed responses |
| Output Consistency | Measures structured-output reliability |

---

## Tech Stack

### AI / ML
- Python
- Hugging Face Transformers
- QLoRA
- PEFT
- BitsAndBytes

### Infrastructure
- Kubernetes
- Kubeflow
- Docker
- MLflow

### Data & Evaluation
- Pandas
- JSON Schema Validation
- Synthetic Dataset Generation

### Tooling
- Makefiles
- Config-driven workflows
- CLI automation

---

## Example Structured Output

### Input

```text
Customer John Doe purchased 2 wireless keyboards and 1 monitor for $450.
```

### Expected Structured Output

```json
{
  "customer_name": "John Doe",
  "items": [
    {
      "product": "wireless keyboard",
      "quantity": 2
    },
    {
      "product": "monitor",
      "quantity": 1
    }
  ],
  "total_amount": 450
}
```

---

## Running the Project

### Clone Repository

```bash
git clone https://github.com/venujupalli/structured-output-llm.git
cd structured-output-llm
```

### Install Dependencies

```bash
uv sync
```

or

```bash
pip install -r requirements.txt
```

---

## Run Training

```bash
make train
```

---

## Run Evaluation

```bash
make evaluate
```

---

## Kubernetes Deployment

```bash
make deploy-k8s
```

---

## Kubeflow Pipeline

Kubeflow pipelines are available under:

```text
kubeflow/
```

These pipelines support:
- training orchestration
- evaluation workflows
- experiment automation
- scalable execution

---

## Future Improvements

- Grammar-constrained decoding
- Advanced schema validation
- FastAPI inference serving
- Real-time monitoring dashboards
- Drift detection
- Multi-schema evaluation
- Distributed fine-tuning
- Human evaluation workflows

---

## Contributors

- Venugopal Rao Jupalli
- Narender Rao Surabhi

---

## License

MIT License
