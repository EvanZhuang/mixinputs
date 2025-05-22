# MoI Example Scripts

This directory contains example scripts demonstrating how to use the Mixture of Inputs (MoI) technique with various models and datasets.

## Overview

Mixture of Inputs (MoI) is a simple, training-free technique that improves autoregressive text generation by mixing the sampled token with the original distribution to retain more information during inference. These example scripts show how to apply MoI to different tasks and models.

## Examples

### 1. AIME Math Problems

Run a model on the AIME (American Invitational Mathematics Examination) dataset:

```bash
# Run with example script
./aime.sh
```

**Key Scripts:**
- `aime_moi.py`: Python script for evaluating models on AIME dataset
- `aime.sh`: Shell script with preconfigured settings for running `aime_moi.py`

### 2. TinyZero Countdown Tasks

Run a model on the TinyZero Countdown arithmetic task dataset:

```bash
# Run with example script
./countdown.sh
```

**Key Scripts:**
- `tinyzero_moi.py`: Python script for evaluating models on the TinyZero Countdown dataset
- `countdown.sh`: Shell script with preconfigured settings for running `tinyzero_moi.py`

### 3. Model Serving

Run a model as a server with MoI enabled:

```bash
./serve.sh
```

## Configuration

### Environment Variables

- `MIXINPUTS_BETA`: Controls the strength of the MoI effect (default: 1.0)
- `MIXINPUTS_LOGGING_LEVEL`: Set logging level (e.g., INFO, DEBUG)
- `VLLM_USE_V1=1`: Enable vLLM v1 compatibility, must be turned on for MoI

### Common Parameters

The Python scripts accept several command-line arguments:

- `--model_name`: Hugging Face model name or path
- `--dataset_name`: Hugging Face dataset name or path
- `--max_new_tokens`: Maximum number of tokens to generate
- `--temperature`: Sampling temperature (lower is more deterministic)
- `--top_p`: Top-p (nucleus) sampling parameter
- `--eager`: Use eager mode for generation
- `--output_dir`: Directory to save results
- `--debug`: Enable debug output

## Results

Results are stored in the `results/` directory as JSON files with naming convention:
```
{model_name}={dataset_name}={top_p}={temperature}.json
```
We provide results for 2 sample runs.