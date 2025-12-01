# Evo2 Embeddings Framework
## Overview

The Evo2 Embeddings framework provides tools to:
- Extract embeddings from FASTA sequences using pre-trained Evo2 models
- Process sequences in batches for efficient inference on single or multiple GPUs
- Support multiple model sizes (1B, 7B, 40B parameters) with varying context lengths
- Handle variable-length sequences with automatic padding and BOS token management

## Prerequisites

- Python 3.11 or 3.12
- CUDA-capable GPU (recommended for faster inference)
- Miniconda or Anaconda (for environment management)
- CUDA toolkit (for GPU support)

## Environment Setup

### 1. Install Miniconda (if not already installed)

Download and install Miniconda from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

### 2. Create Virtual Environment and Install Dependencies

Run the environment setup script:

```bash
bash env_setup.sh
```

This script will:
- Create a conda environment named `evo2_embeddings` with Python 3.12
- Install CUDA toolkit and development tools
- Install transformer-engine with PyTorch support
- Install required dependencies including `evo2`, `biopython`, `torch`, and others

### 3. Activate the Environment

```bash
conda activate evo2_embeddings
```

### 4. Model Download

Evo2 models are automatically downloaded from HuggingFace when first used. The models will be cached in your HuggingFace cache directory (typically `~/.cache/huggingface/`).

**Note:** Model files can be large (several GB for 7B models, tens of GB for 40B models). Ensure you have sufficient disk space and bandwidth.

## Available Models

| Model Name | Parameters | Context Length | Description |
|------------|------------|----------------|-------------|
| `evo2_7b` | 7B | 1,000,000 (1M) | 7B parameter model with 1M context |
| `evo2_40b` | 40B | 1,000,000 (1M) | 40B parameter model with 1M context (requires multiple GPUs) |
| `evo2_7b_base` | 7B | 8,192 (8K) | 7B parameter model with 8K context |
| `evo2_40b_base` | 40B | 8,192 (8K) | 40B parameter model with 8K context |
| `evo2_1b_base` | 1B | 8,192 (8K) | Smaller 1B parameter model with 8K context |
| `evo2_7b_262k` | 7B | 262,144 (262K) | 7B parameter model with 262K context |
| `evo2_7b_microviridae` | 7B | 8,192 (8K) | 7B parameter base model fine-tuned on Microviridae genomes |

**Notes:**
- **Context Length**: Maximum number of tokens/base pairs the model can process in a single forward pass
- **40B Models**: Require multiple GPUs. The framework automatically handles device placement, splitting the model across available CUDA devices
- **FP8 Requirements**: The 40B and 1B models require FP8 for numerical accuracy. The 7B models can run without FP8
- **Model Selection**: For single GPU setups, use `evo2_1b_base` or `evo2_7b_base` for faster processing

## Usage

### Extract Embeddings from FASTA Sequences

The main script for extracting embeddings is `retrieve_embeddings/retrieve_embeddings.py`.

#### Basic Usage

```bash
python -m retrieve_embeddings.retrieve_embeddings \
    --fasta_path test_files/test.fasta \
    --output_path embeddings.npz
```

#### Full Command with All Options

```bash
python -m retrieve_embeddings.retrieve_embeddings \
    --fasta_path <path-to-input.fasta> \
    --output_path <path-to-output.npz> \
    --model_name evo2_7b \
    --batch_size 8 \
    --layer_name blocks.31.mlp.l3 \
    --device cuda:0 \
    --prepend_bos \
    --mean_pooling
```

#### Command-Line Arguments

- `--fasta_path`, `-i` (required): Path to input FASTA file containing DNA sequences
- `--output_path`, `-o` (optional): Path to output `.npz` file where embeddings will be saved (default: `./embeddings.npz`)
- `--model_name`, `-m` (optional): Evo2 model name to use (default: `evo2_7b`)
  - Options: `evo2_1b_base`, `evo2_7b`, `evo2_7b_base`, `evo2_7b_262k`, `evo2_40b`, `evo2_40b_base`, `evo2_7b_microviridae`
- `--batch_size`, `-b` (optional): Batch size for processing sequences (default: 8)
- `--layer_name`, `-l` (optional): Layer name to extract embeddings from (default: `blocks.31.mlp.l3`)
- `--device`, `-d` (optional): Device to run inference on (default: `cuda:0`)
  - Options: `cuda:0`, `cuda:1`, `cpu`, etc.
- `--prepend_bos` (optional): Prepend BOS (Beginning of Sequence) token to sequences (default: False)
- `--mean_pooling` (optional): Apply mean pooling to embeddings across sequence length. Averages per-token embeddings into a single vector per sequence, significantly reducing memory usage and output file size (default: False)

#### Output Format

The script outputs a compressed NumPy archive (`.npz`) file containing:
- `ids`: Array of sequence IDs from the FASTA file (string array)
- `embeddings`: Array of embeddings
  - **Without mean pooling**: Variable-length embeddings per sequence with shape `(sequence_length, embedding_dim)`
  - **With mean pooling** (`--mean_pooling`): Single vector per sequence with shape `(embedding_dim,)` - averages all per-token embeddings
  - Embedding dimension depends on the model and layer (typically 1024 or 2048)

**Memory Usage**: Using `--mean_pooling` significantly reduces memory usage during processing and output file size, especially for long sequences. Recommended for downstream tasks that don't require per-token embeddings.

#### Examples

```bash
# Extract embeddings using the default 7B model
python -m retrieve_embeddings.retrieve_embeddings \
    --fasta_path test_files/test.fasta \
    --output_path output/embeddings.npz

# Extract embeddings using the smaller 1B model (faster, less memory)
python -m retrieve_embeddings.retrieve_embeddings \
    --fasta_path test_files/test.fasta \
    --output_path output/embeddings.npz \
    --model_name evo2_1b_base \
    --batch_size 16

# Extract embeddings with BOS token prepended
python -m retrieve_embeddings.retrieve_embeddings \
    --fasta_path test_files/test.fasta \
    --output_path output/embeddings.npz \
    --prepend_bos

# Extract embeddings from a specific layer
python -m retrieve_embeddings.retrieve_embeddings \
    --fasta_path test_files/test.fasta \
    --output_path output/embeddings.npz \
    --layer_name blocks.15.mlp.l3

# Extract embeddings with mean pooling (reduces memory usage)
python -m retrieve_embeddings.retrieve_embeddings \
    --fasta_path test_files/test.fasta \
    --output_path output/embeddings_pooled.npz \
    --mean_pooling

# Extract embeddings with mean pooling and larger batch size
python -m retrieve_embeddings.retrieve_embeddings \
    --fasta_path test_files/test.fasta \
    --output_path output/embeddings_pooled.npz \
    --mean_pooling \
    --batch_size 16
```

### Loading Embeddings

You can load the saved embeddings in Python:

```python
import numpy as np

# Load embeddings
data = np.load('output/embeddings.npz', allow_pickle=True)
sequence_ids = data['ids']
embeddings = data['embeddings']  # Object array of variable-length arrays

print(f"Loaded {len(sequence_ids)} sequences")
print(f"Sequence IDs: {sequence_ids}")

# Access embeddings for each sequence
for i, seq_id in enumerate(sequence_ids):
    seq_embedding = embeddings[i]
    print(f"Sequence {seq_id}: embedding shape {seq_embedding.shape}")
    # Without mean pooling: (sequence_length, embedding_dim)
    # With mean pooling: (embedding_dim,)
```

**Note**: When using `--mean_pooling`, each embedding is a 1D array of shape `(embedding_dim,)` instead of `(sequence_length, embedding_dim)`. This is useful for tasks like sequence classification, similarity search, or clustering where a single vector per sequence is sufficient.

## Project Structure

```
evo2_embeddings/
├── evo2/                      # Evo2 model package
│   ├── models.py              # Evo2 model class and loading
│   ├── scoring.py             # Sequence scoring utilities
│   ├── utils.py               # Model utilities
│   └── configs/               # Model configuration files
│       ├── evo2-7b-1m.yml
│       ├── evo2-1b-8k.yml
│       └── ...
├── retrieve_embeddings/        # Embedding extraction scripts
│   ├── retrieve_embeddings.py  # Main embedding extraction script
│   └── util.py                # Utility functions (FASTA loading, validation, saving)
├── retrieve_test/             # Test scripts
├── test_files/                # Test data
│   ├── test.fasta             # Example input file
│   └── test_embeddings.npz    # Example output file
├── env_setup.sh               # Environment setup script
├── Dockerfile                 # Docker configuration
├── pyproject.toml              # Project dependencies
└── README.md                  # This file
```

## Input Format

The input FASTA file should contain DNA sequences with standard FASTA format:

```
>sequence_id_1
ATCGATCAGTACGATCAGATTTAGACGT
>sequence_id_2
TTTTGGGCGCGCGGCATCGATCAGTACGATCAGATTTAGACGTAAAAAA
>sequence_id_3
AGCTGATGCTAGCAGTGACGATGACAGTACAGTACAGAT
```

**Requirements:**
- Sequences must contain only valid DNA characters: A, T, C, G
- Sequences are automatically converted to uppercase
- Invalid sequences are skipped with a warning
- Maximum sequence length: 1,000,000 base pairs (for 1M context models)
- Minimum sequence length: 1 base pair