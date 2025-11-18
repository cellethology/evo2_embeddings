# Evo2 Models and Context Lengths

| Model Name | Parameters | Context Length | Description |
|------------|------------|----------------|-------------|
| `evo2_7b` | 7B | 1,000,000 (1M) | 7B parameter model with 1M context |
| `evo2_40b` | 40B | 1,000,000 (1M) | 40B parameter model with 1M context (requires multiple GPUs) |
| `evo2_7b_base` | 7B | 8,192 (8K) | 7B parameter model with 8K context |
| `evo2_40b_base` | 40B | 8,192 (8K) | 40B parameter model with 8K context |
| `evo2_1b_base` | 1B | 8,192 (8K) | Smaller 1B parameter model with 8K context |
| `evo2_7b_262k` | 7B | 262,144 (262K) | 7B parameter model with 262K context |
| `evo2_7b_microviridae` | 7B | 8,192 (8K) | 7B parameter base model fine-tuned on Microviridae genomes |

## Notes

- **Context Length**: Maximum number of tokens/base pairs the model can process in a single forward pass
- **40B Models**: Require multiple GPUs. Vortex automatically handles device placement, splitting the model across available CUDA devices
- **FP8 Requirements**: The 40B and 1B models require FP8 for numerical accuracy. The 7B models can run without FP8

