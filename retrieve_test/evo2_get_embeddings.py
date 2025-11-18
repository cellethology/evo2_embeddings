import argparse
import gc
import math
import os
import signal
import sys
from typing import List

import numpy as np
import torch
from Bio import SeqIO
from evo2 import Evo2
from evo2.scoring import logits_to_logprobs, prepare_batch
from safetensors.torch import save_file
from tqdm.auto import tqdm

# apptainer exec --nv --bind $PWD:/app --bind ./models:/root/.cache/huggingface --pwd /app \
#   ./evo2.sif python3 scripts/alcantar_2025/evo_2_embed_with_csv_alcantar_single_gpu.py \
#   --csv_path embeddings/alcantar_2025/pre_embeddings/processed_chromatin_sequences.csv \
#   -o embeddings/alcantar_2025/post_embeddings --batch_size 4

LAYER_NAME = "blocks.31.mlp.l3"


def signal_handler(signum: int, frame) -> None:
    """Handle signals gracefully."""
    print(f"Received signal {signum}, cleaning up...", flush=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sys.exit(0)


def compute_log_likelihood(
    input_ids: torch.Tensor,
    model,
    seq_lengths: List[int],
    reduce_method: str = "mean",
    prepend_bos: bool = True,
) -> torch.Tensor:
    """
    Compute per-sequence log-likelihoods.

    Args:
        input_ids: Input token IDs tensor [B, T]
        model: Evo2 model
        seq_lengths: List of actual sequence lengths for each batch item
        reduce_method: "mean" or "sum" for reducing per-token logprobs
        prepend_bos: Whether BOS token was prepended

    Returns:
        Tensor of log-likelihoods per sequence [B]
    """
    with torch.inference_mode():
        outputs = model(input_ids)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

    logprobs = logits_to_logprobs(logits, input_ids)
    if isinstance(logprobs, np.ndarray):
        logprobs = torch.from_numpy(logprobs)
    logprobs = logprobs.float()

    B, T_eff = logprobs.shape[:2]
    if len(seq_lengths) != B:
        raise ValueError(f"seq_lengths size {len(seq_lengths)} != batch size {B}")

    # Adjust lengths if BOS was prepended
    adj_lengths = [
        max(0, min((L - 1 if prepend_bos and L > 0 else L), T_eff))
        for L in seq_lengths
    ]

    reduce_fn = torch.mean if reduce_method == "mean" else torch.sum
    if reduce_method not in ("mean", "sum"):
        raise ValueError(f"Invalid reduce_method {reduce_method}")

    results = []
    for i, Li in enumerate(adj_lengths):
        if Li == 0:
            results.append(torch.tensor(0.0))
        else:
            results.append(reduce_fn(logprobs[i, :Li]).detach().cpu())
    return torch.stack(results)


def run_single_gpu(fasta_path: str, output_dir: str, batch_size: int) -> None:
    """Run Evo2 embedding and log-likelihood computation on a single GPU."""
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    device = torch.device("cuda:0")
    os.makedirs(output_dir, exist_ok=True)

    # Load Evo2 model
    print("Loading Evo2 model...", flush=True)
    evo2_model = Evo2("evo2_7b")
    model, tokenizer = evo2_model.model, evo2_model.tokenizer
    model.to(device)
    model.eval()

    # Read FASTA data
    print(f"Reading FASTA file from {fasta_path}...", flush=True)
    sequences: List[str] = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append(str(record.seq))

    total_sequences = len(sequences)
    if total_sequences == 0:
        print("No sequences found in FASTA file, exiting...", flush=True)
        return
    print(f"Total sequences: {total_sequences}", flush=True)

    # Storage for embeddings and log-likelihoods
    all_embeddings: list[torch.Tensor] = []
    all_log_likelihoods: list[float] = []

    num_batches = math.ceil(total_sequences / batch_size)
    print(f"Processing {num_batches} batches with batch size {batch_size}", flush=True)

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_sequences)
        batch_sequences = sequences[batch_start:batch_end]

        if not batch_sequences:
            continue

        try:
            # Prepare batch
            input_ids, seq_lengths = prepare_batch(
                batch_sequences,
                tokenizer,
                prepend_bos=True,
                device=device,
            )

            with torch.no_grad():
                _, embeddings = evo2_model(
                    input_ids,
                    return_embeddings=True,
                    layer_names=[LAYER_NAME],
                )
                batch_embeddings = embeddings[LAYER_NAME]
                batch_log_likelihoods = compute_log_likelihood(
                    input_ids, model, seq_lengths
                )

            # Average over valid sequence length for each sequence
            batch_final_embeddings = [
                torch.mean(batch_embeddings[i, :seq_len], dim=0).cpu()
                for i, seq_len in enumerate(seq_lengths)
            ]

            # Store embeddings and log-likelihoods
            all_embeddings.extend(batch_final_embeddings)
            all_log_likelihoods.extend(batch_log_likelihoods.cpu().tolist())

        except torch.cuda.OutOfMemoryError:
            print(f"CUDA OOM in batch {batch_idx}, skipping batch", flush=True)
            torch.cuda.empty_cache()
            gc.collect()
            continue
        finally:
            torch.cuda.empty_cache()

    # Save results
    if not all_embeddings:
        print("No embeddings to save", flush=True)
        return

    print(f"Saving {len(all_embeddings)} embeddings and log-likelihoods...", flush=True)
    fasta_basename = os.path.splitext(os.path.basename(fasta_path))[0]

    # Save embeddings and log-likelihoods
    safetensors_path = os.path.join(output_dir, f"{fasta_basename}.safetensors")
    save_file(
        {
            "embeddings": torch.stack(all_embeddings),
            "log_likelihoods": torch.tensor(all_log_likelihoods, dtype=torch.float32),
        },
        safetensors_path,
    )

    print(f"Successfully saved {len(all_embeddings)} embeddings and log-likelihoods", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-GPU Evo2 embedding computation"
    )
    parser.add_argument(
        "--fasta_path",
        "-i",
        required=True,
        help="Path to input FASTA file",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        default="./embedding/alcantar_2025",
        help="Output directory for safetensor files (default: ./embedding/alcantar_2025)",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=8,
        help="Batch size for processing sequences (default: 8)",
    )
    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.fasta_path):
        parser.error(f"Input FASTA file not found: {args.fasta_path}")

    if not torch.cuda.is_available():
        parser.error("CUDA is not available. This script requires GPU support.")

    print(f"Processing file: {args.fasta_path}", flush=True)
    print(f"Output directory: {args.output_dir}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)

    run_single_gpu(args.fasta_path, args.output_dir, args.batch_size)


if __name__ == "__main__":
    main()
