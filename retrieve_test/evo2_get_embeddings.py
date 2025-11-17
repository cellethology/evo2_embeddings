import argparse
import gc
import math
import os
import signal
import sys
import traceback
from typing import List

import numpy as np
import pandas as pd
import psutil
import torch
from evo2 import Evo2
from evo2.scoring import logits_to_logprobs, prepare_batch
from safetensors.torch import save_file
from tqdm.auto import tqdm

# apptainer exec --nv --bind $PWD:/app --bind ./models:/root/.cache/huggingface --pwd /app \
#   ./evo2.sif python3 scripts/alcantar_2025/evo_2_embed_with_csv_alcantar_single_gpu.py \
#   --csv_path embeddings/alcantar_2025/pre_embeddings/processed_chromatin_sequences.csv \
#   -o embeddings/alcantar_2025/post_embeddings --batch_size 4

LAYER_NAME = "blocks.31.mlp.l3"


def get_memory_info():
    """Get current memory usage information."""
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        gpu_memory = (
            torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        )
        gpu_allocated = torch.cuda.memory_allocated() / 1024**3
        gpu_cached = torch.cuda.memory_reserved() / 1024**3
    else:
        gpu_memory = gpu_allocated = gpu_cached = 0.0

    ram_info = psutil.virtual_memory()
    ram_total = ram_info.total / 1024**3
    ram_used = ram_info.used / 1024**3

    return {
        "gpu_total": gpu_memory,
        "gpu_allocated": gpu_allocated,
        "gpu_cached": gpu_cached,
        "ram_total": ram_total,
        "ram_used": ram_used,
    }


def signal_handler(signum, frame):
    """Handle signals gracefully."""
    print(f"Received signal {signum}, cleaning up...", flush=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sys.exit(0)


def compute_log_likelihood(
    input_ids: torch.Tensor,
    model,
    seq_lengths: List[int] | None = None,
    reduce_method: str = "mean",
    prepend_bos: bool = True,
) -> torch.Tensor:
    """
    Compute per-sequence log-likelihoods robustly.

    Assumes logits_to_logprobs returns per-token logprobs aligned to input_ids
    (usually shifted by one).
    """
    try:
        with torch.inference_mode():
            outputs = model(input_ids)
            if hasattr(outputs, "logits"):
                logits = outputs.logits  # [B, T, V]
            else:
                logits = outputs[0]  # [B, T, V]

        logprobs = logits_to_logprobs(logits, input_ids)
        if isinstance(logprobs, np.ndarray):
            logprobs = torch.from_numpy(logprobs)
        logprobs = logprobs.float()  # [B, T_eff]

        B, T_eff = logprobs.shape[:2]

        if seq_lengths is None:
            raise ValueError("seq_lengths is None; please pass seq_lengths from prepare_batch")

        if len(seq_lengths) != B:
            raise ValueError(f"seq_lengths size {len(seq_lengths)} != batch size {B}")

        # Adjust lengths if BOS was prepended
        adj_lengths: list[int] = []
        for L in seq_lengths:
            eff = L - 1 if prepend_bos and L > 0 else L
            eff = max(0, min(eff, T_eff))
            adj_lengths.append(eff)

        if reduce_method == "mean":
            reduce_fn = torch.mean
        elif reduce_method == "sum":
            reduce_fn = torch.sum
        else:
            raise ValueError(f"Invalid reduce_method {reduce_method}")

        out: list[torch.Tensor] = []
        for i in range(B):
            Li = adj_lengths[i]
            if Li == 0:
                out.append(torch.tensor(0.0))
            else:
                out.append(reduce_fn(logprobs[i, :Li]).detach().cpu())
        return torch.stack(out)

    except Exception as e:  # noqa: BLE001
        try:
            shapes = {"input_ids": tuple(input_ids.shape)}
        except Exception:  # noqa: BLE001
            shapes = {}
        print(f"Error in compute_log_likelihood: {e}. Shapes: {shapes}", flush=True)
        return torch.zeros(
            len(seq_lengths) if seq_lengths is not None else input_ids.size(0)
        )


def run_single_gpu(csv_path: str, output_dir: str, batch_size: int) -> None:
    """Run Evo2 embedding and log-likelihood computation on a single GPU."""
    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    device = torch.device("cuda:0")

    # Log initial memory state
    mem_info = get_memory_info()
    print(
        (
            f"Initial memory - GPU: {mem_info['gpu_allocated']:.2f}GB/"
            f"{mem_info['gpu_total']:.2f}GB, "
            f"RAM: {mem_info['ram_used']:.2f}GB/{mem_info['ram_total']:.2f}GB"
        ),
        flush=True,
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load Evo2 model
    print("Loading Evo2 model...", flush=True)
    try:
        evo2_model = Evo2("evo2_7b")
        model, tokenizer = evo2_model.model, evo2_model.tokenizer
        model.to(device)
        model.eval()

        mem_info = get_memory_info()
        print(
            (
                f"Memory after model load - GPU: {mem_info['gpu_allocated']:.2f}GB/"
                f"{mem_info['gpu_total']:.2f}GB"
            ),
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"Failed to load model: {e}", flush=True)
        traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    # Read CSV data
    print(f"Reading CSV data from {csv_path} ...", flush=True)
    try:
        df = pd.read_csv(csv_path)
        total_rows = len(df)
        if total_rows == 0:
            print("No data found in CSV, exiting...", flush=True)
            return
        print(f"Total rows: {total_rows}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"Failed to read CSV: {e}", flush=True)
        traceback.print_exc()
        return

    # Storage for all data
    all_original_combos: list[str] = []
    all_sequences: list[str] = []
    all_measured_fluorescence: list[float] = []
    all_predicted_fluorescence: list[float] = []
    all_embeddings: list[torch.Tensor] = []
    all_log_likelihoods: list[float] = []

    # Process data in smaller batches to manage memory
    effective_batch_size = min(batch_size, 8)  # Cap at 8 to prevent OOM
    num_batches = math.ceil(total_rows / effective_batch_size)
    print(
        f"Processing {num_batches} batches with batch size {effective_batch_size}",
        flush=True,
    )

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        try:
            batch_start = batch_idx * effective_batch_size
            batch_end = min(batch_start + effective_batch_size, total_rows)
            batch_df = df.iloc[batch_start:batch_end]

            batch_original_combos = batch_df["original_combo"].tolist()
            batch_sequences = batch_df["final_sequence"].tolist()
            batch_measured_fluorescence = batch_df["measured_fluorescence"].tolist()
            batch_predicted_fluorescence = batch_df["predicted_fluorescence"].tolist()

            if not batch_sequences:
                continue

            # Periodic memory check
            if batch_idx % 10 == 0:
                mem_info = get_memory_info()
                if mem_info["gpu_allocated"] > mem_info["gpu_total"] * 0.85:
                    print(
                        "High GPU memory usage detected, forcing cleanup",
                        flush=True,
                    )
                    torch.cuda.empty_cache()
                    gc.collect()

            # Prepare batch
            input_ids, seq_lengths = prepare_batch(
                batch_sequences,
                tokenizer,
                prepend_bos=True,
                device=device,
            )

            try:
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
                batch_final_embeddings: list[torch.Tensor] = []
                for i, seq_len in enumerate(seq_lengths):
                    valid_length = seq_len
                    seq_embedding = torch.mean(
                        batch_embeddings[i, :valid_length], dim=0
                    )
                    batch_final_embeddings.append(seq_embedding.cpu())

                # Store results
                all_original_combos.extend(batch_original_combos)
                all_sequences.extend(batch_sequences)
                all_measured_fluorescence.extend(batch_measured_fluorescence)
                all_predicted_fluorescence.extend(batch_predicted_fluorescence)
                all_embeddings.extend(batch_final_embeddings)
                all_log_likelihoods.extend(batch_log_likelihoods.cpu().tolist())

            except torch.cuda.OutOfMemoryError:
                print(
                    f"CUDA OOM in batch {batch_idx}, skipping batch",
                    flush=True,
                )
                torch.cuda.empty_cache()
                gc.collect()
                continue
            except Exception as e:  # noqa: BLE001
                print(f"Error processing batch {batch_idx}: {e}", flush=True)
                traceback.print_exc()
                continue
            finally:
                if "input_ids" in locals():
                    del input_ids
                if "batch_embeddings" in locals():
                    del batch_embeddings
                if "batch_log_likelihoods" in locals():
                    del batch_log_likelihoods
                torch.cuda.empty_cache()

        except Exception as e:  # noqa: BLE001
            print(f"Fatal error in batch {batch_idx}: {e}", flush=True)
            traceback.print_exc()
            continue

    # Save results
    if not all_embeddings:
        print("No embeddings to save", flush=True)
    else:
        print(f"Saving {len(all_original_combos)} sequences...", flush=True)
        try:
            final_embeddings = torch.stack(all_embeddings)
            final_log_likelihoods = torch.tensor(
                all_log_likelihoods,
                dtype=torch.float32,
            )
            final_measured_fluorescence = torch.tensor(
                all_measured_fluorescence,
                dtype=torch.float32,
            )
            final_predicted_fluorescence = torch.tensor(
                all_predicted_fluorescence,
                dtype=torch.float32,
            )

            # Encode original combos as bytes
            original_combos_str = "|".join(all_original_combos)
            original_combos_bytes = original_combos_str.encode("utf-8")
            original_combos_tensor = torch.tensor(
                list(original_combos_bytes),
                dtype=torch.uint8,
            )

            csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
            safetensors_path = os.path.join(
                output_dir,
                f"{csv_basename}.safetensors",
            )

            save_data: dict[str, torch.Tensor] = {
                "embeddings": final_embeddings,
                "original_combos_encoded": original_combos_tensor,
                "measured_fluorescence": final_measured_fluorescence,
                "predicted_fluorescence": final_predicted_fluorescence,
                "log_likelihoods": final_log_likelihoods,
            }

            # Encode sequences as bytes
            sequences_str = "|".join(all_sequences)
            sequences_bytes = sequences_str.encode("utf-8")
            save_data["sequences_encoded"] = torch.tensor(
                list(sequences_bytes),
                dtype=torch.uint8,
            )

            save_file(save_data, safetensors_path)

            # Also save CSV with metadata
            csv_output_path = os.path.join(
                output_dir,
                f"{csv_basename}.csv",
            )
            results_df = pd.DataFrame(
                {
                    "Original_Combo": all_original_combos,
                    "Sequence": all_sequences,
                    "Measured_Fluorescence": all_measured_fluorescence,
                    "Predicted_Fluorescence": all_predicted_fluorescence,
                    "Log_Likelihood": all_log_likelihoods,
                }
            )
            results_df.to_csv(csv_output_path, index=False)

            print(
                f"Successfully saved {len(all_original_combos)} sequences",
                flush=True,
            )

        except Exception as e:  # noqa: BLE001
            print(f"Error saving results: {e}", flush=True)
            traceback.print_exc()

    # Cleanup
    try:
        del model, tokenizer, evo2_model
    except Exception:  # noqa: BLE001
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("Done.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Single-GPU Evo2 embedding with log likelihood computation "
            "for Alcantar dataset"
        )
    )
    parser.add_argument(
        "--csv_path",
        "-i",
        required=True,
        help="Path to input CSV file (processed_chromatin_sequences.csv)",
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
    if not os.path.exists(args.csv_path):
        parser.error(f"Input CSV file not found: {args.csv_path}")

    if not torch.cuda.is_available():
        parser.error("CUDA is not available. This script requires GPU support.")

    print(f"Processing file: {args.csv_path}", flush=True)
    print(f"Output directory: {args.output_dir}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)
    print(
        "Will compute: Original_Combo, sequences, "
        "measured/predicted fluorescence, embeddings, log_likelihood",
        flush=True,
    )

    run_single_gpu(args.csv_path, args.output_dir, args.batch_size)


if __name__ == "__main__":
    main()
