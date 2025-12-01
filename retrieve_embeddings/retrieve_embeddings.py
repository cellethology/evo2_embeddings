import argparse
import logging
import os
from typing import List, Tuple

import numpy as np
import torch
from evo2 import Evo2
from evo2.scoring import prepare_batch
from tqdm import tqdm

from .util import (
    load_sequences_from_fasta,
    save_embeddings_to_npz,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

LAYER_NAME = "blocks.31.mlp.l3"


def extract_embeddings_batch(
    model: Evo2,
    sequences: List[str],
    layer_name: str,
    device: str = "cuda:0",
    prepend_bos: bool = False,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Extract embeddings for a batch of sequences.

    Args:
        model: Evo2 model instance
        sequences: List of DNA sequences to process
        layer_name: Name of the layer to extract embeddings from
        device: Device to run inference on (default: "cuda:0")
        prepend_bos: Whether to prepend BOS token (default: False)

    Returns:
        Tuple of (embeddings tensor, sequence_lengths list)
        Embeddings have shape [batch_size, max_seq_length + (1 if prepend_bos else 0), embedding_dim]
        sequence_lengths are the original sequence lengths (without BOS or padding)
    """
    # Prepare batch: tokenize and pad sequences
    input_ids, seq_lengths = prepare_batch(
        seqs=sequences,
        tokenizer=model.tokenizer,
        prepend_bos=prepend_bos,
        device=device,
    )

    # Extract embeddings using forward pass
    with torch.no_grad():
        _, embeddings_dict = model.forward(
            input_ids=input_ids,
            return_embeddings=True,
            layer_names=[layer_name],
        )

    embeddings = embeddings_dict[layer_name]

    return embeddings, seq_lengths


def process_sequences(
    model: Evo2,
    sequences: List[str],
    sequence_ids: List[str],
    output_path: str,
    batch_size: int = 8,
    layer_name: str = LAYER_NAME,
    device: str = "cuda:0",
    prepend_bos: bool = True,
    mean_pooling: bool = False,
) -> None:
    """
    Process sequences in batches and extract embeddings.

    Args:
        model: Evo2 model instance
        sequences: List of DNA sequences to process
        sequence_ids: List of sequence identifiers corresponding to sequences
        output_path: Path to save the .npz file with all embeddings
        batch_size: Number of sequences to process per batch (default: 8)
        layer_name: Name of the layer to extract embeddings from (default: LAYER_NAME)
        device: Device to run inference on (default: "cuda:0")
        prepend_bos: Whether to prepend BOS token (default: True)
        mean_pooling: If True, average embeddings across sequence length to get a single
            vector per sequence. Reduces memory usage significantly (default: False)

    Raises:
        ValueError: If sequences and sequence_ids have different lengths
    """
    if len(sequences) != len(sequence_ids):
        raise ValueError(
            f"Sequences ({len(sequences)}) and sequence_ids ({len(sequence_ids)}) "
            "must have the same length"
        )

    logger.info(
        f"Processing {len(sequences)} sequences in batches of {batch_size} "
        f"on device {device}"
    )
    if mean_pooling:
        logger.info("Mean pooling enabled: embeddings will be averaged across sequence length")

    # Lists to collect all embeddings and IDs
    all_ids: List[str] = []
    all_embeddings: List[np.ndarray] = []

    # Process sequences in batches
    num_batches = (len(sequences) + batch_size - 1) // batch_size

    with tqdm(total=len(sequences), desc="Extracting embeddings") as pbar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(sequences))

            batch_sequences = sequences[start_idx:end_idx]
            batch_ids = sequence_ids[start_idx:end_idx]

            try:
                # Extract embeddings for the batch
                batch_embeddings, batch_seq_lengths = extract_embeddings_batch(
                    model=model,
                    sequences=batch_sequences,
                    layer_name=layer_name,
                    device=device,
                    prepend_bos=prepend_bos,
                )

                # Extract embeddings for each sequence in the batch
                for i, seq_id in enumerate(batch_ids):
                    # Extract embeddings for this specific sequence
                    # Note: embeddings are padded, so we need to extract the actual sequence length
                    seq_length = batch_seq_lengths[i]
                    if prepend_bos:
                        # Skip BOS token if prepended (first position)
                        # Take seq_length tokens after BOS
                        seq_embeddings = batch_embeddings[i, 1 : seq_length + 1]
                    else:
                        # Take seq_length tokens from the start
                        seq_embeddings = batch_embeddings[i, :seq_length]

                    # Apply mean pooling if requested (reduces memory usage)
                    if mean_pooling:
                        # Average across sequence length dimension: (seq_length, embedding_dim) -> (embedding_dim,)
                        seq_embeddings = torch.mean(seq_embeddings, dim=0)

                    # Convert to numpy and store
                    # Convert to float32 first to avoid BFloat16 issues with numpy
                    all_ids.append(seq_id)
                    all_embeddings.append(seq_embeddings.cpu().float().numpy())

                pbar.update(len(batch_sequences))

            except Exception as e:
                logger.error(
                    f"Error processing batch {batch_idx + 1}/{num_batches}: {str(e)}"
                )
                # Continue with next batch
                pbar.update(len(batch_sequences))
                continue

    # Save all embeddings to a single .npz file
    logger.info(f"Saving {len(all_ids)} embeddings to {output_path}...")
    save_embeddings_to_npz(
        ids=all_ids,
        embeddings=all_embeddings,
        output_path=output_path,
    )

    logger.info(f"Completed processing. Embeddings saved to {output_path}")


def main() -> None:
    """
    Main function to orchestrate sequence loading, embedding extraction, and saving.
    """
    parser = argparse.ArgumentParser(
        description="Single-GPU Evo2 embedding computation from FASTA files"
    )
    parser.add_argument(
        "--fasta_path",
        "-i",
        type=str,
        required=True,
        help="Path to input FASTA file",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        default="./embeddings.npz",
        help="Output path for .npz file with embeddings (default: ./embeddings.npz)",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=8,
        help="Batch size for processing sequences (default: 8)",
    )
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        default="evo2_7b",
        help="Evo2 model name to use (default: evo2_7b)",
    )
    parser.add_argument(
        "--layer_name",
        "-l",
        type=str,
        default=LAYER_NAME,
        help=f"Layer name to extract embeddings from (default: {LAYER_NAME})",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda:0",
        help="Device to run inference on (default: cuda:0)",
    )
    parser.add_argument(
        "--prepend_bos",
        action="store_true",
        help="Prepend BOS token to sequences (default: False)",
    )
    parser.add_argument(
        "--mean_pooling",
        action="store_true",
        help="Apply mean pooling to embeddings across sequence length. "
        "Reduces memory usage by averaging per-token embeddings into a single vector per sequence (default: False)",
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.fasta_path):
        parser.error(f"Input FASTA file not found: {args.fasta_path}")

    # Validate CUDA availability if using GPU
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error(
            f"CUDA is not available but device {args.device} was specified. "
            "Please use 'cpu' or ensure CUDA is properly configured."
        )

    # Validate batch size
    if args.batch_size < 1:
        parser.error(f"Batch size must be at least 1, got {args.batch_size}")

    logger.info(f"Processing file: {args.fasta_path}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Layer: {args.layer_name}")
    logger.info(f"Device: {args.device}")

    try:
        # Load sequences from FASTA file
        sequences, sequence_ids = load_sequences_from_fasta(args.fasta_path)

        # Initialize model
        logger.info(f"Loading Evo2 model: {args.model_name}...")
        model = Evo2(model_name=args.model_name)
        logger.info("Model loaded successfully")

        # Process sequences and extract embeddings
        process_sequences(
            model=model,
            sequences=sequences,
            sequence_ids=sequence_ids,
            output_path=args.output_path,
            batch_size=args.batch_size,
            layer_name=args.layer_name,
            device=args.device,
            prepend_bos=args.prepend_bos,
            mean_pooling=args.mean_pooling,
        )

        logger.info("Embedding extraction completed successfully")

    except Exception as e:
        logger.error(f"Error during embedding extraction: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
