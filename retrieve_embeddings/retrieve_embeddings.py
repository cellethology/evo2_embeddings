import argparse
import logging
import os
from typing import List, Tuple

import numpy as np
import torch
from evo2.scoring import prepare_batch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from evo2 import Evo2

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

LAYER_NAME = "blocks.28.mlp.l3"


class SequenceDataset(Dataset):
    def __init__(self, sequences: List[str], ids: List[str]):
        self.sequences = sequences
        self.ids = ids

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.ids[idx]


def _collate(batch):
    seqs, ids = zip(*batch)
    return list(seqs), list(ids)


def prepare_model(model_name: str, device: str) -> Evo2:
    """Load Evo2, move the underlying model to the device, and enable eval mode."""
    logger.info(f"Loading Evo2 model: {model_name} on device {device}...")
    model = Evo2(model_name=model_name)

    base_model = getattr(model, "model", None)
    if base_model is None:
        raise AttributeError("Evo2 instance does not expose the underlying model")

    base_model.to(device)
    base_model.eval()
    logger.info("Model loaded and ready for inference")
    return model


def extract_embeddings_batch(
    model: Evo2,
    sequences: List[str],
    layer_name: str,
    device: str = "cuda:0",
    prepend_bos: bool = False,
    final_token_only: bool = False,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Extract embeddings for a batch of sequences.

    Args:
        model: Evo2 model instance
        sequences: List of DNA sequences to process
        layer_name: Name of the layer to extract embeddings from
        device: Device to run inference on (default: "cuda:0")
        prepend_bos: Whether to prepend BOS token (default: False)
        final_token_only: Whether to return only the final token embedding (default: False)
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

    # Extract embeddings
    _, embeddings_dict = model(
        input_ids=input_ids,
        return_embeddings=True,
        layer_names=[layer_name],
    )

    embeddings = embeddings_dict[layer_name]
    if final_token_only:
        seq_len_idx = torch.tensor(seq_lengths, device=embeddings.device)
        if prepend_bos:
            seq_len_idx += 1  # last real token is shifted by the BOS token
        embeddings = embeddings[:, seq_len_idx - 1, :]

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
    final_token_only: bool = False,
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
        final_token_only: If True, return only the final token embedding (default: False)
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
    if final_token_only:
        logger.info(
            "Final token only enabled: embeddings will be returned for the final token only"
        )

    # Lists to collect all embeddings and IDs
    all_ids: List[str] = []
    all_embeddings: List[np.ndarray] = []

    dataset = SequenceDataset(sequences, sequence_ids)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate
    )

    failed_ids: List[str] = []

    with torch.inference_mode():
        for batch_sequences, batch_ids in tqdm(
            dataloader, desc="Extracting embeddings", unit="batch"
        ):
            try:
                # Extract embeddings for the batch
                batch_embeddings, batch_seq_lengths = extract_embeddings_batch(
                    model=model,
                    sequences=batch_sequences,
                    layer_name=layer_name,
                    device=device,
                    prepend_bos=prepend_bos,
                    final_token_only=final_token_only,
                )

                if final_token_only:
                    batch_embeddings_np = batch_embeddings.cpu().float().numpy()
                else:
                    seq_lens = torch.tensor(
                        batch_seq_lengths, device=batch_embeddings.device
                    )
                    bos_offset = 1 if prepend_bos else 0
                    max_len = int(seq_lens.max().item()) if seq_lens.numel() > 0 else 0
                    if max_len == 0:
                        batch_embeddings_np = np.zeros(
                            (len(batch_seq_lengths), batch_embeddings.shape[-1]),
                            dtype=np.float32,
                        )
                    else:
                        token_embeddings = batch_embeddings[
                            :, bos_offset : bos_offset + max_len, :
                        ]
                        mask = torch.arange(
                            max_len, device=batch_embeddings.device
                        ).unsqueeze(0) < seq_lens.unsqueeze(1)
                        pooled = (token_embeddings * mask.unsqueeze(-1)).sum(
                            dim=1
                        ) / mask.sum(dim=1, keepdim=True).clamp_min(1)
                        batch_embeddings_np = pooled.cpu().float().numpy()

                all_embeddings.extend(batch_embeddings_np)
                all_ids.extend(batch_ids)

            except Exception as e:
                logger.error(
                    f"Error processing batch containing IDs {batch_ids}: {str(e)}",
                    exc_info=True,
                )
                failed_ids.extend(batch_ids)
                continue

    if failed_ids:
        raise RuntimeError(
            "Failed to process the following sequence IDs: "
            f"{', '.join(failed_ids[:10])}"
            f"{' ...' if len(failed_ids) > 10 else ''}"
        )

    if len(all_embeddings) != len(sequence_ids):
        raise ValueError(
            "Mismatch between embeddings and sequence IDs: "
            f"{len(all_embeddings)} embeddings vs {len(sequence_ids)} ids"
        )

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
        "--final_token_only",
        action="store_true",
        help="Return only the final token embedding (default: False)",
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
        model = prepare_model(args.model_name, args.device)
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
            final_token_only=args.final_token_only,
        )

        logger.info("Embedding extraction completed successfully")

    except Exception as e:
        logger.error(f"Error during embedding extraction: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
