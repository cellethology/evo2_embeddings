import logging
import os
from typing import List, Tuple

import numpy as np
from Bio import SeqIO

# Configure logging
logger = logging.getLogger(__name__)

# Valid DNA/RNA characters (including IUPAC ambiguity codes)
VALID_DNA_CHARS = set("ATCG")
# Maximum sequence length to prevent OOM (adjust based on model context length)
MAX_SEQUENCE_LENGTH = 1_000_000  # 1M context for evo2_7b
MIN_SEQUENCE_LENGTH = 1


def validate_sequence(sequence: str, seq_id: str = "") -> Tuple[bool, str]:
    """
    Validate a sequence for length and invalid characters.

    Args:
        sequence: DNA/RNA sequence to validate
        seq_id: Optional sequence identifier for error messages

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not sequence:
        return False, f"Empty sequence found{': ' + seq_id if seq_id else ''}"

    seq_len = len(sequence)
    if seq_len < MIN_SEQUENCE_LENGTH:
        return False, (
            f"Sequence too short (length {seq_len}, minimum {MIN_SEQUENCE_LENGTH})"
            f"{': ' + seq_id if seq_id else ''}"
        )

    if seq_len > MAX_SEQUENCE_LENGTH:
        return False, (
            f"Sequence too long (length {seq_len}, maximum {MAX_SEQUENCE_LENGTH})"
            f"{': ' + seq_id if seq_id else ''}"
        )

    # Check for invalid characters
    sequence_upper = sequence.upper()
    invalid_chars = set(sequence_upper) - VALID_DNA_CHARS
    if invalid_chars:
        return False, (
            f"Invalid characters found: {sorted(invalid_chars)}"
            f"{' in sequence: ' + seq_id if seq_id else ''}"
        )

    return True, ""


def load_sequences_from_fasta(
    fasta_path: str,
) -> Tuple[List[str], List[str]]:
    """
    Load sequences from a FASTA file using SeqIO.

    Args:
        fasta_path: Path to the input FASTA file

    Returns:
        Tuple of (sequences, sequence_ids) where sequences are validated DNA sequences
        and sequence_ids are their corresponding identifiers

    Raises:
        FileNotFoundError: If the FASTA file does not exist
        ValueError: If no valid sequences are found
    """
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    sequences: List[str] = []
    sequence_ids: List[str] = []
    invalid_count = 0

    logger.info(f"Loading sequences from {fasta_path}...")

    with open(fasta_path, "r", encoding="utf-8") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq_id = record.id
            sequence = str(record.seq).upper()

            # Validate sequence
            is_valid, error_msg = validate_sequence(sequence, seq_id)
            if not is_valid:
                logger.warning(f"Skipping invalid sequence {seq_id}: {error_msg}")
                invalid_count += 1
                continue

            sequences.append(sequence)
            sequence_ids.append(seq_id)

    if not sequences:
        raise ValueError(
            f"No valid sequences found in {fasta_path}. "
            f"Total invalid sequences: {invalid_count}"
        )

    logger.info(
        f"Loaded {len(sequences)} valid sequences "
        f"(skipped {invalid_count} invalid sequences)"
    )

    return sequences, sequence_ids


def save_embeddings_to_npz(
    ids: List[str],
    embeddings: List[np.ndarray],
    output_path: str,
) -> None:
    """
    Save embeddings to a compressed numpy .npz file.

    Args:
        ids: List of sequence identifiers
    embeddings: List of embedding arrays (one per sequence)
        output_path: Path to save the .npz file

    Raises:
        OSError: If the output file cannot be written to
        ValueError: If ids and embeddings have different lengths
    """
    if len(ids) != len(embeddings):
        raise ValueError(
            f"ids ({len(ids)}) and embeddings ({len(embeddings)}) "
            "must have the same length"
        )

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Convert ids to numpy array of strings
    ids_array = np.array(ids, dtype=str)

    # For embeddings, since sequences may have different lengths,
    # we'll save as a list/array of arrays. numpy.savez_compressed can handle this.
    # Convert list of arrays to object array for variable-length sequences
    embeddings_array = np.array(embeddings, dtype=object)

    # Save as compressed numpy file
    np.savez_compressed(output_path, ids=ids_array, embeddings=embeddings_array)

    logger.info(f"Saved {len(ids)} embeddings to {output_path}")
