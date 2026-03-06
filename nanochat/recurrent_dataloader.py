"""
Recurrent-document dataloaders for pretraining.

Goal:
- preserve in-document token order across multiple consecutive chunks
- cap number of chunks per document to avoid very long docs monopolizing training
- keep interface compatible with existing train loop (inputs, targets, state_dict)

Unlike bos_bestfit packer, this loader does not interleave many documents inside one row.
Each row comes from one contiguous chunk of one document.
"""

from collections import deque

import torch

from nanochat.dataloader import _document_batches


def _doc_chunks(tokens: list[int], row_capacity: int, max_chunks_per_doc: int) -> list[tuple[int, list[int]]]:
    """Split one tokenized doc into contiguous chunks of size row_capacity.

    Notes:
    - tokens are expected to already include BOS prefix from tokenizer.encode(..., prepend=bos)
    - only full rows are emitted (tail shorter than row_capacity is dropped)
    - at most max_chunks_per_doc chunks are returned
    """
    if max_chunks_per_doc <= 0:
        return []
    out = []
    start = 0
    for chunk_idx in range(max_chunks_per_doc):
        end = start + row_capacity
        if end > len(tokens):
            break
        out.append((chunk_idx, tokens[start:end]))
        start = end
    return out


def _document_batches_tinystories(split, resume_state_dict, tokenizer_batch_size):
    """Infinite iterator over TinyStories text batches.

    Uses HF datasets. Resume support is approximate (doc index skip).
    """
    try:
        from datasets import load_dataset
    except Exception as e:
        raise ImportError("TinyStories requires `datasets` package. Install with: pip install datasets") from e

    hf_split = "train" if split == "train" else "validation"
    ds = load_dataset("roneneldan/TinyStories", split=hf_split, streaming=True)

    resume_i = 0 if resume_state_dict is None else int(resume_state_dict.get("doc_index", 0))
    it = iter(ds)
    for _ in range(resume_i):
        try:
            next(it)
        except StopIteration:
            it = iter(ds)
            break

    batch = []
    doc_index = resume_i
    epoch = 1
    while True:
        try:
            ex = next(it)
        except StopIteration:
            it = iter(ds)
            epoch += 1
            ex = next(it)
        text = ex.get("text", "")
        if isinstance(text, str) and text:
            batch.append(text)
        doc_index += 1
        if len(batch) >= tokenizer_batch_size:
            yield batch, (0, doc_index, epoch)
            batch = []


def tokenizing_distributed_data_loader_with_state_recurrent_doc(
    tokenizer,
    B,
    T,
    split,
    *,
    tokenizer_threads=4,
    tokenizer_batch_size=128,
    device="cuda",
    resume_state_dict=None,
    max_chunks_per_doc=5,
    source_dataset="fineweb_edu",
):
    """Yield recurrent contiguous chunks from documents.

    Returns:
      inputs:  [B, T]
      targets: [B, T]
      state_dict with cursors and counters
    """
    assert split in ["train", "val"]
    assert max_chunks_per_doc >= 1
    if B != 1:
        raise ValueError(
            "recurrent_doc loader currently requires B=1. "
            "Per-row stream/state routing for B>1 is not implemented yet."
        )

    row_capacity = T + 1
    if source_dataset == "fineweb_edu":
        batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    elif source_dataset == "tinystories":
        batches = _document_batches_tinystories(split, resume_state_dict, tokenizer_batch_size)
    else:
        raise ValueError(f"Unknown source_dataset: {source_dataset}")
    bos_token = tokenizer.get_bos_token_id()

    chunk_queue = deque()
    pq_idx, rg_idx, epoch = 0, 0, 1
    docs_seen = 0

    use_cuda = device == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[: B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T :].view(B, T)
    inputs = gpu_buffer[: B * T].view(B, T)
    targets = gpu_buffer[B * T :].view(B, T)

    def refill_chunks():
        nonlocal pq_idx, rg_idx, epoch, docs_seen
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for toks in token_lists:
            docs_seen += 1
            for chunk_idx, chunk in _doc_chunks(toks, row_capacity=row_capacity, max_chunks_per_doc=max_chunks_per_doc):
                chunk_queue.append((chunk_idx, chunk))

    while True:
        chunk_idx_current = None
        for b in range(B):
            while not chunk_queue:
                refill_chunks()
            chunk_idx, row = chunk_queue.popleft()
            chunk_idx_current = chunk_idx
            row_buffer[b] = torch.tensor(row, dtype=torch.long)

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        state_dict = {
            "pq_idx": pq_idx,
            "rg_idx": rg_idx,
            "epoch": epoch,
            "docs_seen": docs_seen,
            "queue_depth": len(chunk_queue),
            "loader_mode": "recurrent_doc",
            "max_chunks_per_doc": max_chunks_per_doc,
            "chunk_idx_in_doc": int(chunk_idx_current) if chunk_idx_current is not None else -1,
        }

        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict


def tokenizing_distributed_data_loader_recurrent_doc(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, _state in tokenizing_distributed_data_loader_with_state_recurrent_doc(*args, **kwargs):
        yield inputs, targets
