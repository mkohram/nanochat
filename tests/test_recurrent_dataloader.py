import torch

from nanochat.recurrent_dataloader import _doc_chunks, tokenizing_distributed_data_loader_with_state_recurrent_doc


class FakeTokenizer:
    def __init__(self, bos=1):
        self._bos = bos

    def get_bos_token_id(self):
        return self._bos

    def encode(self, docs, prepend, num_threads=1):
        out = []
        for d in docs:
            # docs are already lists of ints in these tests
            out.append([prepend] + list(d))
        return out


def test_doc_chunks_preserve_order_and_size():
    tokens = list(range(1, 1 + 30))
    chunks = _doc_chunks(tokens, row_capacity=10, max_chunks_per_doc=5)
    assert len(chunks) == 3
    assert chunks[0] == (0, list(range(1, 11)))
    assert chunks[1] == (1, list(range(11, 21)))
    assert chunks[2] == (2, list(range(21, 31)))


def test_doc_chunks_respect_max_chunks_per_doc():
    tokens = list(range(100))
    chunks = _doc_chunks(tokens, row_capacity=10, max_chunks_per_doc=2)
    assert len(chunks) == 2
    assert chunks[0] == (0, list(range(10)))
    assert chunks[1] == (1, list(range(10, 20)))


def test_doc_chunks_drop_tail_shorter_than_row_capacity():
    tokens = list(range(25))
    chunks = _doc_chunks(tokens, row_capacity=10, max_chunks_per_doc=5)
    assert len(chunks) == 2
    assert chunks[-1] == (1, list(range(10, 20)))


def test_recurrent_loader_shapes_and_state(monkeypatch):
    # create deterministic fake document stream: each doc is a list[int]
    docs = [list(range(40)), list(range(50, 90)), list(range(100, 140))]

    def fake_document_batches(split, resume_state_dict, tokenizer_batch_size):
        assert split in ["train", "val"]
        i = 0
        while True:
            # yield one doc per batch to make ordering easy to reason about
            doc = docs[i % len(docs)]
            yield [doc], (0, i, 1)
            i += 1

    import nanochat.recurrent_dataloader as rd

    monkeypatch.setattr(rd, "_document_batches", fake_document_batches)

    tok = FakeTokenizer(bos=7)
    loader = tokenizing_distributed_data_loader_with_state_recurrent_doc(
        tok,
        B=1,
        T=8,
        split="train",
        device="cpu",
        max_chunks_per_doc=2,
    )

    x, y, st = next(loader)

    # shapes
    assert x.shape == (1, 8)
    assert y.shape == (1, 8)

    # each row was built from row_capacity=T+1=9 tokens, so shift relation must hold
    assert torch.equal(x[0, 1:], y[0, :-1])

    # state bookkeeping
    assert st["loader_mode"] == "recurrent_doc"
    assert st["max_chunks_per_doc"] == 2
    assert st["docs_seen"] >= 1
    assert st["chunk_idx_in_doc"] in [0, 1]


def test_recurrent_loader_rejects_batch_size_gt_one(monkeypatch):
    docs = [list(range(40))]

    def fake_document_batches(split, resume_state_dict, tokenizer_batch_size):
        while True:
            yield [docs[0]], (0, 0, 1)

    import nanochat.recurrent_dataloader as rd

    monkeypatch.setattr(rd, "_document_batches", fake_document_batches)

    tok = FakeTokenizer(bos=1)
    loader = tokenizing_distributed_data_loader_with_state_recurrent_doc(
        tok,
        B=2,
        T=8,
        split="train",
        device="cpu",
        max_chunks_per_doc=2,
    )
    try:
        _ = next(loader)
        assert False, "Expected ValueError for B>1"
    except ValueError as e:
        assert "requires B=1" in str(e)


def test_recurrent_loader_chunk_idx_progresses_within_doc(monkeypatch):
    docs = [list(range(3000, 3060))]

    def fake_document_batches(split, resume_state_dict, tokenizer_batch_size):
        while True:
            yield [docs[0]], (0, 0, 1)

    import nanochat.recurrent_dataloader as rd
    monkeypatch.setattr(rd, "_document_batches", fake_document_batches)

    tok = FakeTokenizer(bos=1)
    loader = tokenizing_distributed_data_loader_with_state_recurrent_doc(
        tok, B=1, T=8, split="train", device="cpu", max_chunks_per_doc=3
    )

    _, _, st1 = next(loader)
    _, _, st2 = next(loader)
    _, _, st3 = next(loader)

    assert st1["chunk_idx_in_doc"] == 0
    assert st2["chunk_idx_in_doc"] == 1
    assert st3["chunk_idx_in_doc"] == 2


def test_recurrent_loader_uses_tinystories_source(monkeypatch):
    def fake_tiny_batches(split, resume_state_dict, tokenizer_batch_size):
        i = 0
        while True:
            # text docs; tokenizer in tests accepts list[int], so override encode below
            yield [[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]], (0, i, 1)
            i += 1

    import nanochat.recurrent_dataloader as rd

    monkeypatch.setattr(rd, "_document_batches_tinystories", fake_tiny_batches)

    tok = FakeTokenizer(bos=1)
    loader = tokenizing_distributed_data_loader_with_state_recurrent_doc(
        tok, B=1, T=8, split="train", device="cpu", max_chunks_per_doc=2, source_dataset="tinystories"
    )
    x, y, st = next(loader)
    assert x.shape == (1, 8)
    assert st["chunk_idx_in_doc"] in [0, 1]


def test_recurrent_loader_multiple_chunks_same_doc_before_next(monkeypatch):
    # One long doc, then a distinct doc, to verify chunk ordering from same doc is consecutive.
    docs = [list(range(1000, 1050)), list(range(2000, 2050))]

    def fake_document_batches(split, resume_state_dict, tokenizer_batch_size):
        i = 0
        while True:
            yield [docs[i % len(docs)]], (0, i, 1)
            i += 1

    import nanochat.recurrent_dataloader as rd

    monkeypatch.setattr(rd, "_document_batches", fake_document_batches)

    tok = FakeTokenizer(bos=1)
    loader = tokenizing_distributed_data_loader_with_state_recurrent_doc(
        tok,
        B=1,
        T=8,
        split="train",
        device="cpu",
        max_chunks_per_doc=3,
    )

    x1, y1, _ = next(loader)
    x2, y2, _ = next(loader)

    # reconstruct first token of each emitted row (after BOS shift at x[0,0])
    # x row starts with BOS, so y[0, -1] is still inside same chunk; use x[0,1] as first real token.
    first_real_tok_1 = int(x1[0, 1].item())
    first_real_tok_2 = int(x2[0, 1].item())

    # second batch should still come from the same first document because max_chunks_per_doc=3.
    assert 1000 <= first_real_tok_1 < 1050
    assert 1000 <= first_real_tok_2 < 1050
