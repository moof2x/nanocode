from functools import partial

import jax
import numpy as np

from nanojax.dataset import parquets_iter_batched


def tokenizing_data_loader(B, T, split, tokenizer, tokenizer_threads=4, tokenizer_batch_size=128, buffer_size=1000):
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    bos_token = tokenizer.get_bos_token_id()
    # document buffer holds tokenized documents for best-fit packing
    doc_buffer = []
    B *= jax.local_device_count() # per-device batch size to process batch size
    row_capacity = T + 1 # +1 is because we also need the target at the last token
    world_size, rank = jax.process_count(), jax.process_index()

    # infinite iterator over document batches
    def document_batches():
        while True:
            # batch will iterate in group size of the parquet files, usually e.g. 1024 rows
            for batch in parquets_iter_batched(split=split, start=rank, step=world_size):
                # for the tokenizer we might want to go in usually smaller batches, e.g. 128 rows
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size]
    batches = document_batches()

    while True:
        # Accumulate enough documents in buffer before packing
        while len(doc_buffer) < buffer_size:
            doc_batch = next(batches)
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
            for tokens in token_lists:
                doc_buffer.append(tokens)

        # Pack B rows using best-fit algorithm
        rows = []
        for _ in range(B):
            row = []
            while len(row) < row_capacity:
                remaining = row_capacity - len(row)
                # best-fit: find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len
                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row.extend(doc)
                else:
                    # no doc fits - crop shortest to fill remaining
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row.extend(doc[:remaining])
            rows.append(row[:row_capacity])

        # note: JAX does not natively support int64 (see JAX gotchas), but this isn't really an issue
        # as torch's cross entropy requires int64 targets for only historical(?) reasons
        row_data = np.array(rows, dtype=np.int32)
        # Create the inputs/targets and yield
        inputs = row_data[:, :-1]
        targets = row_data[:, 1:]
        yield inputs, targets

def get_distributed_dataloader(batch_size, seq_len, split, tokenizer, mesh):
    sharding = jax.NamedSharding(mesh, jax.P("b", None))
    global_batch_size = batch_size * jax.local_device_count() * jax.process_count()
    loader = tokenizing_data_loader(batch_size, seq_len, split, tokenizer)
    return map(
        partial(jax.make_array_from_process_local_data, sharding, global_shape=(global_batch_size, seq_len)),
        loader
    )
