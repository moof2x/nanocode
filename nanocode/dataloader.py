from functools import partial

import jax
import numpy as np
import random

from data.pretrain import parquets_iter_batched


def document_batch_iterator(split, rank, world_size, tokenizer_batch_size, code_ratio):
    def parquet_iter(dataset):
        while True:
            yield from parquets_iter_batched(dataset, split=split, start=rank, step=world_size)

    fineweb = parquet_iter("fineweb-edu") if code_ratio < 1 else None
    stack_v2 = parquet_iter("the-stack-v2-dedup") if code_ratio > 0 else None
    batch_iter = lambda: stack_v2 if stack_v2 and (not fineweb or random.random() < code_ratio) else fineweb
    while True:
        batch = next(batch_iter())
        for i in range(0, len(batch), tokenizer_batch_size):
            yield batch[i:i+tokenizer_batch_size]


def tokenizing_data_loader(B, T, split, tokenizer, code_ratio, tokenizer_threads=4, tokenizer_batch_size=128, buffer_size=1000):
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    B *= jax.local_device_count() # per-device batch size to process batch size
    world_size, rank = jax.process_count(), jax.process_index()

    batch_iterator = document_batch_iterator(split, rank, world_size, tokenizer_batch_size, code_ratio)
    # we'll use a single buffer for our collated tokens
    row_buffer = np.empty((B, T + 1), dtype=np.int32)
    while True:
        for row_idx in range(B):
            pos = 0
            while pos < T + 1:
                # accumulate enough documents in buffer before packing
                while len(doc_buffer) < buffer_size:
                    doc_batch = next(batch_iterator)
                    token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
                    for tokens in token_lists:
                        doc_buffer.append(tokens)
                remaining = (T + 1) - pos

                # pack the current row using best-fit algorithm
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                # best_idx, best_len: index and length of the best-fit doc (or -1, 0)
                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = doc
                    pos += len(doc)
                else:
                    # no doc fits - crop shortest to fill remaining empty space
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = doc[:remaining]
                    pos += remaining

        # all rows have been filled, we can now yield
        inputs = row_buffer[:, :-1]
        targets = row_buffer[:, 1:]
        yield inputs, targets

def get_distributed_dataloader(batch_size, seq_len, split, tokenizer, code_ratio, mesh):
    sharding = jax.NamedSharding(mesh, jax.P("b", None))
    global_batch_size = batch_size * jax.local_device_count() * jax.process_count()
    loader = tokenizing_data_loader(batch_size, seq_len, split, tokenizer, code_ratio)
    return map(
        partial(jax.make_array_from_process_local_data, sharding, global_shape=(global_batch_size, seq_len)),
        loader
    )
