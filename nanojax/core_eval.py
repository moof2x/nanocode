"""
CORE metric https://arxiv.org/abs/2406.11794
Mostly copy-pasted from nanochat/core_eval.py
"""
import random
from functools import partial

from jinja2 import Template
import jax
import jax.numpy as jnp
import numpy as np

def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """render complete prompts for a multiple choice question"""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(choice=choice, **context) for choice in item['choices']]
    return prompts


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """render complete prompts for a schema question"""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(context=context_option, **context)
               for context_option in item['context_options']]
    return prompts


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """render complete prompt for a language modeling task"""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompt_without = template.render(include_continuation=False, **context)
    prompt_with = template.render(include_continuation=True, **context)
    prompt_without = prompt_without.strip()
    return [prompt_without, prompt_with]


def find_common_length(token_sequences, direction='left'):
    """find the length of the common prefix or suffix across token sequences"""
    min_len = min(len(seq) for seq in token_sequences)
    indices = {
        'left': range(min_len),
        'right': range(-1, -min_len-1, -1)
    }[direction]
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def stack_sequences(tokens, pad_token_id):
    # stack a list of sequences and pad to one of bucket sizes for JIT compatibility
    bucket_sizes = (256, 512, 1024, 2048, 4096)

    bsz, seq_len = len(tokens), max(len(x) for x in tokens)
    seq_len = min(b for b in bucket_sizes if b >= seq_len)
    
    input_ids = np.full((bsz, seq_len), pad_token_id, dtype=jnp.int32)    
    for i, x in enumerate(tokens):
        input_ids[i, :len(x)] = np.array(x, dtype=jnp.int32)
    return jnp.asarray(input_ids)


def batch_sequences_mc(tokenizer, prompts):
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    answer_start_idx = find_common_length(tokens, direction='left')
    start_indices = [answer_start_idx] * len(prompts)
    end_indices = [len(x) for x in tokens]
    return tokens, start_indices, end_indices


def batch_sequences_schema(tokenizer, prompts):
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    suffix_length = find_common_length(tokens, direction='right')
    end_indices = [len(x) for x in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenizer, prompts):
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    tokens_without, tokens_with = tokens
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    assert start_idx < end_idx, "prompt without is supposed to be a prefix of prompt with"
    assert tokens_without == tokens_with[:start_idx], "prompt without is supposed to be a prefix of prompt with"
    return [tokens_with], [start_idx], [end_idx]


# def forward_model_jax(input_ids, model, compute_dtype, mesh):

#     return _forward(input_ids, model,)





def extract_correctness(losses, predictions, metadata, examples, task_meta, input_ids):
    correct = []
    num_seqs_per_example = metadata['num_seqs_per_example']
    start_idxs = metadata['start_idxs']
    end_idxs = metadata['end_idxs']
    task_type = task_meta['task_type']

    seq_idx = 0
    for i, example in enumerate(examples):
        num_seqs = num_seqs_per_example[i]
        example_losses = losses[seq_idx:seq_idx+num_seqs]
        example_preds = predictions[seq_idx:seq_idx+num_seqs]
        example_start_idxs = start_idxs[seq_idx:seq_idx+num_seqs]
        example_end_idxs = end_idxs[seq_idx:seq_idx+num_seqs]
        example_input_ids = input_ids[seq_idx:seq_idx+num_seqs]

        if task_type == 'language_modeling':
            si = example_start_idxs[0]
            ei = example_end_idxs[0]
            predicted_tokens = example_preds[0, si-1:ei-1]
            actual_tokens = example_input_ids[0, si:ei]
            is_correct = bool(jnp.all(predicted_tokens == actual_tokens))
        elif task_type in ['multiple_choice', 'schema']:
            mean_losses = [float(jnp.nanmean(example_losses[j, si-1:ei-1]))
                          for j, (si, ei) in enumerate(zip(example_start_idxs, example_end_idxs))]
            pred_idx = mean_losses.index(min(mean_losses))
            is_correct = pred_idx == example['gold']
        else:
            raise ValueError(f"unsupported task type: {task_type}")

        correct.append(float(is_correct))
        seq_idx += num_seqs

    return correct

def prepare_batch(examples, tokenizer, task_meta, max_seq_len, data):
    ids = []
    all_start_idxs = []
    all_end_idxs = []
    num_seqs_per_example = []

    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']

    for idx, example in enumerate(examples):
        fewshot_examples = []
        if num_fewshot > 0:
            global_idx = data.index(example)
            rng = random.Random(1234 + global_idx)
            available_indices = [i for i in range(len(data)) if i != global_idx]
            fewshot_indices = rng.sample(available_indices, num_fewshot)
            fewshot_examples = [data[i] for i in fewshot_indices]

        if task_type == 'multiple_choice':
            prompts = render_prompts_mc(example, continuation_delimiter, fewshot_examples)
            tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
        elif task_type == 'schema':
            prompts = render_prompts_schema(example, continuation_delimiter, fewshot_examples)
            tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
        elif task_type == 'language_modeling':
            prompts = render_prompts_lm(example, continuation_delimiter, fewshot_examples)
            tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
        else:
            raise ValueError(f"unsupported task type: {task_type}")

        new_tokens, new_start_idxs, new_end_idxs = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_seq_len:
                num_to_crop = len(t) - max_seq_len
                new_tokens.append(t[-max_seq_len:])
                new_start_idxs.append(s - num_to_crop)
                new_end_idxs.append(e - num_to_crop)
                assert s - num_to_crop >= 0
                assert e - num_to_crop >= 0
            else:
                new_tokens.append(t)
                new_start_idxs.append(s)
                new_end_idxs.append(e)

        all_tokens.extend(new_tokens)
        all_start_idxs.extend(new_start_idxs)
        all_end_idxs.extend(new_end_idxs)
        num_seqs_per_example.append(len(new_tokens))

    pad_token_id = tokenizer.get_bos_token_id()
    input_ids = stack_sequences(all_tokens, pad_token_id)

    return input_ids, {
        'num_seqs_per_example': num_seqs_per_example,
        'start_idxs': all_start_idxs,
        'end_idxs': all_end_idxs,
    }

def evaluate_task(model, tokenizer, data, minibatch_size, task_meta, compute_dtype, mesh):
    world_size, rank = jax.process_count(), jax.process_index()
    minibatch_size *= jax.local_device_count()
    ignore_idx = tokenizer.get_bos_token_id()
    
    @jax.jit
    @jax.shard_map(
        mesh=mesh,
        in_specs=(
            jax.P("b", None), 
            jax.tree.map(lambda _: jax.P(), model) 
        ),
        out_specs=(
            jax.P("b", None),
            jax.P("b", None)
        ),
        check_vma=False
    )
    def eval_forward(idx, model):
        logits, _ = model.forward(idx, compute_dtype=compute_dtype)
        targets = jnp.roll(idx, -1, axis=1)

        logsumexp = jax.nn.logsumexp(logits.astype(jnp.float32))
        valid_targets = jnp.not_equal(targets, ignore_idx)
        loss = -jnp.take_along_axis(logits, targets[:, :, None], axis=-1).squeeze(-1) + logsumexp
        loss = jnp.where(valid_targets, loss, 0.0)

        preds = jnp.argmax(logits, axis=-1)
        return loss, preds

    all_correct = []
    sharding = jax.NamedSharding(mesh, jax.P("b", None))
    for i in range(rank, len(data), minibatch_size):
        # each process collects data for all of its local devices
        batch_data = data[i:(i + 1) * minibatch_size]

        ids, meta = prepare_batch(
            batch_data, tokenizer, task_meta, model.cfg.sequence_len, data
        )

        # create a sharded global view of our data across the entire world size
        global_batch_size = ids.shape[0] * world_size
        ids = jax.make_array_from_process_local_data(
            sharding, ids,
            global_shape=(global_batch_size, ids.shape[1])
        )

        loss, preds= eval_forward(ids, model)

        is_correct = extract_correctness(
            loss, preds, meta, batch_data, task_meta, ids
        )
        all_correct.extend(is_correct)

    return sum(all_correct) / len(all_correct) if all_correct else 0.0
