"""
CORE metric https://arxiv.org/abs/2406.11794
"""

import random

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

def batch_sequences_mc(tokenizer, prompts):
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    prefix_length = find_common_length(tokens, direction='left')
    masks = [[0] * prefix_length + [1] * (len(t) - prefix_length) for t in tokens]
    return tokens, masks # start_indices, end_indices


def batch_sequences_schema(tokenizer, prompts):
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    suffix_length = find_common_length(tokens, direction='right')
    masks = [[0] * (len(t) - suffix_length) + [1] * suffix_length for t in tokens]
    return tokens, masks # start_indices, end_indices


def batch_sequences_lm(tokenizer, prompts):
    tokens_without, tokens_with = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    mask = [0] * len(tokens_without) + [1] * (len(tokens_with) - len(tokens_without))
    return [tokens_with], [mask] # [start_idx], [end_idx]

def collate(idx, masks, pad_token_id, max_seq_len):
    seq_len = max(len(s) for s in idx)
    # seq_len should always be <= model.cfg.seq_len
    assert seq_len <= max_seq_len
    # stack a list of sequences and pad to one of the bucket sizes for JIT compatibility
    seq_len = min([s for s in [128, 256, 512, 1024, 2048, 4096] if s >= seq_len])
    input_ids = np.full((len(idx), seq_len), pad_token_id, dtype=np.int32)
    input_masks = np.full((len(idx), seq_len), 0, dtype=np.int32)
    for i, (idx, mask) in enumerate(zip(idx, masks)):
        input_ids[i, :len(idx)] = idx
        input_masks[i, :len(mask)] = mask
    return input_ids, input_masks.astype(bool)

    
def tokenize_example(example, tokenizer, task_meta, max_seq_len, data):
    # takes single example and tokenizes, returning tuple of lists ids and masks
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']
    
    # for >0-shot tasks we prepend some examples
    fewshot_examples = []
    if num_fewshot > 0:
        global_idx = data.index(example)
        rng = random.Random(1234 + global_idx)
        available_idxs = [i for i in range(len(data)) if i != global_idx]
        fewshot_idxs = rng.sample(available_idxs, num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_idxs]

    if task_type == 'multiple_choice':
        prompts = render_prompts_mc(example, continuation_delimiter, fewshot_examples)
        tokens, masks = batch_sequences_mc(tokenizer, prompts)
    elif task_type == 'schema':
        prompts = render_prompts_schema(example, continuation_delimiter, fewshot_examples)
        tokens, masks = batch_sequences_schema(tokenizer, prompts)
    elif task_type == 'language_modeling':
        prompts = render_prompts_lm(example, continuation_delimiter, fewshot_examples)
        tokens, masks = batch_sequences_lm(tokenizer, prompts)

    all_idxs, all_masks = [], [] 
    for t, m in zip(tokens, masks):
        if len(t) > max_seq_len:
            all_idxs.append(t[-max_seq_len:])
            all_masks.append(m[-max_seq_len:])
        else:
            all_idxs.append(t)
            all_masks.append(m)
 
    return all_idxs, all_masks, len(tokens), example.get("gold", None)

@jax.jit(static_argnames=['compute_dtype', 'ignore_idx', 'mesh'])
def eval_forward(idx, masks, model, compute_dtype, ignore_idx, mesh):
    @jax.shard_map(
        mesh=mesh,
        in_specs=(
            jax.P("b", None), 
            jax.P("b", None), 
            jax.tree.map(lambda _: jax.P(), model) 
        ),
        out_specs=(
            jax.P(),
            jax.P(),
            jax.P()
        ),
        check_vma=False
    )
    def forward(idx, masks, model):
        logits, _ = model.forward(idx, compute_dtype=compute_dtype)
        logits = logits.astype(jnp.float32)
        targets = jnp.roll(idx, -1, axis=1)
        target_masks = jnp.roll(masks, -1, axis=1)

        logsumexp = jax.nn.logsumexp(logits, axis=-1)
        valid_targets = jnp.not_equal(targets, ignore_idx) & target_masks
        loss = -jnp.take_along_axis(logits, targets[:, :, None], axis=-1).squeeze(-1) + logsumexp
        loss = jnp.where(valid_targets, loss, 0.0)
        
        loss = jnp.sum(loss, axis=1)
        loss = jax.lax.all_gather(loss, "b").reshape(-1)
        
        preds = jnp.argmax(logits, axis=-1)
        # score language modeling task targets while we're here
        is_correct = jnp.all((preds == targets) | ~valid_targets, axis=1)
        is_correct = jax.lax.all_gather(is_correct, "b").reshape(-1)
        
        sequence_lengths = jnp.sum(valid_targets, axis=1)
        sequence_lengths = jax.lax.all_gather(sequence_lengths, "b").reshape(-1)
        return loss, is_correct, sequence_lengths
    
    return forward(idx, masks, model)
     
def evaluate_task(model, tokenizer, minibatch_size, data, task_meta, compute_dtype, mesh):
    task_type = task_meta["task_type"]
    assert task_type in ["language_modeling", "multiple_choice", "schema"], f"Unsupported task type{task_type}"

    minibatch_size *= jax.local_device_count()
    ignore_idx = tokenizer.get_bos_token_id()
    max_seq_len = model.cfg.sequence_len
    correct = []
    cursor = 0
    
    while cursor < len(data):
        # collect minibatch_size number of rows. this may not always be minibatch_size number of examples
        idx, masks, num_seqs, answers = [], [], [], []
        while len(idx) < minibatch_size:
            if cursor >= len(data):
                # we've reached the end of our dataset.
                # pad to a fixed batch size and evaluate
                items_to_pad = minibatch_size - len(idx)
                # but only pad ids and masks, so our scoring loop later still only looks at valid rows
                idx.extend([idx[-1]] * items_to_pad)
                masks.extend([masks[-1]] * items_to_pad)
                break
            batch_idx, batch_masks, batch_num_seqs, batch_answers = tokenize_example(data[cursor], tokenizer, task_meta, max_seq_len, data)
            idx.extend(batch_idx)
            masks.extend(batch_masks)
            num_seqs.append(batch_num_seqs)
            answers.append(batch_answers)
            cursor += 1

        # stack and pad sequences to a fixed bucket size
        idx, masks = collate(idx, masks, ignore_idx, max_seq_len)
        # forward samples through our model
        loss, pred_match, sequence_lengths = eval_forward(idx, masks, model, compute_dtype, ignore_idx, mesh)

        # score samples, using our stored metadata to unpack grouped sequences
        batch_idx = 0
        is_correct = False
        for n, a in zip(num_seqs, answers):
            # for multiple choice samples, we calcualte the mean loss over the unmasked tokens
            # then select the answer with the lowest loss, and compare against gold
            if task_type in ["multiple_choice", "schema"]:
                choice_losses = loss[batch_idx:batch_idx + n]
                seq_lens = sequence_lengths[batch_idx:batch_idx + n]
                mean_losses = choice_losses / seq_lens
                pred_idx = np.argmin(mean_losses)
                is_correct = bool(pred_idx == a)
                batch_idx += n
            elif task_type == "language_modeling":     
                # for language modeling tasks, we simply check if all of preds are equal to targets
                # here's one I made earlier
                is_correct = bool(pred_match[batch_idx])
                batch_idx += 1
            correct.append(is_correct)
            
    return sum(correct) / len(correct) if correct else 0.0

