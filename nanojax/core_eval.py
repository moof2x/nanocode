"""
CORE metric https://arxiv.org/abs/2406.11794
Mostly copy-pasted from nanochat/core_eval.py
"""
import random
from functools import partial

from jinja2 import Template
import jax
import jax.numpy as jnp


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
    """stack up a list of token sequences, pad to longest on the right"""
    bsz, seq_len = len(tokens), max(len(x) for x in tokens)
    input_ids = jnp.full((bsz, seq_len), pad_token_id, dtype=jnp.int32)
    for i, x in enumerate(tokens):
        input_ids = input_ids.at[i, :len(x)].set(jnp.array(x, dtype=jnp.int32))
    return input_ids


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


@partial(jax.jit, static_argnames=['compute_dtype'])
def forward_model_jax(input_ids, model, compute_dtype):
    """
    take BxT tensor of token ids, return BxT tensor of losses and argmax predictions.
    the last column of losses is set to nan because we don't have autoregressive targets there.
    """
    batch_size, seq_len = input_ids.shape
    logits, _ = model.forward(input_ids, compute_dtype=compute_dtype)
    target_ids = jnp.roll(input_ids, -1, axis=1)

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    losses = -jnp.take_along_axis(log_probs, target_ids[:, :, None], axis=-1).squeeze(-1)
    losses = losses.at[:, -1].set(jnp.nan)

    predictions = jnp.argmax(logits, axis=-1)
    return losses, predictions


def evaluate_example(idx, model, tokenizer, data, task_meta, compute_dtype):
    """evaluate a single example, return true if correct, false otherwise"""
    item = data[idx]
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']

    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]

    if task_type == 'multiple_choice':
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_type == 'schema':
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    elif task_type == 'language_modeling':
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"unsupported task type: {task_type}")

    if hasattr(model, 'cfg') and hasattr(model.cfg, 'sequence_len') and model.cfg.sequence_len is not None:
        max_tokens = model.cfg.sequence_len
        new_tokens, new_start_idxs, new_end_idxs = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_tokens:
                num_to_crop = len(t) - max_tokens
                new_tokens.append(t[-max_tokens:])
                new_start_idxs.append(s - num_to_crop)
                new_end_idxs.append(e - num_to_crop)
                assert s - num_to_crop >= 0, "this should never happen right?"
                assert e - num_to_crop >= 0, "this should never happen right?"
            else:
                new_tokens.append(t)
                new_start_idxs.append(s)
                new_end_idxs.append(e)
        tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs

    pad_token_id = tokenizer.get_bos_token_id()
    input_ids = stack_sequences(tokens, pad_token_id)

    losses, predictions = forward_model_jax(input_ids, model, compute_dtype)

    if task_type == 'language_modeling':
        si = start_idxs[0]
        ei = end_idxs[0]
        predicted_tokens = predictions[0, si-1:ei-1]
        actual_tokens = input_ids[0, si:ei]
        is_correct = bool(jnp.all(predicted_tokens == actual_tokens))
    elif task_type in ['multiple_choice', 'schema']:
        mean_losses = [float(jnp.nanmean(losses[i, si-1:ei-1]))
                        for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))]
        pred_idx = mean_losses.index(min(mean_losses))
        is_correct = pred_idx == item['gold']
    else:
        raise ValueError(f"unsupported task type: {task_type}")

    return is_correct


def evaluate_task(model, tokenizer, data, task_meta, compute_dtype):
    """
    evaluate one task across many examples.
    """
    correct = []
    for idx in range(len(data)):
        is_correct = evaluate_example(idx, model, tokenizer, data, task_meta, compute_dtype)
        correct.append(float(is_correct))

    mean_correct = sum(correct) / len(correct)
    return mean_correct
