import argparse
from functools import partial

import jax
import jax.numpy as jnp

from nanojax.common import print0, init_distributed, get_base_dir
from nanojax.checkpointing import load_checkpoint, load_model_config
from nanojax.gpt import GPT
from nanojax.tokenizer import get_tokenizer
from nanojax.generation import generate

from tasks.mmlu import MMLU
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.humaneval import HumanEval


def run_generative_eval(task_object, tokenizer, model, num_samples, max_new_tokens, temperature, compute_dtype, max_problems=None):
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    num_passed, total = 0, 0
    for i in range(num_problems):
        conversation = task_object[i]

        encoded_prompt = tokenizer.render_for_completion(conversation)

        results = []
        for _ in range(num_samples):
            rng = jax.random.key(42)
            pad_token_id = tokenizer.encode_special("<|assistant_end|>")
            completion_tokens = generate(
                encoded_prompt,
                model,
                max_new_tokens,
                temperature,
                compute_dtype,
                pad_token_id,
                rng,
                assistant_end_id=tokenizer.encode_special("<|assistant_end|>")
            )
            completion_tokens_list = [int(t[0]) for t in completion_tokens]
            results.append(encoded_prompt + completion_tokens_list)

        prefix_length = len(encoded_prompt)
        completions = [tokenizer.decode(result_tokens[prefix_length:]) for result_tokens in results]

        outcomes = [task_object.evaluate(conversation, completion) for completion in completions]
        passed = any(outcomes)

        total += 1
        num_passed += int(passed)

        print(f"passed/total: \r\033[k{num_passed}/{total} ({100*num_passed/total:.2f}%)", end='', flush=True)

    print()
    print0("=" * 50)
    print0(f"final passed/total: {num_passed}/{total} ({100*num_passed/total:.2f}%)")

    return num_passed/total


def run_categorical_eval(task_object, tokenizer, model, batch_size, compute_dtype, max_problems=None):
    bos = tokenizer.get_bos_token_id()

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    ceil_div = lambda x, y: -(-x // y)
    num_batches = ceil_div(num_problems, batch_size)

    letter_to_id_cache = {}
    num_passed, total = 0, 0

    for i in range(num_batches):
        i0, i1 = i * batch_size, min((i + 1) * batch_size, num_problems)

        conversations = [task_object[ii] for ii in range(i0, i1)]
        prompt_ids = [tokenizer.render_for_completion(conversation) for conversation in conversations]
        max_length = max(len(ids) for ids in prompt_ids)
        # pad to one of the below bucket sizes for JIT compatibility
        bucket_sizes = (256, 512, 1024, 2048)
        max_length = min(b for b in bucket_sizes if b >= max_length)
        
        answer_time_positions = [len(ids) - 1 for ids in prompt_ids]
        padded_prompt_ids = [ids + [bos] * (max_length - len(ids)) for ids in prompt_ids]
        prompt_ids_array = jnp.array(padded_prompt_ids, dtype=jnp.int32)

        logits, _ = model.forward(prompt_ids_array, compute_dtype=compute_dtype)

        for idx, conversation in enumerate(conversations):
            letters = conversation['letters']
            letter_ids = []
            for letter in letters:
                if letter not in letter_to_id_cache:
                    encoded_letter = tokenizer.encode(letter)
                    assert len(encoded_letter) == 1, "each letter must be a single token"
                    letter_to_id_cache[letter] = encoded_letter[0]
                letter_ids.append(letter_to_id_cache[letter])

            answer_pos = answer_time_positions[idx]
            focus_logits = logits[idx, answer_pos, jnp.array(letter_ids)]
            argmax_letter_id = int(jnp.argmax(focus_logits))
            predicted_letter = letters[argmax_letter_id]

            outcome = task_object.evaluate(conversation, predicted_letter)
            num_passed += int(outcome)
            total += 1

    average = num_passed/total
    print0(f"final: {num_passed}/{total} ({100*average:.2f}%)")
    return average


def run_chat_eval(task_name, model, tokenizer, compute_dtype, seed,
                   batch_size=1, num_samples=1, max_new_tokens=512, temperature=0.0,
                   max_problems=None):
    task_module = {
        'MMLU': partial(MMLU, subset="all", split="test", seed=seed),
        'ARC-Easy': partial(ARC, subset="ARC-Easy", split="test", seed=seed),
        'ARC-Challenge': partial(ARC, subset="ARC-Challenge", split="test", seed=seed),
        'GSM8K': partial(GSM8K, subset="main", split="test", seed=seed),
        'HumanEval': partial(HumanEval, seed=seed),
    }[task_name]
    task = task_module()

    if task.eval_type == 'generative':
        acc = run_generative_eval(task, tokenizer, model, num_samples, max_new_tokens, temperature, compute_dtype, max_problems=max_problems)
    elif task.eval_type == 'categorical':
        acc = run_categorical_eval(task, tokenizer, model, batch_size, compute_dtype, max_problems=max_problems)
    else:
        raise ValueError(f"unsupported task evaluation type: {task.eval_type}")
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help="checkpoint to load: mid|sft")
    parser.add_argument('--task-name', type=str, default=None, help="task name. default = all tasks. use | to split multiple tasks.")
    parser.add_argument('--compute-dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8, help='batch size for categorical evaluation')
    parser.add_argument('--max-problems', type=int, default=None, help='max problems to evaluate')
    args = parser.parse_args()

    compute_dtype = jnp.bfloat16 if args.compute_dtype == 'bfloat16' else jnp.float32

    base_dir = get_base_dir()
    checkpoint_dir = base_dir / f"{args.checkpoint}_checkpoints"
    model_cfg = load_model_config(checkpoint_dir / "model.zarr")

    print0(f"Loading model from {checkpoint_dir}")
    seed = 42
    rng = jax.random.key(seed)
    model = GPT.init(model_cfg, rng)
    model = load_checkpoint(checkpoint_dir / "model.zarr", model)

    tokenizer = get_tokenizer()

    all_tasks = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval']
    baseline_accuracies = {
        'ARC-Easy': 0.25, # multiple choice 1 of 4 => 25%
        'ARC-Challenge': 0.25, # multiple choice 1 of 4 => 25%
        'MMLU': 0.25, # multiple choice 1 of 4 => 25%
        'GSM8K': 0.0, # open-ended => 0%
        'HumanEval': 0.0, # open-ended => 0%
    }
    task_names = all_tasks if args.task_name is None else args.task_name.split('|')

    results = {}
    for task_name in task_names:
        acc = run_chat_eval(
            task_name,
            model,
            tokenizer,
            compute_dtype,
            seed,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_problems=args.max_problems,
        )
        results[task_name] = acc
        print0(f"{task_name} accuracy: {100 * acc:.2f}%")

    all_tasks_were_evaluated = all(task_name in results for task_name in all_tasks)
    if all_tasks_were_evaluated and jax.process_index() == 0:
        centered_mean = 0
        for task_name, acc in results.items():
            baseline_acc = baseline_accuracies.get(task_name, 0.0)
            centered_acc = (acc - baseline_acc) / (1.0 - baseline_acc)
            centered_mean += centered_acc
        chatcore_metric = centered_mean / len(results)
        print0(f"ChatCORE metric: {chatcore_metric:.4f}")
