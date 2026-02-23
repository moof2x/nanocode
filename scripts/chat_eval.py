import argparse
from functools import partial

import jax
import jax.numpy as jnp

from nanojax.common import print0, init_distributed, get_base_dir, setup_logging
from nanojax.checkpointing import load_checkpoint, load_model_config
from nanojax.gpt import GPT
from nanojax.tokenizer import get_tokenizer
from nanojax.generation import generate

from tasks.mmlu import MMLU
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.humaneval import HumanEval


def run_generative_eval(task_object, tokenizer, model, num_samples, max_new_tokens, temperature, compute_dtype, rng, max_problems=None):
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    num_passed, total = 0, 0
    for i in range(num_problems):
        conversation = task_object[i]

        encoded_prompt = tokenizer.render_for_completion(conversation)

        results = []
        for s in range(num_samples):
            sample_rng = jax.random.fold_in(rng, i * num_samples + s)
            pad_token_id = tokenizer.encode_special("<|assistant_end|>")
            completion_tokens = generate(
                encoded_prompt,
                model,
                max_new_tokens,
                temperature,
                compute_dtype,
                pad_token_id,
                sample_rng,
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


@jax.jit(static_argnames=['compute_dtype', 'mesh'])
def categorical_forward(prompt_ids, answer_positions, model, compute_dtype, mesh):
    @jax.shard_map(
        mesh=mesh,
        in_specs=(
            jax.P("b", None),
            jax.P("b"),
            jax.tree.map(lambda _: jax.P(), model)
        ),
        out_specs=jax.P(),
        check_vma=False
    )
    def forward(prompt_ids, answer_positions, model):
        logits, _ = model.forward(prompt_ids, compute_dtype=compute_dtype)
        logits = logits.astype(jnp.float32)
        focused = jax.vmap(lambda l, p: l[p])(logits, answer_positions)
        return jax.lax.all_gather(focused, "b").reshape(-1, focused.shape[-1])
    return forward(prompt_ids, answer_positions, model)


def run_categorical_eval(task_object, tokenizer, model, minibatch_size, compute_dtype, mesh, max_problems=None):
    bos = tokenizer.get_bos_token_id()
    minibatch_size *= jax.local_device_count()

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    letter_to_id_cache = {}
    num_passed, total = 0, 0
    cursor = 0

    while cursor < num_problems:
        conversations, prompt_ids = [], []
        while len(prompt_ids) < minibatch_size:
            if cursor >= num_problems:
                items_to_pad = minibatch_size - len(prompt_ids)
                prompt_ids.extend([prompt_ids[-1]] * items_to_pad)
                break
            conversation = task_object[cursor]
            conversations.append(conversation)
            prompt_ids.append(tokenizer.render_for_completion(conversation))
            cursor += 1
        actual_batch = len(conversations)

        max_length = max(len(ids) for ids in prompt_ids)
        bucket_sizes = (128, 256, 512, 1024, 2048, 4096)
        max_length = min(b for b in bucket_sizes if b >= max_length)

        answer_time_positions = [len(ids) - 1 for ids in prompt_ids]
        padded_prompt_ids = [ids + [bos] * (max_length - len(ids)) for ids in prompt_ids]
        prompt_ids_array = jnp.array(padded_prompt_ids, dtype=jnp.int32)
        answer_positions_array = jnp.array(answer_time_positions, dtype=jnp.int32)

        focused_logits = categorical_forward(prompt_ids_array, answer_positions_array, model, compute_dtype, mesh)

        for idx, conversation in enumerate(conversations[:actual_batch]):
            letters = conversation['letters']
            letter_ids = []
            for letter in letters:
                if letter not in letter_to_id_cache:
                    encoded_letter = tokenizer.encode(letter)
                    assert len(encoded_letter) == 1, "each letter must be a single token"
                    letter_to_id_cache[letter] = encoded_letter[0]
                letter_ids.append(letter_to_id_cache[letter])

            focus_logits = focused_logits[idx, jnp.array(letter_ids)]
            argmax_letter_id = int(jnp.argmax(focus_logits))
            predicted_letter = letters[argmax_letter_id]

            outcome = task_object.evaluate(conversation, predicted_letter)
            num_passed += int(outcome)
            total += 1

    average = num_passed/total
    print0(f"final: {num_passed}/{total} ({100*average:.2f}%)")
    return average


def run_chat_eval(task_name, model, tokenizer, compute_dtype, mesh, seed, rng,
                   minibatch_size=1, num_samples=1, max_new_tokens=512, temperature=0.0,
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
        acc = run_generative_eval(task, tokenizer, model, num_samples, max_new_tokens, temperature, compute_dtype, rng, max_problems=max_problems)
    elif task.eval_type == 'categorical':
        acc = run_categorical_eval(task, tokenizer, model, minibatch_size, compute_dtype, mesh, max_problems=max_problems)
    else:
        raise ValueError(f"unsupported task evaluation type: {task.eval_type}")
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help="checkpoint to load: sft|dpo")
    parser.add_argument('--task-name', type=str, default=None, help="task name. default = all tasks. use | to split multiple tasks.")
    parser.add_argument('--compute-dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
    parser.add_argument('--attn-impl', type=str, default="splash", help="Attention backend: splash (TPU) or eager")
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--minibatch-size', type=int, default=8, help='per-device batch size for categorical evaluation')
    parser.add_argument('--max-problems', type=int, default=None, help='max problems to evaluate')
    args = parser.parse_args()
    world_size, mesh = init_distributed()

    if jax.process_index() == 0:
        compute_dtype = jnp.bfloat16 if args.compute_dtype == 'bfloat16' else jnp.float32

        base_dir = get_base_dir()
        checkpoint_dir = base_dir / f"{args.checkpoint}_checkpoints"
        model_cfg = load_model_config(checkpoint_dir / "model.zarr")
        setup_logging(base_dir / "chat_eval" / f"{args.checkpoint}.txt")

        print0(f"Loading model from {checkpoint_dir}")
        seed = 42
        rng = jax.random.key(seed)
        model = GPT.init(model_cfg, rng, attn_impl=args.attn_impl)
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
        centered_results = {}
        for task_name in task_names:
            acc = run_chat_eval(
                task_name,
                model,
                tokenizer,
                compute_dtype,
                mesh,
                seed,
                rng,
                minibatch_size=args.minibatch_size,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                max_problems=args.max_problems,
            )
            results[task_name] = acc
            baseline_acc = baseline_accuracies.get(task_name, 0.0)
            centered_results[task_name] = (acc - baseline_acc) / (1.0 - baseline_acc)
            print0(f"{task_name} accuracy: {100 * acc:.2f}%")

        output_csv_path = base_dir / "chat_eval" / f"{args.checkpoint}.csv"
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
            f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
            for task_name in results:
                f.write(f"{task_name:<35}, {results[task_name]:<10.6f}, {centered_results[task_name]:<10.6f}\n")

        all_tasks_were_evaluated = all(task_name in results for task_name in all_tasks)
        if all_tasks_were_evaluated:
            chatcore_metric = sum(centered_results.values()) / len(centered_results)
            with open(output_csv_path, 'a', encoding='utf-8') as f:
                f.write(f"{'ChatCORE':<35}, {'':<10}, {chatcore_metric:<10.6f}\n")
            print0(f"ChatCORE metric: {chatcore_metric:.4f}")
