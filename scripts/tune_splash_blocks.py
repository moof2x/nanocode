"""tune splash attention block sizes for different model configurations"""
import time
import jax
import jax.numpy as jnp
from nanojax import configs
from nanojax.gpt import GPT
from nanojax.common import init_distributed

# block size configurations to test
BLOCK_CONFIGS = [
    # (block_q/kv, block_compute)
    (128, 128),
    (256, 128),
    (512, 128),
    (1024, 128),
    (256, 256),
    (512, 256),
]

def benchmark_config(model, idx, block_q, block_kv, block_compute, num_steps=10, warmup=3):
    """benchmark a specific block configuration"""
    from nanojax import gpt
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask

    # clear cache
    gpt._splash_kernel_cache.clear()

    # temporarily replace create_splash_kernel
    original_create = gpt.create_splash_kernel

    def patched_create(seq_len, n_head, n_kv_head, head_dim):
        mask = splash_attention_mask.CausalMask(shape=(seq_len, seq_len))
        block_config = splash_attention_kernel.BlockSizes(
            block_q=block_q,
            block_kv=block_kv,
            block_kv_compute=block_compute,
            block_q_dkv=block_q,
            block_kv_dkv=block_kv,
            block_kv_dkv_compute=block_compute,
            block_q_dq=block_q,
            block_kv_dq=block_kv,
        )
        multi_head_mask = splash_attention_mask.MultiHeadMask(masks=(mask,) * n_head)
        return splash_attention_kernel.make_splash_mha(
            mask=multi_head_mask,
            head_shards=1,
            q_seq_shards=1,
            block_sizes=block_config,
        )

    gpt.create_splash_kernel = patched_create

    try:
        # warmup
        for _ in range(warmup):
            logits, _ = model.forward(idx)
            logits.block_until_ready()

        # benchmark
        times = []
        for _ in range(num_steps):
            start = time.time()
            logits, _ = model.forward(idx)
            logits.block_until_ready()
            times.append(time.time() - start)

        avg_time = sum(times) / len(times)
        return avg_time
    finally:
        # restore original
        gpt.create_splash_kernel = original_create
        gpt._splash_kernel_cache.clear()

def tune_model(config_name):
    """tune block sizes for a specific model config"""
    print(f"\n{'='*60}")
    print(f"tuning {config_name}")
    print(f"{'='*60}")

    # get config
    cfg = getattr(configs, config_name)
    print(f"config: {cfg}")

    # init model
    rng = jax.random.PRNGKey(42)
    model = GPT.init(cfg, rng)

    # create test input
    batch_size = 2
    seq_len = cfg.sequence_len
    idx = jax.random.randint(rng, (batch_size, seq_len), 0, cfg.vocab_size)

    results = []
    for block_q, block_compute in BLOCK_CONFIGS:
        # check if valid (block must divide seq_len and be multiple of block_compute)
        if seq_len % block_q != 0:
            continue
        if block_q % block_compute != 0:
            continue

        try:
            print(f"\ntesting block_q={block_q}, block_kv={block_q}, block_compute={block_compute}")
            avg_time = benchmark_config(model, idx, block_q, block_q, block_compute)
            tokens_per_sec = (batch_size * seq_len) / avg_time

            print(f"  avg time: {avg_time:.3f}s")
            print(f"  tokens/sec: {tokens_per_sec:.0f}")

            results.append({
                'block_q': block_q,
                'block_kv': block_q,
                'block_compute': block_compute,
                'time': avg_time,
                'tokens_per_sec': tokens_per_sec,
            })
        except Exception as e:
            print(f"  failed: {e}")

    # find best
    if results:
        best = max(results, key=lambda x: x['tokens_per_sec'])
        print(f"\nbest configuration:")
        print(f"  block_q/kv: {best['block_q']}")
        print(f"  block_compute: {best['block_compute']}")
        print(f"  tokens/sec: {best['tokens_per_sec']:.0f}")
        print(f"  time: {best['time']:.3f}s")

        return best
    return None

if __name__ == "__main__":
    world_size, mesh = init_distributed()
    print(f"world size: {world_size}")

    # configs to tune
    model_configs = ['d12', 'd16', 'd20', 'd24']

    all_results = {}
    for config_name in model_configs:
        try:
            result = tune_model(config_name)
            all_results[config_name] = result
        except Exception as e:
            print(f"failed to tune {config_name}: {e}")

    # summary
    print(f"\n{'='*60}")
    print("summary")
    print(f"{'='*60}")
    for config_name, result in all_results.items():
        if result:
            print(f"{config_name}: block_q={result['block_q']}, "
                  f"block_compute={result['block_compute']}, "
                  f"tokens/sec={result['tokens_per_sec']:.0f}")
