from nanojax.gpt import GPTConfig

# nanochat's d20 model, ~560m
d20 = GPTConfig(n_layer=20, n_embed=1280, n_head=5, n_kv_head=5, vocab_size=65536, sequence_len=2048)
# this is roughly the GPT2-small hyper-params
# you'd end must closer to the original 117M param count without tied weights 
d12 = GPTConfig() # ~162M
d6 = GPTConfig(n_layer=6, n_embed=384, n_head=3, n_kv_head=3, vocab_size=32000, sequence_len=512) # ~23M
d3 = GPTConfig(n_layer=3, n_embed=192, n_head=2, n_kv_head=2, vocab_size=8000, sequence_len=256) # 4M, but at this size the model is dominated by the embedding and classifier weights
