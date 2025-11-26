from nanojax.gpt import GPTConfig

# this is roughly the GPT2-small hyper-params
# you'd end must closer to the original 117M param count without tied weights 
d12_162m = GPTConfig() # ~162M
d6_23m = GPTConfig(n_layer=6, n_embed=384, n_head=3, n_kv_head=3, vocab_size=32000, sequence_len=512) # ~23M
d3_4m = GPTConfig(n_layer=3, n_embed=192, n_head=2, n_kv_head=2, vocab_size=8000, sequence_len=256) # 4M, but at this size the model is dominated by the embedding and classifier weights
