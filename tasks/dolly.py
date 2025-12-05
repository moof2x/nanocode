"""
A high quality instruction-following dataset.
https://huggingface.co/datasets/databricks/databricks-dolly-15
"""

from datasets import load_dataset

class Dolly:
    """ 10K rows after excluding information extraction esque tasks."""
    def __init__(self, seed: int, **kwargs):
        super().__init__(**kwargs)
        self.ds = load_dataset("databricks/databricks-dolly-15k", split="train").shuffle(seed=seed)
        self.ds = self.ds.filter(lambda a: not a["context"])

    def __len__():
        return len(self.ds)

    def __getitem__(self, idx: int):
        row = self.ds[idx]
        messages = [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["response"]}            
        ]
        return {
            "messages": messages
        }


