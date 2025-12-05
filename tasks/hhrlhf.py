"""
This is a 'sft-ed' version of Anthtropic's hh-rlhf preference dataset.
https://huggingface.co/datasets/Anthropic/hh-rlhf
We're training on the "chosen" conversations which I've pre-converted to
the messages format and filtered for some invalid rows.
"""

from datasets import load_dataset

class HHRLHF:
    """
    train is 160k rows, test is 8.53k rows
    """
    def __init__(self, split: str, seed: int, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "split must be one of train|test"
        self.ds = load_dataset("smohammadi/hh-rlhf", split=split).shuffle(seed=seed)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        return {
            "messages": self.ds[idx]["messages"]
        }

