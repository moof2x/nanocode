
"""
smol-SmolTalk from HuggingFace. I've stripped the validation logic from nanochat's implementation.
"""

from datasets import load_dataset

class SmolTalk:
    """ smol-smoltalk dataset. train is 460K rows, test is 24K rows. """

    def __init__(self, split: str, seed: int, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SmolTalk split must be train|test"
        self.ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=split).shuffle(seed=seed)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx: int):
        return {
            "conversation": self.ds[idx]["messages"]
        }
