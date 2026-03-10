"""
Generic dataset loader for Hugging Face datasets.
"""

from datasets import load_dataset


class HuggingFaceDataset:
    """
    Generic Hugging Face dataset loader.
    """

    def __init__(self, dataset : str, messages_key: str, split: str, seed: int,  **kwargs):
        self.messages_key = messages_key
        self.ds = load_dataset(dataset, split=split, **kwargs).shuffle(seed=seed)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        return {"messages": self.ds[idx][self.messages_key]}


