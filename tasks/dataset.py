"""
Generic dataset loaders for huggingface datasets.
"""

from datasets import load_dataset


class Dataset:
    """
    Generic dataset loader.
    """

    def __init__(self, dataset : str, messages_key: str, split: str, seed: int,  **kwargs):
        self.messages_key = messages_key
        self.ds = load_dataset(dataset, split=split, **kwargs).shuffle(seed=seed)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        return {"messages": self.ds[idx][self.messages_key]}


class PreferenceDataset:
    """
    Generic preference dataset loader.
    """

    def __init__(self, dataset: str, chosen_messages_key : str, rejected_messages_key: str, split: str, seed: int, **kwargs):
        super().__init__(**kwargs)
        self.chosen_messages_key = chosen_messages_key
        self.rejected_messages_key = rejected_messages_key
        self.ds = load_dataset(dataset, split=split).shuffle(seed=seed)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        return {"messages": self.ds[idx][self.chosen_messages_key]}, {"messages": self.ds[idx][self.rejected_messages_key]}


class JSONPreferenceDataset:
    """
    Load preference pairs from a JSONL file.
    Each line should be a JSON object with 'chosen' and 'rejected' keys,
    each containing a dict with a 'messages' field.
    """

    def __init__(self, filepath, **kwargs):
        import json
        super().__init__(**kwargs)
        self.ds = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.ds.append(json.loads(line))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        row = self.ds[idx]
        return row["chosen"], row["rejected"]

