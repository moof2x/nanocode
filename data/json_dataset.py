
"""
Dataset classes for loading custom JSONL formatted datasets.
"""

import json
import random
from data.common import SYSTEM_PROMPT


class JSONDataset:
    """
    Load conversations from a JSONL file.
    Each line should be a JSON object with a 'messages' field.
    Example line: {"messages": [{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"}]}
    """

    def __init__(self, filepath, seed, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.ds= []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    pass
                messages = json.loads(line)
                self.ds.append(messages)
        random.Random(seed).shuffle(self.ds)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        messages = self.ds[idx]["messages"]
        # prepend the nanocode system prompt
        # change or remove this for your own use case
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        return {"messages": messages}


class JSONPreferenceDataset:
    """
    Load preference pairs from a JSONL file.
    Each line should be a JSON object with 'chosen' and 'rejected' keys,
    each containing a dict with a 'messages' field.
    Example line: {"chosen": {"messages": [...]}, "rejected": {"messages": [...]}}
    """

    def __init__(self, filepath, seed, frac=1.0, **kwargs):
        super().__init__(**kwargs)
        self.ds = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.ds.append(json.loads(line))
        random.Random(seed).shuffle(self.ds)
        self.ds = self.ds[:int(len(self.ds) * frac)]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        row = self.ds[idx]
        # prepend the nanocode system prompt to chosen and rejected
        # change or remove this for your own use case
        chosen = [{"role": "system", "content": SYSTEM_PROMPT}] + row["chosen"]["messages"]
        rejected = [{"role": "system", "content": SYSTEM_PROMPT}] + row["rejected"]["messages"]
        return {"messages": chosen}, {"messages": rejected}


class PairedJSONPreferenceDataset:
    """
    Load preference pairs from a JSONL file, filtering out pairs where
    either side has no assistant message with content.
    Each line should be a JSON object with 'chosen' and 'rejected' keys,
    each containing a dict with a 'messages' field.
    Example line: {"chosen": {"messages": [...]}, "rejected": {"messages": [...]}}
    """

    def __init__(self, filepath, seed, frac=1.0, **kwargs):
        super().__init__(**kwargs)
        self.ds = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    # filter out pairs where either side has no assistant content
                    has_chosen = any(m.get("content", "") for m in row["chosen"]["messages"] if m["role"] == "assistant")
                    has_rejected = any(m.get("content", "") for m in row["rejected"]["messages"] if m["role"] == "assistant")
                    if has_chosen and has_rejected:
                        self.ds.append(row)
        random.Random(seed).shuffle(self.ds)
        self.ds = self.ds[:int(len(self.ds) * frac)]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        row = self.ds[idx]
        # prepend the nanocode system prompt to chosen and rejected
        # change or remove this for your own use case
        chosen = [{"role": "system", "content": SYSTEM_PROMPT}] + row["chosen"]["messages"]
        rejected = [{"role": "system", "content": SYSTEM_PROMPT}] + row["rejected"]["messages"]
        return {"messages": chosen}, {"messages": rejected}

