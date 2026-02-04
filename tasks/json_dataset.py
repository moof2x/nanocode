
"""
Dataset class for loading custom JSONL formatted datasets.
Each line in the JSONL file should be a JSON array of messages.
"""

import json

class JSONDataset:
    """
    Load conversations from a JSONL file.
    Each line should be a JSON array of message objects with 'role' and 'content' fields.
    Example line: [{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"}]
    """

    def __init__(self, filepath, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.ds= []
        
        # TODO allow users to download from huggingface
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    pass
                messages = json.loads(line)

                # this would be a good place to put your own validation logic here
                # I like to pre-process and validate my datasets so nanojax's soul
                # dataset should work OOTB
                self.ds.append(messages)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        return {
            "messages": self.ds[idx]["messages"],
        }

