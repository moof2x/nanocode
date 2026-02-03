"""
ARC dataset https://huggingface.co/datasets/allenai/ai2_arc
"""

from datasets import load_dataset
from tasks.common import render_mc


class ARC:
    eval_type = 'categorical'

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["ARC-Easy", "ARC-Challenge"], "arc subset must be arc-easy or arc-challenge"
        assert split in ["train", "validation", "test"], "arc split must be train|validation|test"
        self.ds = load_dataset("allenai/ai2_arc", subset, split=split).shuffle(seed=42)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        question = row["question"]
        choices = row["choices"]["text"]
        answer_string = row["answerKey"]
        letters = row["choices"]["label"]
        assert answer_string in letters, f"arc answer {answer_string} must be one of {letters}"
        user_message = render_mc(question, letters, choices)
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer_string}
        ]
        conversation = {
            "messages": messages,
            "letters": letters,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        assert assistant_response in conversation['letters'], f"arc answer {assistant_response} is expected to be one of {conversation['letters']}"
        assistant_message = conversation['messages'][-1]['content']
        return assistant_response == assistant_message
