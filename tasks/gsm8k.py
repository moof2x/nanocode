"""
GSM8K https://huggingface.co/datasets/openai/gsm8k
"""

import re
from datasets import load_dataset


GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def extract_answer(completion):
    # extract the numerical answer after #### marker.
    match = GSM_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return None


class GSM8K:
    eval_type = 'generative'

    def __init__(self, subset, split, seed, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["main", "socratic"], "gsm8k subset must be main|socratic"
        assert split in ["train", "test"], "gsm8k split must be train|test"
        self.ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=seed)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        parts = re.split(r'(<<[^>]+>>)', row["answer"])
        messages = [{"role": "user", "content": row["question"]}]
        content = ""
        
        # we're going to split this into our nanojax-agent tool calling format
        for part in parts:
            if part.startswith('<<') and part.endswith('>>'):
                inner = part[2:-2] # strip << >>
                expr, result = inner.rsplit("=", 1)
                tool_call = {
                    "role": "assistant",                    
                    "content": content.lower(), 
                    "tool_call": {
                        "name": "Bash",
                        "args": {"command": f"python3 -c 'print({expr})'"}
                    }
                    
                }
                tool_result = {
                    "role": "tool_result",
                    "content": result.strip()
                }
                messages.append(tool_call)
                messages.append(tool_result)
                content = ""
            else:
                # accumulate the model's thinking 
                content += part
        if content:
            messages.append({
                "role": "assistant",
                "content": content.lower().strip()                    
            })
        return {"messages": messages}

    def evaluate(self, conversation, assistant_response):
        assert isinstance(assistant_response, str), "assuming simple string response for now"
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant", "last message must be from the assistant"
        assert isinstance(assistant_message['content'], list), "this is expected to be a list of parts"
        last_text_part = assistant_message['content'][-1]['text']
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)
        is_correct = int(pred_num == ref_num)
        return is_correct
