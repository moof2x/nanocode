"""
Taken from karparthy/nanochat/nanochat/ and stripped of the HF tokenizer. 
"""

import copy
import pickle
from functools import lru_cache
from pathlib import Path

import rustbpe
import tiktoken

SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>", # user messages
    "<|user_end|>",
    "<|assistant_start|>", # assistant messages
    "<|assistant_end|>",
    "<|tool_call_start|>",
    "<|tool_arg|>",
    "<|tool_val|>",
    "<|tool_call_end|>",
    "<|tool_result_start|>",
    "<|tool_result_end|>",
]

# NOTE: this split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}
# I did this because I didn't want to "waste" too many tokens on numbers for smaller vocab sizes.
# I haven't validated that this is actually a good idea, TODO.
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RustBPETokenizer:
    """Light wrapper around tiktoken (for efficient inference) but train with rustbpe"""

    def __init__(self, enc, bos_token):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # 1) train using rustbpe
        tokenizer = rustbpe.Tokenizer()
        # the special tokens are inserted later in __init__, we don't train them here
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        # 2) construct the associated tiktoken encoding for inference
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks, # dict[bytes, int] (token bytes -> merge priority rank)
            special_tokens=special_tokens, # dict[str, int] (special token name -> token id)
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir: Path):
        with open(tokenizer_dir / "tokenizer.pkl", "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        # https://github.com/openai/tiktoken/blob/eedc8563/tiktoken_ext/openai_public.py
        enc = tiktoken.get_encoding(tiktoken_name)
        # tiktoken calls the special document delimiter token "<|endoftext|>"
        # yes this is confusing because this token is almost always PREPENDED to the beginning of the document
        # it most often is used to signal the start of a new sequence to the LLM during inference etc.
        # so in nanoChat we always use "<|bos|>" short for "beginning of sequence", but historically it is often called "<|endoftext|>".
        return cls(enc, "<|endoftext|>")

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        # text can be either a string or a list of strings

        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id) # TODO: slightly inefficient here? :( hmm
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id) # TODO: same
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

        return ids

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.enc.decode(ids)

    def save(self, tokenizer_dir: Path):
        # save the encoding object to disk
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        pickle_path = tokenizer_dir / "tokenizer.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")
    
    def render_conversation(self, conversation, max_tokens=2048):
        """
        Tokenize a single Chat conversation (which we call a "doc" or "document" here).
        Returns:
        - ids: list[int] is a list of token ids of this rendered conversation
        - mask: list[int] of same length, mask = 1 for tokens that the Assistant is expected to train on.
        """
        # ids, masks that we will return and a helper function to help build them up.
        ids, mask = [], []
        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # sometimes the first message is a system message...
        # => just merge it with the second (user) message
        if conversation["messages"][0]["role"] == "system":
            # some conversation surgery is necessary here for now...
            conversation = copy.deepcopy(conversation) # avoid mutating the original
            messages = conversation["messages"]
            assert messages[1]["role"] == "user", "System message must be followed by a user message"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        # fetch all the special tokens we need
        bos = self.get_bos_token_id()
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
        tool_call_start, tool_call_end = self.encode_special("<|tool_call_start|>"), self.encode_special("<|tool_call_end|>")
        tool_result_start, tool_result_end = self.encode_special("<|tool_result_start|>"), self.encode_special("<|tool_result_end|>")
        tool_arg, tool_val = self.encode_special("<|tool_arg|>"), self.encode_special("<|tool_val|>")        
        
        # now we can tokenize the conversation
        add_tokens(bos, 0)
        for i, message in enumerate(messages):
            # some sanity checking here around assumptions, to prevent footguns
            # must_be_from = ["user", "tool_result"] if i % 2 == 0 else "assistant"
            # must_be_from = "assistant" if i > 0 and messages[i-1]["role"] in ["user", "tool_result"] else ["user", "tool_result"]
            must_be_from = ["user", "assistant", "tool_result"] if i > 0 else ["user", "tool_result"]
            assert message["role"] in must_be_from, f"Message {i} is from {message['role']} but should be one of {must_be_from}"


            if message["role"] == "user":
                add_tokens(user_start, 0)
                add_tokens(self.encode(message["content"]), 0)
                add_tokens(user_end, 0)
                
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if "tool_call" in message:
                    if "content" in message:
                        add_tokens(self.encode(message["content"]), 1)
                    # assert "content" not in message, f"Assistant tool call message has a 'content' entry: '{message['content']}'"

                    tool_call = message["tool_call"]
                    # ensure the tool call is correctly formatted
                    assert "name" in tool_call
                    assert "args" in tool_call and isinstance(tool_call["args"], dict), f"Expected tool call arguments as 'args': {'arg_1': val_1, 'arg_2': val_2}, but found {tool_call['args']}"
                    add_tokens(tool_call_start, 1)
                    add_tokens(self.encode(tool_call["name"]), 1)
                    for arg, val in tool_call["args"].items():
                        # <|tool_arg|>{arg}<|tool_val|>{val}
                        add_tokens(tool_arg, 1) 
                        add_tokens(self.encode(arg), 1)
                        add_tokens(tool_val, 1)
                        add_tokens(self.encode(str(val)), 1)
                    add_tokens(tool_call_end, 1)
                else:        
                    add_tokens(self.encode(message["content"]), 1)
                add_tokens(assistant_end, 1)
    
            elif message["role"] == "tool_result":

                add_tokens(tool_result_start, 0)
                add_tokens(self.encode(message["content"]), 0)
                add_tokens(tool_result_end, 0)
            else:
                raise ValueError(f"Unsupported role: {message['role']}")

        # truncate to max_tokens tokens MAX (helps prevent OOMs)
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def visualize_tokenization(self, ids, mask, with_token_id=False):
        """Small helper function useful in debugging: visualize the tokenization of render_conversation"""
        RED = '\033[91m'
        GREEN = '\033[92m'
        RESET = '\033[0m'
        GRAY = '\033[90m'
        tokens = []
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return '|'.join(tokens)

    def render_for_completion(self, conversation):
        """
        Used during Reinforcement Learning. In that setting, we want to
        render the conversation priming the Assistant for a completion.
        Unlike the Chat SFT case, we don't need to return the mask.
        """
        # We have some surgery to do: we need to pop the last message (of the Assistant)
        conversation = copy.deepcopy(conversation) # avoid mutating the original
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", "Last message must be from the Assistant"
        messages.pop() # remove the last message (of the Assistant) inplace

        # Now tokenize the conversation
        ids, mask = self.render_conversation(conversation)

        # Finally, to prime the Assistant for a completion, append the Assistant start token
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids

# -----------------------------------------------------------------------------

def get_tokenizer():
    from nanocode.common import get_model_dir
    return RustBPETokenizer.from_directory(get_model_dir() / "tokenizer")

def get_token_bytes():
    import zarr

    from nanocode.common import get_model_dir
    tokenizer_dir = get_model_dir() / "tokenizer"
    token_bytes_path = tokenizer_dir / "token_bytes.zarr"
    
    assert token_bytes_path.exists(), f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    token_bytes = zarr.load(token_bytes_path)
    return token_bytes
