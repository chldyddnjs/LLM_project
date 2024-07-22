import os
from logging import getLogger
from pathlib import Path
import functools
from concurrent.futures import ThreadPoolExecutor

from typing import (
    AbstractSet,
    cast,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Sequence,
    TypedDict,
    Union,
)
import tiktoken


logger = getLogger(__name__)

Role = Literal["system","user","assistant"]

class Message(TypedDict):
    role: Role
    content: str

Dialog = Sequence[Message]
Dialogs = Sequence[Dialog] 

class Tokenizer:
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer
    """

    special_tokens: Dict[str,int]

    num_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self,model_name:str):
        """
        Initializes the Tokenizer with a Tiktoken model

        Args:
            model_name(str): The name in the tiktoken
            Check the below 
            https://github.com/openai/tiktoken/blob/c0ba74c238d18b4824c25f3c27fc8698055b9a76/tiktoken/model.py#L7
            https://github.com/openai/tiktoken/blob/c0ba74c238d18b4824c25f3c27fc8698055b9a76/tiktoken/model.py#L20
        """
        assert type(model_name)==str
        
        #TODO
        #vocab.model 파일이 없으므로 tiktoken에서 제공하는 메소드를 기본으로 사용하도록 바꿈
        model_base = tiktoken.get_encoding(model_name)

        mergeable_ranks = model_base._mergeable_ranks
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>", #end of turn
        ] + [
            f"<|reserved_special_token_{i}>"
            for i in range(5,self.num_reserved_special_tokens - 5)
        ]
        
        self.special_tokens = {
            token: num_base_tokens + i for i , token in enumerate(special_tokens)
        }
        
        self.model = tiktoken.Encoding(
            # If you're changing the set of special tokens, make sure to use a different name
            # It should be clear from the name what behaviour to expect.
            name=f"{model_name}_llama3",
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens
        )

        logger.info(f"Reloaded tiktoken model from {model_name}")

        self.n_words: int = self.model.n_vocab
        #BOS / EOS token IDs
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]
        self.pad_id: int = -1
        self.stop_tokens = {
            self.special_tokens["<|end_of_text|>"],
            self.special_tokens["<|eot_id|>"]
        }
        logger.info(
            f"#words: {self.n_words} - BOS ID : {self.bos_id} - EOS ID: {self.eos_id}"
        )
    
    def encode(
        self,
        s: str,
        *,
        bos:bool,
        eos:bool,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = ()
    ) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): Theinput string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token
            allowed_tokens ("all"|set[str]): allowed special tokens in string
            disallowed_tokens ("all|set[str]"): special tokens that raise an error when in string
        
        Returns:
            list[int]: A list of token IDs.
        
        By default, setting disallowed_special=() encodes a string by ignoring special tokens. Specifically:
        - Setting `disallowed_special` to () will cause all text corresponding to special tokens to be encoded as natural text (insteading of raising an error)
        - Setting `allowed_special` to "all" will treat all text corresponding to special tokens to be encoded as special tokens.
        
        """

        assert type(s) is str

        #The tiktoken tokenizer can handle <= 400k chars without
        #pyo3_runtime.PanicExeption
        TIKTOKEN_MAX_ENCODE_CHARS = 400_000

        # https://github.com/openai/tiktoken/issues/195
        # Here we iterate over subsequences and split if we exceed the limit
        # of max consecutive non-whitespace or whitespace characters.
        MAX_NO_WHITESPACES_CHARS = 25_000

        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )

        t: List[int] = []
        for substr in substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def encode_batch(
        self,
        text: list[str],
        *,
        bos:bool,
        eos:bool,
        num_threads: int = 8,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),  # noqa: B006
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
    ) -> list[list[int]]:
        """Encodes a list of strings into tokens, in parallel.

        See `encode` for more details on `allowed_special` and `disallowed_special`.

        ```
        >>> enc.encode_batch(["hello world", "goodbye world"])
        [[31373, 995], [11274, 16390, 995]]
        ```
        """

        encoder = functools.partial(
            self.encode, bos=bos, eos=eos,allowed_special=allowed_special, disallowed_special=disallowed_special
        )
        with ThreadPoolExecutor(num_threads) as e:
            return list(e.map(encoder, text))

    def decode(self,t:Sequence[int]) -> str:
        """
            Decodes a list of token IDs into a string

            Args:
                t (List[int]): The list of token Ids to be decoded

            Returns:
                str: Tge decoded string
        """

        #Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence
        return self.model.decode(cast(List[int],t))
    
    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
        s: str,max_consecutive_slice_len: int
    ) -> Iterator[str]:

        """
            Splits the string `s` so that each suvstring contain no more than `max_consecutive_slice_len`
            consecutive whitespaces or consecutive non-whitespaces.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1

        yield s[slice_start:]

class ChatFormat:
    def __init__(self, tokenizer:Tokenizer):
        self.tokenizer = tokenizer
    
    def encode_header(self,message:Message) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        tokens.extend(self.tokenizer.encode(message['role'], bos=False,eos=False))
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        tokens.extend(self. tokenizer.encode("\n\n",bos=False,eos=False))
        return tokens

    def encode_message(self, message: Message) -> List[int]:
        tokens = self.encode_header(message)
        tokens.extend(
            self.tokenizer.encode(message["content".strip()],bos=False,eos=False)
        )
        tokens.append(self.tokenizer.special_tokens["<|eot_id|>"])
        return tokens
    
    def encode_dialog_prompt(self,dialog:Dialog) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|begin_of_text|>"])
        for message in dialog:
            tokens.extend(self.encode_message(message))
        #Add the start of an assistant message for the model to complete.
        tokens.extend(self.encode_header({"role":"assistant", "content":""}))
        return tokens
    
    def encode_dialog_prompt_batch(self,dialogs:Dialogs) -> List[int]:
        batch = [self.encode_dialog_prompt(dialog) for dialog in dialogs]
        return batch 