from tokenizer import Tokenizer,ChatFormat
from model import Transformer
from typing import List
import torch
import torch.nn.functional as F

class generator:
    def __init__(
            self,
            model:Transformer,
            tokenizer:Tokenizer,
    ):
        self.model=model
        self.tokenizer = tokenizer
        self.formatter = ChatFormat(tokenizer)
        
    @torch.inference_mode()
    def generate(
            self,
            prompt_tokens: List[List[int]],
            max_gen_len: int,
            temperature: float = 0.6,
            top_p: float = 0.9,
            logprobs: bool = False,
            echo: bool = False,
        ):
        
        params = self.model.args
        bsz = len(prompt_tokens)

        assert bsz <= params.max_batch_size,(bsz,params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        
        assert max_prompt_len <= params.max_seq_len

        total_len = min(params.max_seq_len,max_gen_len + max_prompt_len)
        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz,total_len),pad_id,dtype=torch.long,device="cpu")
        
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cpu")
        
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz,device="cpu")
        input_text_mask = tokens != pad_id
        
        if min_prompt_len == total_len:
            logits = self.model.forward(tokens,prev_pos)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1,2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id
            )

        stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens))

        for cur_pos in range(min_prompt_len,total_len):
            logits = self.model.forward(tokens[:,prev_pos:cur_pos],prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:,-1] / temperature,dim=-1)
                #only replace token if prompt has already been generated
                next_token = sample_top_p(probs,top_p)
            else:
                next_token = torch.argmax(logits[:,-1],dim=-1)
            
            next_token = next_token.reshape(-1)

            next_token = torch.where(
                input_text_mask[:,cur_pos],tokens[:,cur_pos],next_token
            )

            tokens[:,cur_pos] = next_token
            if logprobs:
                token_logprobs[:,prev_pos+1:cur_pos+1] = -F.cross_entropy(
                    input=logits.transpose(1,2),
                    target=tokens[:,prev_pos+1:cur_pos+1],
                    reduction="none",
                    ignore_index=pad_id
                )
            eos_reached |= (~input_text_mask[:,cur_pos]) & (
                torch.isin(next_token,stop_tokens)
                )
            
            prev_pos = cur_pos
            if all(eos_reached):
                break
        
        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens,out_logprobs = [],[]

        for i, toks in enumerate(tokens.tolist()):
            #cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start:len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start:len(prompt_tokens[i] + max_gen_len)]
            #cut to after eos tok if any
            for stop_token in self.tokenizer.stop_tokens:
                try:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx]
                    probs = probs[:eos_idx]
                    probs = probs[:eos_idx] if logprobs else None
                except ValueError:
                    pass
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens,out_logprobs if logprobs else None)

def sample_top_p(probs,p):
    probs_sort,probs_idx = torch.sort(probs,dim=-1,descending=True)
    probs_sum = torch.cumsum(probs_sort,dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1,keepdim=True))
    next_token = torch.multinomial(probs_sort,num_samples=1)
    next_token = torch.gather(probs_idx,-1,next_token)
    return next_token