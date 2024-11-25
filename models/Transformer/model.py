import math
import torch
import torch.nn.functional as F
from torch import nn
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class ModelAgrs:
    dim:int = 512
    n_layers:int = 4
    n_heads:int = 4
    n_kv_heads:Optional[int]=None
    vocab_size:int = 200254
    multiple_of:int=256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000

    max_batch_size: int = 32
    max_seq_len: int = 512
    device:str = "cpu"

class LayerNorm(nn.Module):
    def __init__(self,dim:int,eps:float=1e-6):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(dim))
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self,x:torch.tensor):
        mean = x.mean(-1,keepdim=True)
        std = x.std(-1,keepdim=True)
        return self.gamma*mean/(std+self.eps) + self.beta

def repeat_kv(x:torch.tensor,n_rep:int) -> torch.Tensor:
    """torch.repeat_interleave(x,dim=2,repeats=n_rep)"""
    bs,slen,n_kv_heads,head_dim = x.shape
    if n_rep == 1:
        return x
    
    return (
        x[:,:,:,None,:]
        .expand(bs,slen,n_kv_heads,n_rep,head_dim)
        .reshape(bs,slen,n_kv_heads * n_rep,head_dim)
    )

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2) /dim).float())
    t = torch.arange(end, device=freqs.device, dtype=torch.float32).unsqueeze(dim=1)
    freqs_cis = torch.zeros(end, dim, device=freqs.device)
    freqs_cis.requires_grad=False
    freqs_cis[:,0::2] = torch.sin(t*freqs)
    freqs_cis[:,1::2] = torch.cos(t*freqs)

    return freqs_cis

class Attention(nn.Module):
    def __init__(self,args:ModelAgrs):
        super().__init__()

        model_parallel_size = torch.distributed.get_world_size()
        
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = args.n_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim,args.n_heads * self.head_dim,bias=False)
        self.wk = nn.Linear(args.dim,args.n_heads * self.head_dim,bias=False)
        self.wv = nn.Linear(args.dim,args.n_heads * self.head_dim,bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim,bias=False)
        
        
        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim
            )#cuda
        )
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim
            )#cuda
        )
    def forward(
            self,
            x:torch.Tensor,
            start_pos:int,
            mask: Optional[torch.Tensor],
    ):
        bsz,seqlen, _ = x.shape
        q,k,v = self.wq(x),self.wk(x),self.wv(x)

        q = q.view(bsz,seqlen,self.n_local_heads,self.head_dim)
        k = k.view(bsz,seqlen,self.n_local_kv_heads,self.head_dim)
        v = v.view(bsz,seqlen,self.n_local_kv_heads,self.head_dim)
        '''
            if you want to use the rotary_emb
                write down it
        '''
        self.cache_k = self.cache_k.to(q)
        self.cache_v = self.cache_v.to(q)

        self.cache_k[:bsz,start_pos:start_pos + seqlen] = k
        self.cache_v[:bsz,start_pos:start_pos + seqlen] = v

        keys = self.cache_k[:bsz,:start_pos + seqlen]
        values = self.cache_v[:bsz,:start_pos + seqlen]

        #repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys,self.n_rep
        )
        values = repeat_kv(
            values,self.n_rep
        )

        q = q.transpose(1,2) # (bs, n_local_heads, seqlen)
        keys = keys.transpose(1,2) #(bs, n_local_heads, cache+seqlen, head_dim)
        values = values.transpose(1,2) #(bs,n_local_heads,cache_len + seqlen, head_dim)

        scores = torch.matmul(q,keys.transpose(2,3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask # (bs, n_local_heads,seqlen,cache_len + seqlen)
        scores = F.softmax(scores.float(),dim=-1).type_as(q)
        output = torch.matmul(scores,values)
        output = output.transpose(1,2).contiguous().view(bsz,seqlen,-1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim:int,
        hidden_dim:int
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        
        self.w1 = nn.Linear(dim,hidden_dim)
        self.w2 = nn.Linear(hidden_dim,dim)
        self.w3 = nn.Linear(dim,hidden_dim)

    def forward(self,x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    
class TransformerBLock(nn.Module):
    def __init__(self,layer_id:int,args:ModelAgrs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4*args.dim
        )
        self.layer_id = layer_id
        self.attention_norm = LayerNorm(args.dim,eps=args.norm_eps)
        self.ffn_norm = LayerNorm(args.dim,eps=args.norm_eps)

    def forward(
        self,
        x:torch.Tensor,
        start_pos:int,
        mask:Optional[torch.Tensor]=None,
        ):
        h = x + self.attention(self.attention_norm(x),start_pos,mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
    
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class Transformer(nn.Module):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head":"colwise_rep"}

    def __init__(self,args:ModelAgrs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        self.tok_embeddings = nn.Embedding(args.vocab_size,args.dim)
        self.encoder_layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.encoder_layers.append(TransformerBLock(layer_id,args))

        self.decoder_layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.decoder_layers.append(TransformerBLock(layer_id,args))
        
        self.norm = LayerNorm(args.dim,eps=args.norm_eps)
    
        self.freqs_cis = precompute_freqs_cis(
            args.dim,
            args.max_seq_len,
            args.rope_theta,
        )
        self.lm_head = nn.Linear(
            args.dim,args.vocab_size,bias=False
        )
    
    def forward(self,tokens:torch.Tensor,start_pos:int):
        _bsz, seqlen = tokens.shape
        hidden_state = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis[:seqlen,:].to(hidden_state.device)
        print(hidden_state.size(),self.freqs_cis.size())
        hidden_state+=self.freqs_cis
        # freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen,seqlen),float("-inf"),device=tokens.device)
            mask = torch.triu(mask,diagonal=1)
            mask = torch.hstack(
                [torch.zeros((seqlen,start_pos),device=tokens.device),mask]
            ).type_as(hidden_state)

        for layer in self.encoder_layers:
            encoder_hidden_state = layer(hidden_state,start_pos)
        
        for layer in self.decoder_layers:
            decoder_hidden_state = layer(encoder_hidden_state,start_pos,mask)
        
        last_hidden_state = self.norm(decoder_hidden_state)
        return self.lm_head(last_hidden_state)
    
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def get_input_embeddings(self):
        return self.tok_embeddings
    
    def set_input_embeddings(self,value):
        self.tok_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self,new_embeddings):
        self.lm_head = new_embeddings
    
    def get_decoder(self):
        return self.model
    
    def set_decoder(self,decoder):
        self.model = decoder

class TransformerForCausalLM:
    def __init__(
            self,
            model:Transformer,
            tokenizer,
    ):
        self.model=model
        self.tokenizer = tokenizer
        
    def generate(
            self,
            prompt_tokens,
            max_gen_len: int,
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
        tokens = torch.full((bsz,total_len),pad_id,dtype=torch.long,device=params.device)
        
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=params.device)
        
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz,device=params.device)
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

if __name__=="__main__":
    pe = precompute_freqs_cis(512,30)
    print(pe.size())
    args = ModelAgrs()
    emb = nn.Embedding(args.vocab_size,args.dim)