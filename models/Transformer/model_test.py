from tokenizer import Tokenizer
from model import Transformer,ModelAgrs,TransformerForCausalLM
from generation import generator
import torch


tok = Tokenizer("o200k_base")
prompt = ["안녕하세요. 최용원입니다.","만나서 반갑습니다.","오늘 날씨가 참 좋네요."]
prompt_tokens = tok.encode_batch(prompt,bos=True,eos=True)
print(prompt_tokens)
min_prompt_len = min(len(t) for t in prompt)
max_prompt_len = max(len(t) for t in prompt)

# device = torch.device("cpu")
# model = Transformer(ModelAgrs)
# model.to(device)

# _model = TransformerForCausalLM(model,tok)
# output = _model.generate(prompt_tokens,30)
# print(output)