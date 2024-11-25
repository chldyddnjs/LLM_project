from tokenizer import Tokenizer
from model import Transformer,ModelAgrs,TransformerForCausalLM
from generate import generator
import torch


tok = Tokenizer("o200k_base")
prompt = ["안녕하세요. 최용원입니다.","만나서 반갑습니다.","오늘 날씨가 참 좋네요."]
prompt_tokens = tok.encode_batch(prompt,bos=True,eos=True)

min_prompt_len = min(len(t) for t in prompt)
max_prompt_len = max(len(t) for t in prompt)

print(min_prompt_len)
print(max_prompt_len)
device = torch.device("cpu")
model = Transformer(ModelAgrs)
model.to(device)

# print(tok.n_words)
_model = TransformerForCausalLM(model,tok)
output = _model.generate(prompt_tokens,30)
print(output)