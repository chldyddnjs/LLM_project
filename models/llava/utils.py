import torch
# from .constants import IMAGE_TOKEN_INDEX
from transformers import AutoTokenizer
from .constants import IMAGE_TOKEN_INDEX

def print_nparams(model):
    """Calculate the total number of model parameters"""
    nparams = sum(p.numel() for p in model.parameters())
    print(f"The total number of parameters is: {nparams}")
    
def tokenizer_image_token(prompt,tokenizer,image_token_index=IMAGE_TOKEN_INDEX):
    """ 
    Return: Inserted image token in the prompt and deduplicate the tokens both bos and eos 
    """
    
    #프롬프트를 <image> 기준으로 분할하고 각 부분을 토큰화
    #이렇게 해도 되는 이유는 이미지는 맨 앞에 있기 때문임
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    #구분자 삽입
    def insert_separator(X,sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])
    
    #토큰과 이미지 토큰 결합
    for x in insert_separator(prompt_chunks,[image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    return torch.tensor(input_ids,dtype=torch.long)

if __name__=="__main__":
    prompt = "<image> the human is three <image> the human is three <image> the human is three"
    tokenizer = AutoTokenizer.from_pretrained('lmsys/vicuna-7b-v1.3')
    res = tokenizer_image_token(prompt,tokenizer)
    print(res)