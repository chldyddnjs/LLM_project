import os
import numpy as np
from datasets import load_dataset
from models.llava.llava_train import preprocess
from transformers import AutoTokenizer

def test_preprocess():
    ds = load_dataset("data/dacon",num_proc=os.cpu_count()-1,split="train[:]")
    idx = 8
    prompt = ds[idx]['conversations']
    tokenizer = AutoTokenizer.from_pretrained("upstage/TinySolar-248m-4k")
    res = preprocess(prompt,tokenizer)
    print(res['input_ids'][0])

if __name__=="__main__":
    test_preprocess()