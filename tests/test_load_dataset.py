import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer,CLIPImageProcessor
from models.llava.llava_train import LazySupervisedDataset

def test_dacon_load():
    ds = load_dataset("data/dacon",split="train[:10]")
    idx = 8
    print(ds['image'][idx])
    print(np.array(ds['image'][idx]).shape)

def test_load_lazydataset():
    tok = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.3")
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14") 
    ds = LazySupervisedDataset("data/dacon",tok,processor,split="train[:10]")
    print(next(iter(ds)))
    print(ds.__len__())

if __name__=="__main__":
    # test_dacon_load()
    test_load_lazydataset()