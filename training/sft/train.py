"""
    Code based by https://github.com/xyjigsaw/LLM-Pretrain-SFT/blob/master/llm_pretrain/pretrain.py
"""

import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Dict,Optional,Sequence
import wandb

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from datasets import load_dataset
from transformers import (Trainer,
                          TrainingArguments,
                          PreTrainedTokenizer,
                          PreTrainedModel,
                          HfArgumentParser,
                          AutoTokenizer,
                          AutoModelForCausalLM
                          )

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

@dataclass
class DataArguments:
    data_path: str = field(default="nayohan/raw_instruction_en_ko_translation")

@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    output_dir: Optional[str] = field(default='../output')
    optim:str = field(default="adamw_torch")
    learning_rate = 2e-5,
    per_device_train_batch_size = 16
    per_device_train_batch_size = 16
    num_train_epochs = 3
    weight_decay = 0.01,
    eval_strategy = "epoch",
    save_strategy = "epoch",
    load_best_model_at_end=True,
    model_max_length: int = field(default=512)
    use_lora: bool = field(default=False)

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is th unoptimized version that may make your embedding size not be divisible by 64
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0,keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0,keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def _tokenize_fn(examples:Sequence[str], tokenizer: PreTrainedTokenizer):
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            example,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True
        ) for example in examples ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: PreTrainedTokenizer,
) -> Dict:
    """Preproces the data by tokenizing"""
    examples = [s + t for s,t in zip(sources,targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(example,tokenizer) for example in (examples,sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels,sources_tokenized)
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids,labels=labels)

class SupervisedDataset(Dataset):
    """Dataset for superviesed fine-tuning"""

    def __init__(self, data_path:str, tokenizer: PreTrainedTokenizer)
        super(SupervisedDataset,self).__init__()
        logging.info("Loding data...")

        dataset = load_dataset(data_path,split="train[:]")
        dataset = dataset.rename_column("korean","sources")
        dataset = dataset.rename_column("english","targets")

        sources = dataset['sources']
        targets = dataset['targets']

        logging.info("Tokenzing inputs... This may take some time")
        data_dict = preprocess(sources,targets,tokenizer)

        self.input_ids= data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self,idx) -> Dict[str,torch.Tensor]:
        return dict(
                    input_ids=self.input_ids[idx],
                    labels=self.labels[idx]
                    )

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenzier: PreTrainedTokenizer

    def __call__(self,examples:Sequence[Dict]) -> Dict[str,torch.Tensor]:
        input_ids,labels = tuple([example[key] for example in examples] for key in ("input_ids","labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,batch_first=True,padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels,batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        )

def make_supervised_data_module(tokenizer: PreTrainedTokenizer,data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning"""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def train():
    parser = HfArgumentParser((ModelArgs,DataArguments,TrainingArguments))
    model_args, data_args,training_args = parser.parse_args_into_dataclasses()

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code = True,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenzier=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer,data_args=data_args)
    trainer = Trainer(model=model.cuda(), tokenizer=tokenizer,args=training_args,**data_module)
    train.train()
    trainer.save_state()
    trainer.save_model()