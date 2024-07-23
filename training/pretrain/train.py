"""
    Code based by https://github.com/xyjigsaw/LLM-Pretrain-SFT/blob/master/llm_pretrain/pretrain.py
"""
import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Dict,Optional, Sequence
import wandb

import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import Trainer,TrainerCallback
from datasets import load_dataset

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="google/gemma-2b"
        )

@dataclass
class DataArguments:
    data_path: str = field(default="euclaise/gsm8k_multiturn")
    

@dataclass
class TrainingArguments(transformers.TrainingArguments):
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

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
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

def _tokenize_fn(examples,tokenizer:transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a lost of strings"""
    tokenized_list = [
        tokenizer(
            example[0]['content'],
            return_tensors="pt",
            padding="longest",
            # max_length=tokenizer.model_max_length,
            truncation=False,
        ) for example in examples
    ]

    raw_input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    #split input_ids with model_max_length
    input_ids = []
    labels = []
    max_length = tokenizer.model_max_length
    for instance_ids in raw_input_ids:
        for i in range(0,len(instance_ids), max_length):
            input_ids.append(instance_ids[i : i + max_length])
            labels.append(instance_ids[i : i + max_length])
            if len(instance_ids[i : i + max_length]) < max_length:
                print("Warning: len(instance_ids[i : i + max_length]) < max_length")
                logging.warning(f"len(instance_ids[i : i + max_length]) < max_length: {len(instance_ids[i : i + max_length])} < {max_length}")
    return dict(
        input_ids=input_ids,
        labels=labels,
    )           

def preprocess(
    examples: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples_tokenized = _tokenize_fn(examples,tokenizer)
    input_ids = examples_tokenized["input_ids"]
    print('block num : ',len(input_ids))
    cnt = sum([input_id.shape[0] for input_id in input_ids])
    print('# of tokens', cnt)
    labels = copy.deepcopy(input_ids)
    return dict(
                input_ids=input_ids,
                labels=labels
                )

class PretrainDataset(Dataset):
    """Dataset for pretraining."""

    def __init__(self,data_path:str,tokenizer:transformers.PreTrainedTokenizer):
        super(PretrainDataset,self).__init__()
        logging.info("Loading data...")
        dataset = load_dataset(data_path,split="train[:]")['conversations']
        logging.info("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(dataset,tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self,idx) -> Dict[str,torch.Tensor]:
        return dict(input_ids=self.input_ids[idx],labels=self.labels[idx])

class DataCollatorForPretrainDataset(object):
    """Collate examples for pretraining."""

    def __init__(self,tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self,instances:Sequence[Dict]) -> Dict[str,torch.Tensor]:
        _input_ids, _labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        
        logging.info('length of input_ids[0] : ',len(_input_ids[0]))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            _input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            _labels,
            batch_first=True,
            padding_value=IGNORE_INDEX
        )
        logging.info('length of padded input_ids[0] : ',len(input_ids[0]) - len(_input_ids[0]))
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        )


def make_pretrain_data_module(tokenizer: transformers.PreTrainedTokenizer,data_args) -> Dict:
    """Make dataset and collator for pretraining."""
    train_dataset = PretrainDataset(tokenizer=tokenizer,data_path=data_args.data_path)
    data_collator = DataCollatorForPretrainDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset,eval_dataset=None,data_collator=data_collator)

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, num_steps=10):
        self.num_steps = num_steps
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.num_steps:
            return {"should_training_stop": True}
        else:
            return {}

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args,data_args, training_args = parser.parse_args_into_dataclasses()
    
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True, 
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
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
        tokenizer=tokenizer,
        model=model,
    
    )
    
    data_module = make_pretrain_data_module(tokenizer=tokenizer,data_args=data_args)
    
    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        **data_module,
        callback=[EarlyStoppingCallback()],
        )
    
    trainer.train(resume_from_checkpoint=True)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__=="__main__":
    main()