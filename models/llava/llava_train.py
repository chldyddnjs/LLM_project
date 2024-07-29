import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict,Optional,Sequence,List
import numpy as np

import torch
from torch.utils.data import Dataset,DataLoader

import datasets
from datasets import load_dataset

import transformers
from transformers import (
    Trainer,
    TrainerCallback,
    PreTrainedTokenizer,
    PreTrainedModel,
    HfArgumentParser,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    )
import tokenizers
from PIL import Image
from .conversation import *
from .constants import IGNORE_INDEX
from .utils import (
    tokenizer_image_token,
    print_nparams
    )

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    split: str = field(default="train[:10]")
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = './'
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)

def maybe_zero_3(param,ignore_status=False,name=None):
    """
    `deepspeed` 라이브러리를 사용하여 파라미터를 특정 조건에 따라 CPU로 복사하는 함수
    ZeRO (Zero Redundancy Optimizer)라는 기술을 통해 메모리 사용을 최적화
    
    `maybe_zero_3` 함수는 주어진 파라미터가 ZeRO 상태인지 확인하고, 필요한 경우 경고 메시지를 출력하며, 파라미터를 CPU로 복사
    ZeRO 상태인 파라미터는 `zero.GatheredParameters` 컨텍스트 매니저를 사용하여 CPU로 복사하고, 그렇지 않은 경우에는 단순히 `detach` 및 `clone` 메서드를 사용

    
    return: 복사된 파라미터를 반환
    """
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param,ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

#Borrowd from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params,bias):
    if bias == "none":
        to_return = {k: t for k,t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k:t for k,t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k,t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        
        for k,t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k:maybe_zero_3(v,ignore_status=True) for k,v in to_return.items()}
    return to_return

def get_peft_state_none_lora_matbe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k,t in named_params if "lora_" not in k }
    if require_grad_only:
        to_return = {k:t for k,t in to_return.items() if t.require_grad}
    to_return = {k: maybe_zero_3(v,ignore_status=True).cpu() for k,v in to_return.items()}
    return to_return

def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k,t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v,ignore_status=True).cpu() for k,v in to_return.items()}
    return to_return

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector','vision_tower','vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    
    if 'lm_head' in lora_module_names: #needed for 16bit
        lora_module_names.remove('lm_head')

    return list(lora_module_names)

def safe_save_model_for_hf_trainer(trainer:Trainer,
                                   output_dir:str):
    """Collects the state dict and dump to disk."""

    keys_to_match = ['mm_projector']
    if getattr(trainer.args,"tune_mm_mlp_adapter",False):
        #Only save Adapter
        keys_to_match.extend(['embed_tokens','embed_in'])

    weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(),keys_to_match)
    trainer.model.config.save_pretrained(output_dir)

    current_folder = output_dir.split('/')[-1]
    parent_folder = os.path.dirname(output_dir)
    if trainer.args.local_rank == 0 or trainer.args.local_rank==-1:
        if current_folder.startswith("checkpoint-"):
            mm_projector_folder = os.path.join(parent_folder,"mm_projector")
            os.makedirs(mm_projector_folder,exist_ok=True)
            torch.save(weight_to_save,os.path.join(mm_projector_folder, f'{current_folder}.bin'))
        else:
            torch.save(weight_to_save,os.path.join(output_dir,f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return
    
    state_dict = trainer.model.state_dict()

    if trainer.args.should_save:
        cpu_state_dict = {
            key:value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir,state_dict=cpu_state_dict)

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def preprocess(
        sources:Sequence[str],
        tokenizer: PreTrainedTokenizer,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    conv = default_conversation.copy()
    roles = {"human":conv.roles[0], "gpt":conv.roles[1]}
    
    #Apply pormpt templates
    conversations = []
    sentences = []

    for k,v in sources.items(): 
        sentences.append(v)
    
    conv.messages = []
    for f,v in zip(*sentences):
        role = roles[f]
        conv.append_message(role,v)
    conversations.append(conv.get_prompt())
    
    #Tokenize conversation

    input_ids = torch.stack([tokenizer_image_token(prompt,tokenizer) for prompt in conversations],dim=0)
    targets = input_ids.clone()
    # print("targets : ",targets)
    assert conv.sep_style == SeparatorStyle.TWO

    #Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX

        inst,ans = None,None
        for turn in turns:
            if turn == "": break

            try:
                inst,ans = turn.split(sep)
            except:
                ValueError("Instruction or Answer does not exist")

            if inst is not None and ans is not None:    
                inst += sep

            turn_len = len(tokenizer_image_token(turn,tokenizer))
            instruction_len = len(tokenizer_image_token(inst, tokenizer)) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += turn_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    return dict(
        input_ids=input_ids[0],
        labels=targets[0],
    )

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning"""

    def __init__(self, 
                 data_path:str,
                 tokenizer:PreTrainedTokenizer,
                 image_processor,
                 split:str,    
                 text_fields:list=["conversations","image"],             
                 ):
        super(LazySupervisedDataset,self).__init__()
        self.dataset = load_dataset(data_path,split=split)
        
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.text_fields = text_fields
        self.image_aspect_ratio = 'square'

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,idx):
        """
        TODO
        multi-model dataset for image-text pairs
        Note image type is <PIL Object>
        
        Return: {input_ids:"input_ids",labels:"labels"} 
        """
        
        src, img = self.dataset[idx]['conversations'], self.dataset[idx]['image'].convert('RGB')
        # src, img = [self.dataset[idx][text_field] for text_field in self.text_fields]
        
        if self.image_aspect_ratio == 'pad':
            img = self.expand2square(img, tuple(int(x*255) for x in self.image_processor.image_mean))
            img = self.image_processor.preprocess(np.array(img),return_tensors='pt')['pixel_values'][0]
        else:
            img = self.image_processor.preprocess(np.array(img),return_tensors='pt')['pixel_values'][0]            
        
        data_dict = preprocess(src,
                               self.tokenizer
                               )
        data_dict['image'] = img

        return data_dict
    
    def expand2square(pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self,tokenizer:PreTrainedTokenizer):
        self.tokeinzer = tokenizer

    def __call__(self, instances:Sequence[Dict]) -> Dict[str,torch.Tensor]:
        input_ids,labels = tuple([instance[key] for instance in instances] for key in ("input_ids","labels"))
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_id
            )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        input_ids = input_ids[:,:self.tokenizer.model_max_length]
        labels = labels[:,:self.tokenizer.model_max_length]
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenzier.pad_toekn_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                logging.info("Shape of image is not uniform")
                batch['images'] = torch.stack(images)
            else:
                logging.info("Shape of image is uniform")
                batch['images'] = images
    
        return batch

def supervised_data_module(tokenizer: PreTrainedTokenizer,data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                image_processor=data_args.image_processor,
                                split=data_args.split)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

if __name__=="__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    from transformers import CLIPImageProcessor
    data_args.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14") 
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    data_module = supervised_data_module(tokenizer,data_args)
    print(len(data_module['train_dataset']))