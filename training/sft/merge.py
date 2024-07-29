"""
작성중 Base Code만 남겼음 추후 수정

#TODO
1. 완전한 코드로 작성할 것
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig, PeftModelForCausalLM

# 모델과 토크나이저 로드
model_name = "gpt2"
lora_model_path = "path_to_lora_model"

# 원래 모델 로드
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# LoRA 모델 로드
peft_config = PeftConfig.from_pretrained(lora_model_path)
lora_model = PeftModel.from_pretrained(model, peft_config)

# LoRA 가중치를 병합
model = lora_model.merge_and_unload()

# 병합된 모델 저장
merged_model_path = "path_to_save_merged_model"
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

print(f"LoRA 가중치가 병합된 모델이 {merged_model_path}에 저장되었습니다.")