import numpy as np
import os, random
import torch
from typing import Optional, Dict, Sequence

from dataclasses import dataclass, field
from datasets import load_dataset

import transformers
# from transformers import HfArgumentParser
# from transformers import TextDataset
# from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextDataset,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    logging,
)

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from peft import LoraConfig, PeftModel

from accelerate import Accelerator
import mlflow


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)

@dataclass
class DataArguments:
    train_data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    valid_data_path: str = field(default=None, metadata={"help": "Path to the training data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruct'])):
        text = f"<s># 問題: \n{example['instruct'][i]}\n\n# 入力文: {example['input'][i]}\n\n# 回答: \n{example['output'][i]}</s>"
        output_texts.append(text)
    return output_texts

def train():
    mlflow.autolog()
    
    # other args
    max_seq_length = 2048
    packing = False
    
    # 学習準備
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 量子化の有効化
        bnb_4bit_quant_type="nf4",  # 量子化種別 (fp4 or nf4)
        bnb_4bit_compute_dtype=torch.float16,  # 量子化のdtype (float16 or bfloat16)
        bnb_4bit_use_double_quant=False,  # 二重量子化の有効化
    )
    
    device_index = Accelerator().process_index
    device_map = {"": device_index}
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        quantization_config=bnb_config,
        device_map=device_map
        #device_map="auto"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        use_fast=False,
        trust_remote_code=True,
        
    )
    
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"
    
    # CSVを読み込み
    dataset = load_dataset('csv',data_files={'train': data_args.train_data_path,
                                             'valid': data_args.valid_data_path}
    )
    
    response_template = "\n# 回答: \n"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    
    peft_config = LoraConfig(
        r=8,  # LoRAアテンションの次元
        lora_alpha=16,  # LoRAスケーリングのAlphaパラメータ
        lora_dropout=0.05,  # LoRA レイヤーのドロップアウト確率
        bias="none",  # LoRAのバイアス種別 ("none","all", "lora_only")
        task_type="CAUSAL_LM",  # タスク種別
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
    )

    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        peft_config=peft_config,
        args=training_args,
        packing=packing,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )
    
    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    seed = 42
    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    train()


