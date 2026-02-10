# Copyright (2026) Tsinghua University, Tencent Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adopted from https://github.com/bytedance/video-SALMONN-2. The original license is located at 'third-party-license/video-salmonn-2.txt'.
# Adopted from https://github.com/QwenLM/Qwen2.5-VL. The original license is located at 'third-party-license/qwenvl25.txt'.
# Adopted from https://github.com/huggingface/transformers. The original license is located at 'third-party-license/transformers.txt'.

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path
import numpy as np
import torch
import random
import copy
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, WhisperFeatureExtractor
from transformers import AutoConfig

from tqdm import tqdm
import torch.distributed as dist
import yaml

from peft import LoraConfig, get_peft_model, PeftModel
from qwenvl.train.utils import KeywordsStoppingCriteria

from qwenvl.model.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from qwenvl.data.processing_qwen3_vl import Qwen3VLProcessor

local_rank = None

def collate_fn(batch):
    return batch[0]

def set_model(model_args, model):
    if model_args.tune_mm_llm:
        if model_args.use_lora:
            raise Exception("tune_mm_llm is not supported when use_lora is True")
        model.model.requires_grad_(True)
        model.lm_head.requires_grad_(True)
    else:
        model.model.requires_grad_(False)
        model.lm_head.requires_grad_(False)

    if model_args.tune_mm_vision:
        model.visual.requires_grad_(True)
    else:
        model.visual.requires_grad_(False)

    if model_args.tune_mm_mlp:
        model.visual.merger.requires_grad_(True)
    else:
        model.visual.merger.requires_grad_(False)

    if model_args.tune_mm_audio:
        model.audio.requires_grad_(True)
    else:
        model.audio.requires_grad_(False)

    if model_args.tune_mm_qformer:
        model.audio.qformer.requires_grad_(True)
        model.audio.q_tokens.requires_grad_(True)
        model.audio.audio_proj.requires_grad_(True)
    else:
        model.audio.qformer.requires_grad_(False)
        model.audio.q_tokens.requires_grad_(False)
        model.audio.audio_proj.requires_grad_(False)

def train(attn_implementation="flash_attention_2"):
    global local_rank

    seed = 2025
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args.temperature = training_args.temperature

    assert data_args.train_type in ["sft", "grpo"], f"train_type {data_args.train_type} is not supported"

    if data_args.run_demo:
        data_args.run_test = True
    
    training_args.train_type = data_args.train_type

    training_args.remove_unused_columns = False

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    processor = Qwen3VLProcessor.from_pretrained(model_args.model_base)
    data_args.image_processor = processor.image_processor
    data_args.video_processor = processor.video_processor
    data_args.audio_processor = WhisperFeatureExtractor(
        feature_size=data_args.feature_size, 
        sampling_rate=data_args.sampling_rate,
        hop_length=data_args.hop_length,
        chunk_length=data_args.chunk_length,
    )

    tokenizer = processor.tokenizer

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    if not data_args.run_test and not data_args.run_demo:
       raise Exception("Not Implemented.")
       
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            device_map="cpu",
        )
        if model_args.lora_ckpt != "No":
            model = PeftModel.from_pretrained(model, model_args.lora_ckpt)
            model = model.merge_and_unload()
            
        model.eval()
        model.cuda()

        if not data_args.run_demo:
            pred_rank = training_args.pred_rank

            os.makedirs(os.path.join(training_args.output_dir, training_args.run_name), exist_ok=True)

            result = []
            test_data = data_module["train_dataset"]
            loader = DataLoader(
                test_data,
                batch_size=1,
                shuffle=False,
                num_workers=training_args.dataloader_num_workers,
                collate_fn=collate_fn,
                in_order=False,
            )
            for inputs in tqdm(loader, desc=f"RANK {pred_rank}"):
                if inputs:
                    res_i = {
                        "video": inputs.pop("video", None),
                        "prompt": inputs.pop("prompt", None),
                        "ref": inputs.pop("ref", None),
                        "audio": inputs.pop("audio", None),
                    }

                    new_inputs = {}
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            new_inputs[k] = v.to(model.device)
                        elif (k == 'video_second_per_grid' or k == 'pos_video_second_per_grid') and v is not None:
                            new_inputs[k] = torch.tensor([v], device=model.device)
                        elif v is None:
                            continue
                        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                            new_inputs[k] = [it.to(model.device) for it in v]
                        else:
                            new_inputs[k] = v

                    inputs = new_inputs

                    keywords = ["<|im_end|>"]
                    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, inputs["input_ids"])
                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=training_args.max_new_tokens, do_sample=data_args.do_sample, top_p=0.9, stopping_criteria=[stopping_criteria],)
                    output_trimmed = outputs[0, len(inputs["input_ids"][0]):]
                    output_text = tokenizer.decode(output_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    res_i["pred"] = output_text
                    result.append(res_i)

            with open(os.path.join(training_args.output_dir, training_args.run_name, f"test_results_rank{pred_rank}.json"), "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(os.path.join(training_args.output_dir, training_args.run_name, f"test_results_rank{pred_rank}.json"))

            return

        else:
            test_dataset = data_module["train_dataset"]
            data_collator = data_module["data_collator"]
            while True:
                try:
                    yaml_file = input("yaml file: ")
                    with open(yaml_file, 'r') as file:
                        yaml_data = yaml.safe_load(file)

                    video_path = yaml_data.get('video_path', None)
                    audio_path = yaml_data.get('audio_path', None)
                    do_sample = yaml_data.get("do_sample", False)
                    top_p = yaml_data.get("top_p", 0.9)
                    max_new_tokens = yaml_data.get("max_new_tokens", 1024)
                    seed = yaml_data.get("seed", 2025)

                    from transformers import set_seed
                    if seed > 0:
                        set_seed(seed)

                    if video_path is not None:
                        assert os.path.exists(video_path)
                    if audio_path is not None:
                        assert os.path.exists(audio_path)
                    assert video_path is not None or audio_path is not None

                    interval = yaml_data.get("interval", 0.5)
                    max_frames = yaml_data.get("max_frames", 256)

                    test_dataset.list_data_dict = [{}]

                    qs = yaml_data['question']
                    if video_path is not None:
                        qs = "<video>\n" + qs
                        test_dataset.list_data_dict[0]["video"] = video_path
                        if audio_path is not None:
                            test_dataset.list_data_dict[0]["audio"] = audio_path
                    elif audio_path is not None:
                        qs = "<audio>\n" + qs
                        test_dataset.list_data_dict[0]["audio"] = audio_path

                    test_dataset.list_data_dict[0]["conversations"] = []
                    test_dataset.list_data_dict[0]["conversations"] += [
                        {
                            "from": "human",
                            "value": qs.strip(),
                            # "prefix": prefix,
                        },
                        {
                            "from": "gpt",
                            "value": ""
                        }
                    ]

                    inputs = test_dataset._get_item(0)

                    res_i = {
                        "video": inputs.pop("video", None),
                        "prompt": inputs.pop("prompt", None),
                        "ref": inputs.pop("ref", None),
                        "audio": inputs.pop("audio", None),
                    }

                    new_inputs = {}
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            new_inputs[k] = v.to(model.device)
                        elif (k == 'video_second_per_grid' or k == 'pos_video_second_per_grid') and v is not None:
                            new_inputs[k] = torch.tensor([v], device=model.device)
                        elif v is None:
                            continue
                        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                            new_inputs[k] = [it.to(model.device) for it in v]
                        else:
                            new_inputs[k] = v
                    inputs = new_inputs

                    keywords = ["<|im_end|>"]
                    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, inputs["input_ids"])
                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, top_p=top_p, stopping_criteria=[stopping_criteria])

                    output_trimmed = outputs[0, len(inputs["input_ids"][0]):]
                    output_text = tokenizer.decode(output_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    print(output_text)
                    print("=" * 50)
                
                except Exception as e:
                    print(f"Error {e}, line: {e.__traceback__.tb_lineno}")
                    breakpoint()

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
