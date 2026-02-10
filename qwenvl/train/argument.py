import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    model_base: str = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    tune_mm_audio: bool = field(default=False)
    tune_mm_qformer: bool = field(default=False)
    use_lora: bool = field(default=False)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_bias: str = field(default="none")
    lora_ckpt: str = field(default="No")

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=600)
    video_min_frames: Optional[int] = field(default=1)
    base_interval: float = field(default=0.5)
    max_pixels: int = field(default=176400)
    min_pixels: int = field(default=784)
    image_max_frame_pixels: int = field(default=2073600)
    image_min_frame_pixels: int = field(default=784)
    run_test: bool = field(default=False)
    do_sample: bool = field(default=False)
    num_sample: int = field(default=1)
    train_type: str = field(default="sft")
    feature_size: int = field(default=128)
    chunk_length: int = field(default=30)
    hop_length: int = field(default=160)
    sampling_rate: int = field(default=16000)
    run_demo: bool = field(default=False)
    force_audio: bool = field(default=False) 

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    pred_rank: int = field(default=0)
    temperature: float = field(default=0.01)
    train_ori_lora: bool = field(default=False)
    num_generations: int = field(default=1)
    num_iterations: int = field(default=1)
    beta: float = field(default=0.0)
    reward_func: List[str] = field(default_factory=lambda: ["repeat"])
    max_new_tokens: int = field(default=1024)
    reward_weights: List[float] = field(default_factory=lambda: [1.0])

    ip_list: List[str] = field(default_factory=lambda: ["None"])
    reward_enable_steps: List[int] = field(default_factory=list)
    reward_disable_steps: List[int] = field(default_factory=list)
    reward_stable_steps: List[int] = field(default_factory=list)