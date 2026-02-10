import os
import copy
import json
import random
import logging
import re
import time
import math
import itertools
import ast
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple, Union
from io import BytesIO
import base64
from collections.abc import Sequence
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from torchcodec.decoders import VideoDecoder, AudioDecoder
from joblib import Parallel, delayed, cpu_count

from decord import VideoReader, cpu
import soundfile as sf
import transformers

import sys
from pathlib import Path
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))

import zipfile
from qwenvl.train.utils import IGNORE_INDEX, PAD_TOKEN_ID

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def decode_sequentially(indices: List[int], video_path):
    # Decode frames sequentially using a single decoder instance
    decoder = VideoDecoder(video_path)
    return decoder.get_frames_at(indices)

def decode_with_multithreading(
    indices: List[int],
    num_threads: int,
    video_path
):
    # Decode frames using multiple threads with joblib.
    chunks = split_indices(indices, num_chunks=num_threads)

    results = Parallel(n_jobs=num_threads, prefer="threads", verbose=0)(
        delayed(decode_sequentially)(chunk, video_path) for chunk in chunks
    )

    # Concatenate results from all threads
    return torch.cat([frame_batch.data for frame_batch in results], dim=0)

def split_indices(indices: List[int], num_chunks: int) -> List[List[int]]:
    """Split a list of indices into approximately equal chunks."""
    chunk_size = len(indices) // num_chunks
    chunks = []

    for i in range(num_chunks - 1):
        chunks.append(indices[i * chunk_size:(i + 1) * chunk_size])

    # Last chunk may be slightly larger
    chunks.append(indices[(num_chunks - 1) * chunk_size:])
    return chunks

def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def split_into_groups_3vl(counts, groups, second_per_grid_ts=None):
    result = []
    if second_per_grid_ts is None:
        for count, g in zip(counts, groups):
            g = g.item()
            base = count // g
            remainder = count % g
            if remainder == 0:
                group_list = [base] * g
            else:
                group_list = [base] * g
                step = g / remainder
                for i in range(1, remainder + 1):
                    position = i * step
                    index = math.floor(position) - 1
                    if index >= g:
                        index = g - 1
                    group_list[index] += 1
            result.append(group_list)
    else:
        for count, g, second in zip(counts, groups, second_per_grid_ts):
            g = g.item()
            frame_idx = (torch.arange(g) * second * 2).long()
            per_grid_t = torch.diff(frame_idx)
            group_list = per_grid_t.tolist()
            group_list.append(count - sum(group_list))
            assert sum(group_list) == count, f"Strage count under count={count}, g={g}, second={second}"
            result.append(group_list)
    return result


def _calculate_timestamps(indices: Union[list[int], np.ndarray], video_fps: float, merge_size: int = 2):
    if not isinstance(indices, list):
        indices = indices.tolist()
    if len(indices) % merge_size != 0:
        indices.extend(indices[-1] for _ in range(merge_size - len(indices) % merge_size))
    timestamps = [idx / video_fps for idx in indices]
    # @JJJYmmm frames are merged by self.merge_size, \
    # so we need to average the timestamps between the first/last frame within the temporal patch
    timestamps = [
        (timestamps[i] + timestamps[i + merge_size - 1]) / 2 for i in range(0, len(timestamps), merge_size)
    ]
    return timestamps

def generate_id_target_3vl(
    source,
    grid_thw_image, 
    grid_thw_video, 
    audio_lengths, 
    tokenizer, 
    target_role,
    merge_size: int = 2,
    second_per_grid_ts: List = [],
    fps: List = [],
    frame_idx: List = []
):
    visual_replicate_index_image = 0
    visual_replicate_index_video = 0
    roles = {"human": "user", "gpt": "assistant", "chosen": "assistant", "reject": "assistant"}
    system_message = "You are a helpful assistant."
    input_id, target = [], []

    input_id += tokenizer.apply_chat_template(
        [{"role": "system", "content": system_message}]
    )
    target += [IGNORE_INDEX] * len(input_id)
    for conv in source:
        try:
            role = conv["role"]
            content = conv["content"]
        except:
            role = conv["from"]
            content = conv["value"]
        if role not in ["human", target_role]:
            continue

        role = roles.get(role, role)
        if role == "user":
            if "<image>" in content:
                parts = content.split("<image>")
                new_parts = []
                for i in range(len(parts) - 1):
                    new_parts.append(parts[i])
                    replacement = (
                        "<|vision_start|>"
                        + f"<|image_pad|>"
                        * grid_thw_image[i]
                        + "<|vision_end|>"
                    )
                    new_parts.append(replacement)
                new_parts.append(parts[-1])
                content = "".join(new_parts)

            if "<video>" in content:
                parts = content.split("<video>")
                new_parts = []
                frame_seq_length = [
                    merged_thw[1:].prod() // merge_size**2
                    for merged_thw in grid_thw_video
                ]
                if audio_lengths is None:
                    for i in range(len(parts) - 1):
                        curr_timestamp = _calculate_timestamps(
                            frame_idx[i],
                            fps[i],
                            merge_size,
                        )
                        new_parts.append(parts[i])
                        # replacement = (
                        #     "<|vision_start|>"
                        #     + f"<|video_pad|>"
                        #     * grid_thw_video[i]
                        #     + "<|vision_end|>"
                        # )
                        replacement = ""
                        for idx in range(grid_thw_video[i][0]):
                            curr_time = curr_timestamp[idx]
                            replacement += f"<{curr_time:.1f} seconds>"
                            replacement += "<|vision_start|>"
                            replacement += "<|video_pad|>" * frame_seq_length[i]
                            replacement += "<|vision_end|>"
                        new_parts.append(replacement)
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)
                else:
                    for i in range(len(parts) - 1):
                        curr_timestamp = _calculate_timestamps(
                            frame_idx[i],
                            fps[i],
                            merge_size,
                        )
                        new_parts.append(parts[i])
                        if second_per_grid_ts is None:
                            per_timestep_audio_len = split_into_groups_3vl(audio_lengths, [grid_thw_video[i][0] for i in range(len(grid_thw_video))])
                        else:
                            per_timestep_audio_len = split_into_groups_3vl(audio_lengths, [grid_thw_video[i][0] for i in range(len(grid_thw_video))], [ts[0] for ts in second_per_grid_ts])
                        replacement = ""
                        for idx in range(grid_thw_video[i][0]):
                            curr_time = curr_timestamp[idx]
                            replacement += f"<{curr_time:.1f} seconds>"
                            replacement += "<|vision_start|>"
                            replacement += "<|video_pad|>" * frame_seq_length[i]
                            replacement += f"<|audio_pad|>" * per_timestep_audio_len[i][idx]
                            replacement += "<|vision_end|>"
                        new_parts.append(replacement)
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)
                            

            if "<audio>" in content:
                parts = content.split("<audio>")
                new_parts = []
                for i in range(len(parts) - 1):
                    new_parts.append(parts[i])
                    replacement = f"<|audio_pad|>" * audio_lengths[i] # remove vision_start for minimum change on rope index
                    new_parts.append(replacement)
                new_parts.append(parts[-1])
                content = "".join(new_parts)
        conv = [{"role": role, "content": content}]
        encode_id = tokenizer.apply_chat_template(conv)
        input_id += encode_id
        if role in ["user", "system"]:
            target += [IGNORE_INDEX] * len(encode_id)
        else:
            target_mask = encode_id.copy()
            target_mask[:3] = [IGNORE_INDEX] * 3
            target += target_mask
    return input_id, target


def preprocess_qwen_3_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image: List = [],
    grid_thw_video: List = [],
    audio_lengths = None,
    merge_size=2,
    second_per_grid_ts: List = [],
    fps: List = [],
    frame_idx: List = [],
) -> Dict:
    if second_per_grid_ts is not None and isinstance(second_per_grid_ts, list) and not isinstance(second_per_grid_ts[0], list):
        second_per_grid_ts = [second_per_grid_ts]
    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    input_ids, targets = [], []

    is_dpo_data = False
    for i, source in enumerate(sources):
        try:
            if source[0]["from"] != "human":
                source = source[1:]
        except:
            print(sources)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]
        
        input_id, target = generate_id_target_3vl(source, grid_thw_image, grid_thw_video, audio_lengths, tokenizer, "gpt", merge_size, second_per_grid_ts, fps, frame_idx)
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = dataset
        rank0_print(f"Loading datasets: {dataset_list}")
        self.data_args = data_args

        list_data_dict = []

        for data in dataset_list:
            file_format = data.split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data)
            else:
                annotations = json.load(open(data, "r"))
            list_data_dict += annotations

        for d in list_data_dict:
            if d["conversations"][0]["from"] == "system":
                idx = 1
            else:
                idx = 0
            if "<image>" in d["conversations"][idx]["value"] and not "image" in d and ("video" in d or "cos_key" in d or "frame_dir" in d or "zip_file" in d):
                d["conversations"][idx]["value"] = d["conversations"][idx]["value"].replace(
                    "<image>", "<video>"
                )
            if "<image>" in d["conversations"][idx]["value"] and not "image" in d and not ("video" in d or "frame_dir" in d) and ("audio" in d or "cos_audio" in d):
                d["conversations"][idx]["value"] = d["conversations"][idx]["value"].replace(
                    "<image>", "<audio>"
                )

        rank0_print(f"Total training samples: {len(list_data_dict)}")

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        # self.data_args = data_args
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

        self.data_args.video_processor.do_sample_frames = False


    def __len__(self):
        return len(self.list_data_dict)

    def process_audio(self, audio_file=None, audio_wav=None, timestamps=None):
        try:
            audio_kwargs = {"sampling_rate": 16000, "padding": "max_length", "return_attention_mask": False}
            processor = self.data_args.audio_processor

            if audio_wav is None:
                if isinstance(audio_file, list):
                    audio_data = []
                    for file in audio_file:
                        if audio_file.endswith(".mp4"):
                            decoder = AudioDecoder(
                                audio_file,
                                sample_rate=audio_kwargs["sampling_rate"],
                                num_channels=1,
                            )
                            audio = decoder.get_all_samples()
                            audio_data.append(audio.data.numpy().squeeze(0))
                            sr = audio_kwargs["sampling_rate"]
                        else:
                            audio, sr = sf.read(audio_file)
                            if len(audio.shape) == 2:
                                audio = audio[:, 0]
                            assert sr == 16000
                            audio_data.append(audio)
                else:
                    if audio_file.endswith(".mp4"):
                        decoder = AudioDecoder(
                            audio_file,
                            sample_rate=audio_kwargs["sampling_rate"],
                            num_channels=1,
                        )
                        audio = decoder.get_all_samples()
                        audio = audio.data.numpy().squeeze(0)
                        sr = audio_kwargs["sampling_rate"]
                        if timestamps is not None:
                            audio = audio[timestamps[0] * sr: timestamps[1] * sr]
                        audio_data = [audio]
                    else:
                        audio, sr = sf.read(audio_file)
                        if len(audio.shape) == 2:
                            audio = audio[:, 0]
                        assert sr == 16000
                        if timestamps is not None:
                            audio = audio[timestamps[0] * sr: timestamps[1] * sr]
                        if len(audio) == 0:
                            return None, None, None, None
                        audio_data = [audio]
            else:
                sr = 16000
                audio_data = [audio_wav]

            audio_inputs = []
            audio_lengths = []
            for idx in range(len(audio_data)):
                if audio_data[idx].shape[0] < audio_kwargs["sampling_rate"]:
                    padding = audio_kwargs["sampling_rate"] - audio_data[idx].shape[0]
                    audio_data[idx] = np.pad(audio_data[idx], (0, padding), mode="constant", constant_values=0)
                audio_lst = [audio_data[idx][k: k + 30 * audio_kwargs["sampling_rate"]] for k in range(0, len(audio_data[idx]), 30 * audio_kwargs["sampling_rate"])]
                spectrogram_lst = [processor(a, sampling_rate=audio_kwargs["sampling_rate"], return_tensors="pt")["input_features"].squeeze() for a in audio_lst]
                audio_inputs.append(torch.stack(spectrogram_lst, dim=0))
                audio_lengths.append(math.ceil(len(audio_data[idx]) / (30 * audio_kwargs["sampling_rate"])) * 30 * 2)
            
            
            if not self.data_args.run_test:
                assert audio_lengths[0] > 0
            return audio_inputs, audio_lengths, audio_data
        
        except Exception as e:
            print(f"Process Audio Error: {e},  file: {audio_file}, line: {e.__traceback__.tb_lineno}")
            raise e
            
    def process_video(self, video_file, timestamps=None, max_frame_num=-1):
        torchcodec_video = self.real_torchcodec(video_file, timestamps=timestamps, max_frame_num=max_frame_num)
        return torchcodec_video

    def real_torchcodec(self, video_file, timestamps=None, max_frame_num=-1):
        device = "cpu"  # or e.g. "cuda"
        decoder = VideoDecoder(video_file, device=device)
        total_frame_num = decoder.metadata.num_frames
        avg_fps = decoder.metadata.average_fps
        video_length = total_frame_num / avg_fps
        interval = getattr(self.data_args, "base_interval", 0.5)
        start_idx = 0
        end_idx = total_frame_num - 1
        if timestamps is not None:
            timestamps[0] = min(max(timestamps[0], 0), video_length)
            timestamps[1] = min(max(timestamps[1], 0), video_length)
            start_idx = round(timestamps[0] * avg_fps)
            end_idx = round(timestamps[1] * avg_fps)
            start_idx = min(max(start_idx, 0), total_frame_num - 1)
            end_idx = min(max(end_idx, 0), total_frame_num - 1)
            video_length = timestamps[1] - timestamps[0]

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 1)
        video_max_frames = getattr(self.data_args, "video_max_frames", 600) if max_frame_num <= 0 else max_frame_num

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(start_idx, end_idx, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        # frame_batch = decoder.get_frames_at(indices=frame_idx.tolist())
        try:
            frame_batch = decode_with_multithreading(indices=frame_idx.tolist(), num_threads=8, video_path=video_file)
        except:
            frame_idx[-1] = frame_idx[-1] - 1
            frame_batch = decode_with_multithreading(indices=frame_idx.tolist(), num_threads=8, video_path=video_file) 
        video = frame_batch.data.cpu().numpy()
        frame_idx = frame_idx - frame_idx[0]
        return *self.process_video_frames_3vl(video, frame_idx, video_length), avg_fps, frame_idx

    def process_video_frames_3vl(self, video, frame_idx, video_length):
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.data_args.video_processor)
        max_pixels = getattr(self.data_args, "video_max_frames", 128) * getattr(self.data_args, "max_pixels", 176400)
        processor.max_pixels = max_pixels
        processor.min_pixels = getattr(self.data_args, "min_pixels", 784)
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels
        video_processed = processor(videos=video)
        video_tensor = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"][0]
        second_per_grid_ts = [
            self.data_args.image_processor.temporal_patch_size / fps
        ] * len(grid_thw)
        return video_tensor, grid_thw, second_per_grid_ts

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sample = self._get_item(i)
        return sample 

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        try:
            sources = self.list_data_dict[i]
            if isinstance(i, int):
                sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
            
            image = None
            grid_thw_merged = None
            video_grid_thw_merged = None
            grid_thw = None
            video = None
            video_grid_thw = None
            second_per_grid_ts = None
            audio = None
            audio_lengths = None
            raw_wav = None
            fps = None
            frame_idx = None

            if "video" in sources[0]:
                video_file = sources[0]["video"]
                timestamps = sources[0].get("timestamps", None)
                video, video_grid_thw, second_per_grid_ts, fps, frame_idx = self.process_video(video_file, timestamps=timestamps)
                video = [video]
                fps = [fps]
                frame_idx = [frame_idx]

                video_grid_thw_merged = copy.deepcopy(video_grid_thw)
                if not isinstance(video_grid_thw, Sequence):
                    video_grid_thw_merged = [video_grid_thw_merged]
                    video_grid_thw = [video_grid_thw]

            if "audio" in sources[0]:
                audio_file = sources[0]["audio"]
                timestamps = sources[0].get("timestamps", None)
                audio, audio_lengths, raw_wav = self.process_audio(audio_file, timestamps=timestamps)
            
            if "audio" in sources[0] and self.data_args.force_audio:
                assert audio is not None and audio[0]["input_features"] is not None and audio[0]["feature_attention_mask"] is not None and audio_lengths is not None
            
            if raw_wav is not None and len(raw_wav[0]) < 16000: # pad audio to at least 1s
                sil = np.zeros(16000 - len(raw_wav[0]), dtype=float)
                raw_wav[0] = np.concatenate((raw_wav[0], sil), axis=0)

            chat_sources = copy.deepcopy([e["conversations"] for e in sources])

            if True:
                data_dict = preprocess_qwen_3_visual(
                    chat_sources,
                    self.tokenizer,
                    grid_thw_image=grid_thw_merged if grid_thw_merged else None,
                    grid_thw_video=video_grid_thw_merged if video_grid_thw_merged else None,
                    audio_lengths=audio_lengths if audio_lengths else None,
                    merge_size=self.data_args.image_processor.merge_size,
                    second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
                    fps=fps if fps else None,
                    frame_idx=frame_idx if frame_idx else None,
                )

                debug_input_ids = data_dict["input_ids"]
                audio_token_number = (debug_input_ids == 151669).sum()
                assert audio_token_number % 60 == 0, f"Weird data: {sources}"
                position_ids = None
                if "video" not in sources[0] and "audio" not in sources[0]:
                    grid_thw_merged = None
                    sources = copy.deepcopy([e["conversations"] for e in sources])
                    data_dict = preprocess_qwen_3_visual(
                        sources, self.tokenizer, None, None
                    )
                    position_ids = (
                        torch.arange(0, data_dict["input_ids"].size(1))
                        .view(1, -1)
                        .unsqueeze(0)
                        .expand(3, -1, -1)
                    )


            data_dict["position_ids"] = position_ids
            data_dict["attention_mask"] = [data_dict["input_ids"][0].size(0)]

            if "video" in self.list_data_dict[i]:
                data_dict["pixel_values_videos"] = torch.cat(video, dim=0)
                data_dict["video_grid_thw"] = torch.cat(
                    [thw.unsqueeze(0) for thw in video_grid_thw], dim=0
                )
            if audio is not None:
                audio = torch.cat(audio, dim=0)
            data_dict["audio_feature"] = audio
            data_dict["audio_lengths"] = audio_lengths
            data_dict["train_type"] = self.data_args.train_type

            if self.data_args.run_test or self.data_args.train_type == "grpo":
                if self.data_args.run_test:
                    labels = data_dict.pop("labels", None)
                    len_input = sum(labels[0] == IGNORE_INDEX)
                    data_dict["input_ids"] = data_dict["input_ids"][:, :len_input]
                    if data_dict["position_ids"] is not None:
                        data_dict["position_ids"] = data_dict["position_ids"][:, :, :len_input]
                    data_dict["attention_mask"] = torch.ones_like(data_dict["input_ids"])
                else:
                    labels = data_dict["labels"]
                    len_input = sum(labels[0] == IGNORE_INDEX)
                    data_dict["ori_ids"] = copy.deepcopy(data_dict["input_ids"])
                    data_dict["ori_attention_mask"] = torch.ones_like(data_dict["ori_ids"])
                    data_dict["input_ids"] = data_dict["input_ids"][:, :len_input]
                    if data_dict["position_ids"] is not None:
                        data_dict["position_ids"] = data_dict["position_ids"][:, :, :len_input]
                    data_dict["attention_mask"] = torch.ones_like(data_dict["input_ids"])

                if "video" in sources[0]:
                    data_dict["video"] = sources[0]["video"]
                else:
                    data_dict["video"] = None

                if "audio" in sources[0]:
                    data_dict["audio"] = sources[0]["audio"]
                else:
                    data_dict["audio"] = None

                data_dict["prompt"] = sources[0]["conversations"][0]
                data_dict["ref"] = sources[0]["ref"] if "ref" in sources[0] else sources[0]["conversations"][1]["value"]

            return data_dict
        except Exception as e:
            print(f"Error: {e}, line: {e.__traceback__.tb_lineno}")
            # raise e
            if self.data_args.run_test:
                print(f"Error loading {sources[0]}")
                return None
            else:
                randidx = random.choice(range(len(self.list_data_dict))) # self.type_dict[sources[0].get("type", "retrieval")])
                return self.__getitem__(randidx)


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def process_ids(self, input_ids, labels, position_ids):
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        if position_ids is not None and all(id is not None for id in position_ids):
            position_ids = pad_and_cat(position_ids)
            position_ids = position_ids[:, : self.tokenizer.model_max_length]
        else:
            position_ids = None

        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        return input_ids, labels, position_ids, attention_mask

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids, labels, position_ids, attention_mask = self.process_ids(
            input_ids, labels, position_ids
        )

        train_type = [instance["train_type"] for instance in instances][0]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            position_ids=position_ids,
            attention_mask=attention_mask,
            train_type=train_type,
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        audios = list(
            instance["audio_feature"]
            for instance in instances
            if instance["audio_feature"] is not None
        )

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        if len(audios)!= 0:
            concat_audios = torch.cat([audio for audio in audios], dim=0)
            audio_lengths = [
                instance["audio_lengths"]
                for instance in instances
                if "audio_lengths" in instance
            ]
            audio_lengths = [l for length in audio_lengths for l in length]
        else:
            concat_audios = None
            audio_lengths = None

        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["audio_feature"] = concat_audios
        batch["audio_lengths"] = audio_lengths
        return batch

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    if data_args.train_type == "grpo":
        data_collator = lambda x: x[0]
    else:
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )

if __name__ == "__main__":
    from qwenvl.train.argument import DataArguments
    from transformers import AutoTokenizer, WhisperFeatureExtractor
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    
    random.seed(2025)
    data_args = DataArguments()
    data_args.dataset_use = "/mnt/sh/mmvision/home/changlitang/preprocess_dataset/grpo_selfYtb_words_train_v2_100words_reanntTime.json"
    data_args.video_max_frames = 128
    data_args.run_test = True # False # 

    processor = Qwen3VLProcessor.from_pretrained("/mnt/sh/mmvision/home/changlitang/models/Qwen3-VL-8B-Instruct-Audio")
    data_args.image_processor = processor.image_processor
    data_args.video_processor = processor.video_processor
    data_args.audio_processor = WhisperFeatureExtractor(
        feature_size=data_args.feature_size, 
        sampling_rate=data_args.sampling_rate,
        hop_length=data_args.hop_length,
        chunk_length=data_args.chunk_length,
    )

    tokenizer = processor.tokenizer
    dataset = LazySupervisedDataset(tokenizer, data_args=data_args)
    # from tqdm import tqdm
    # for i in tqdm(range(len(dataset))):
    #     item0 = dataset._get_item(i)
    # item0 = dataset._get_item(42041+19180+84723+7481)
    item0 = dataset._get_item(0)
    # item1 = dataset._get_item(1)

    # collate_fn = DataCollatorForOmniDataset()
    collate_fn = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    breakpoint()
    batch = collate_fn([item0])
    breakpoint()