#!/bin/bash

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


cd "$(cd $(dirname $0); pwd)"

bash test.sh \
    --interval 0.5 \
    --run_name debug2 \
    --dataset scripts/example_data.json \
    --max_frames 128 \
    --model /mnt/sh/mmvision/home/changlitang/models/D-ORCA-8B-0210 \
    --model_base /mnt/sh/mmvision/home/changlitang/models/D-ORCA-8B-0210 \
    --gpu_num 1 --max_new_tokens 3072
