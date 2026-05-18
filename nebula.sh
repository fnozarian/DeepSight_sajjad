#!/bin/bash
# LLaMA-Factory 8卡分布式训练脚本

# method1
llamafactory-cli train --config ./configs/ad_bev_ex3_2.yaml

# method2
torchrun --nproc_per_node=8 src/train.py --config ./configs/ad_bev_ex3_2.yaml