#!/bin/bash
# export PYTHONPATH=$PYTHONPATH:/mnt/nas-data-1/zhanglingjun.zlj1/carla/PythonAPI/carla
# export PYTHONPATH=$PYTHONPATH:/mnt/nas-data-1/zhanglingjun.zlj1/Bench2Drive
# export PYTHONPATH=$PYTHONPATH:/mnt/nas-data-1/zhanglingjun.zlj1/Bench2Drive/Bench2DriveZoo

BASE_PORT=30150
BASE_TM_PORT=50150
IS_BENCH2DRIVE=True
# BASE_ROUTES=leaderboard/data/routes_devtest
BASE_ROUTES=leaderboard/data/failed2_1_qw_tj
TEAM_AGENT=team_code/qwen_b2d_agent.py
# TEAM_CONFIG=/mnt/nas-data-1/wuchangjie.wcj/work/bev_ex3_v2_fulldata/checkpoint-19000   # for TCP and ADMLP
TEAM_CONFIG=/mnt/nas-data-1/zhanglingjun.zlj1/modelversion/v5_bev_target_fulldata_resume_1_epoch/checkpoint-20000   # for TCP and ADMLP
# TEAM_CONFIG=/mnt/nas-data-1/wuchangjie.wcj/work/bev_ex3/checkpoint-9000   # for TCP and ADMLP
# TEAM_CONFIG=your_team_agent_config.py+your_team_agent_ckpt.pth # for UniAD and VAD
BASE_CHECKPOINT_ENDPOINT=resume_0112_220_1
SAVE_PATH=resultfailed3/
PLANNER_TYPE=only_traj

GPU_RANK=6
PORT=$BASE_PORT
TM_PORT=$BASE_TM_PORT
ROUTES="${BASE_ROUTES}.xml"
CHECKPOINT_ENDPOINT="${BASE_CHECKPOINT_ENDPOINT}.json"
bash leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK
