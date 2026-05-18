# DeepSight

DeepSight 是一个基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 的自动驾驶感知与推理框架，在 LLaMA-Factory 的基础上针对自动驾驶场景进行了数据处理、模型训练与推理评估的全流程定制。

## 快速开始

### 环境配置

```bash
# 克隆项目
git clone https://github.com/hotdogcheesewhite/DeepSight.git
cd DeepSight

# 创建虚拟环境
conda create -n deepsight python=3.10 -y && conda activate deepsight

# 安装 PyTorch（推荐 2.6.0 或其他兼容版本）
# torch == 2.6.0

# 安装依赖
pip install -r requirements.txt
# pip install -e .
```

## 项目概述

本项目在 LLaMA-Factory 基础之上扩展了以下功能：

- **BEV（鸟瞰图）数据处理与可视化流水线**
- **VLM（视觉语言模型）训练数据构建**
- **DINOv3 特征提取与 BEV Query 监督**
- **基于 Bench2Drive 的开环与闭环评估**

## 目录结构

```
deepsight/
├── configs/                                  # 训练配置文件（YAML）
├── data/                                     # 数据集处理工具
├── src/                                      # 核心源代码
│   ├── tools/                                # 数据与评估工具脚本
│   │   ├── crop_bev_for_bench2drive.py       # BEV 图裁剪
│   │   ├── visual_for_bev.py                 # BEV 可视化
│   │   ├── create_date_set.py                # VLM 训练数据构建
│   │   ├── eval_and_visual.py                # 推理可视化与开环评测
│   │   └── merge_model_weight.py             # 模型权重合并（vLLM 用）
│   ├── transformers/src/transformers/        # 修改后的 transformers
│   │   └── models/qwen2_5_vl/modeling_*.py    # Qwen2.5-VL 模型（含 DINOv3）
│   ├── llamafactory/data/ad_collator.py      # 数据整理器（取消 token CE 损失）
│   ├── infer_for_debug.py                    # 原始 transformers 推理
│   └── infer_with_vllm.py                    # vLLM 推理
├── bench2drive/                              # Bench2Drive 评估框架
│   └── leaderboard/scripts/
│       └── run_evaluation_qwen.sh            # 闭环评测脚本
├── nebula.sh                                 # Nebula 集群训练脚本
└── requirements.txt                          # 训练环境依赖
```

---

## 一、数据准备

### 1. 创建 BEV 图

**脚本：** [src/tools/crop_bev_for_bench2drive.py](src/tools/crop_bev_for_bench2drive.py)

从 Bench2Drive 数据中裁剪生成 BEV（鸟瞰图），每张 BEV 包含 **5 张固定分辨率的未来运动图像**，用于预测车辆轨迹。

**注意事项：**
- 受天气影响较大 —— 可通过调低 BEV 高度来缓解
- 周围高楼存在视角变换问题

**可视化检查：** [src/tools/visual_for_bev.py](src/tools/visual_for_bev.py)

重点检查**拐弯场景**的数据质量。

### 2. 创建 VLM 训练数据

**脚本：** [bench2drive/dataprocess/targetpointgen.py](bench2drive/dataprocess/targetpointgen.py)

将原始数据转换为对话格式的训练数据，输入需要标注文件。

### 3. （可选）自主构造 CoT 标注内容

**脚本：** [src/tools/create_date_set_target_need_to_cot.py](src/tools/create_date_set_target_need_to_cot.py)

将 textprompt 替换为希望的 prompt，生成需要 Qwen-3VL 标注的数据。

### 4. （可选）调用 API 生成标注数据

**脚本：** [bench2drive/dataprocess/jsonopenai.py](bench2drive/dataprocess/jsonopenai.py)

更新 OpenAI 的 API Key，调用 Qwen3VL 模型实现标注。

---

## 二、模型训练

### 训练入口

1. 在 `deepsight/data/dataset_info.json` 中加入对应的数据集信息，参考 [LLaMA-Factory 官网文档](https://llamafactory.readthedocs.io/)，组织数据集，修改路径为之前生成的 JSONL 文件。

2. 执行以下命令：

```bash
bash nebula.sh
```

训练入口为 `src/train.py`，超参数定义在 `configs/` 目录下的 YAML 配置文件中。

### 损失设计

[src/llamafactory/data/ad_collator.py](src/llamafactory/data/ad_collator.py)

设计了训练方式。

---

## 三、模型推理

### 1. 使用原始 Transformers 推理

**脚本：** [src/infer_for_debug.py](src/infer_for_debug.py)

使用修改后的原始 transformers 进行推理（包含 DINOv3 等模块）。

- **可视化：** [src/tools/eval_and_visual.py](src/tools/eval_and_visual.py)
- **开环评测：** [src/tools/eval_and_visual.py](src/tools/eval_and_visual.py)

### 2. 使用 vLLM 推理

**脚本：** [src/infer_with_vllm.py](src/infer_with_vllm.py)

vLLM 使用内部实现的 transformers，**不包含 DINOv3 等模块**。推理前需先合并模型权重：

- **合并脚本：** [src/tools/merge_model_weight.py](src/tools/merge_model_weight.py)
- 固定 target 路径：`/mnt/nas-data-1/wuchangjie.wcj/work/bev_ex3_v3_fulldata_resume/Qwen2.5-VL-3B-Instruct`

---

## 四、闭环评测

### 步骤 1：安装 CARLA

**注意：** CARLA 只有非 root 用户才能使用。

安装 CARLA（0.9.16 才有 Python 3.10，Python 3.10 才能用大模型）：

```bash
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.16.tar.gz
tar -xvf CARLA_0.9.16.tar.gz
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.16.tar.gz
bash ImportAssets.sh
```

解压缩之后执行：

```bash
/mnt/nas-data-1/zhanglingjun.zlj1/carla/carla0916/ImportAssets.sh
```

这会把额外的地图给解压缩。

然后创建 Python 3.10 的 CARLA 环境：

```bash
conda activate /mnt/nas-data-1/zhanglingjun.zlj_env/envs/carla/
pip install carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl
```

启动 CARLA：

```bash
./CarlaUE4.sh -RenderOffScreen -nosound -fps=10 -carla-rpc-port=2000
```

`-RenderOffScreen` 说明是无图形化界面运行。

**检查是否正常：**

重点是 `vulkaninfo | grep "GPU id"` 能识别到物理机器，如果能识别，则 CARLA 启动问题不大。

**如果无法正常启动，需要考虑：**

```bash
cd /etc/vulkan
```

看是否 `icd.d/` 下面有无 JSON 文件。

`icd.d/` 下面：

```bash
sudo touch nvidia_icd.json
```

写入：

```json
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "libEGL_nvidia.so.0",
        "api_version" : "1.3.277"
    }
}
```

`implicit_layer.d/` 下面：

```bash
sudo touch nvidia_layers.json
```

写入：

```json
{
    "file_format_version" : "1.0.0",
    "layer": {
        "name": "VK_LAYER_NV_optimus",
        "type": "INSTANCE",
        "library_path": "libEGL_nvidia.so.0",
        "api_version" : "1.3.277",
        "implementation_version" : "1",
        "description" : "NVIDIA Optimus layer",
        "functions": {
            "vkGetInstanceProcAddr": "vk_optimusGetInstanceProcAddr",
            "vkGetDeviceProcAddr": "vk_optimusGetDeviceProcAddr"
        },
        "enable_environment": {
            "__NV_PRIME_RENDER_OFFLOAD": "1"
        },
        "disable_environment": {
            "DISABLE_LAYER_NV_OPTIMUS_1": ""
        }
    }
}
```

```bash
./CarlaUE4.sh -RenderOffScreen -nosound -fps=10 -carla-rpc-port=2000 -graphicsadapter=5
```

### 步骤 2：安装环境

```bash
cd bench2drive
conda create -n b2d python=3.10 -y && conda activate b2d
```

根据不同的 CUDA 选择 torch（vLLM 0.8.0 需要 torch 2.6.0）：

```bash
export PATH=YOUR_GCC_PATH/bin:$PATH
export CUDA_HOME=YOUR_CUDA_PATH/
```

基本就是 `nvcc --version` 可以正常显示，`gcc` 和 `g++ --version` 可以正常显示。

cd 到 bench2drive zoo 文件夹：

```bash
pip install ninja packaging
pip install -v -e .
```

环境中：

```
numba==0.61.2  # In order to speed up
numpy==1.26.4  # In order to adapt numba
```

需要修改 bench2drive 的 requirements 里面的内容。

### 步骤 3：安装 QwenVL 推理的环境

一个参考版本的推理环境示例：`deepsight/example.txt`

**脚本：** [bench2drive/leaderboard/scripts/run_evaluation_qwen.sh](bench2drive/leaderboard/scripts/run_evaluation_qwen.sh)

使用训练好的 Qwen 模型运行 Bench2Drive 评估流水线的 leaderboard 测试，需要安装 `bench2drive/` 目录下的独立推理环境。

---

## 核心组件说明

| 组件 | 说明 |
|------|------|
| `nebula.sh` | Nebula 集群训练任务提交脚本 |
| `requirements.txt` | 训练环境 Python 依赖 |
| `configs/` | 不同训练任务的 YAML 配置文件 |
| `bench2drive/` | Bench2Drive 评估框架（独立的推理环境） |
| `src/tools/` | 数据处理、可视化、评测工具脚本 |
| `src/transformers/` | 修改后的 transformers（集成 DINOv3） |
| `src/llamafactory/data/ad_collator.py` | 自动驾驶数据整理器 |

## 许可证

本项目遵循 [LICENSE](LICENSE) 文件中描述的条款进行分发。
