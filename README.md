# X2DFD - Face Forgery Detection QA System

## 项目概述
X2DFD 是一个基于 LLaVA 视觉语言模型的深度伪造检测问答系统，包含两类核心能力：
- QA 准确性评估：在真实/伪造数据集上评估预设问题的效果并生成排名与指标。
- 图像注释生成：为图像生成“真/伪”判断及可读解释，可结合传统检测器得分进行增强。

## 快速开始

1) 安装环境
- 运行：`bash install.sh`
- 激活：`conda activate X2DFD`

2) 下载权重
- LLaVA-1.5-7B 基座模型（Hugging Face）：
  https://huggingface.co/liuhaotian/llava-v1.5-7b
  - 将下载内容放到：`weights/base/llava-v1.5-7b`
  - 或设置环境变量：`X2DFD_BASE_MODEL=/path/to/llava-v1.5-7b`
- CLIP ViT-L/14-336 视觉塔（Hugging Face）：
  https://huggingface.co/openai/clip-vit-large-patch14-336
  - 将下载内容放到：`weights/base/clip-vit-large-patch14-336`
  - 示例绝对路径：`/data/250010183/workspace/X2DFD/weights/base/clip-vit-large-patch14-336`

3) 准备数据
- 图片目录：`datasets/raw/images`（自行复现数据集的目录结构，如 FaceForensics++）。
- 数据 JSON：使用仓库内 `datasets/raw/data/*.json`。数据格式规则见“数据与路径规则”。

4) 常用命令
- QA 评估：
  `python -m eval.qa.run --config eval/configs/config.yaml`
- 生成注释：
  `python -m eval.tools.feature_annotation --config eval/configs/config.yaml`
- 快速推理：
  `python -m eval.infer.runner --config eval/configs/infer_config.yaml --json datasets/raw/data/test/Deepfakes_tiny.json`
- 端到端流水线（注释 -> 合并 -> LoRA 训练 -> LoRA 测试）：
  `./train.sh --run-train`
  - 仅测试：`./train.sh --test-only`
  - 复用已有注释：`./train.sh --skip-annotate`

4) 常用环境变量
- `X2DFD_BASE_MODEL`：覆盖基座模型路径。
- `X2DFD_DATASETS`：自定义数据集根目录。
- `X2DFD_OUTPUT`：评估/推理结果根（默认 `eval/outputs`）。
- `X2DFD_TRAIN_OUTPUT`：训练流水线输出根（默认 `train/outputs`）。

## 核心功能

### 1. QA 准确性评估（python -m eval.qa.run）
- 功能：在真实与伪造数据集上评估问题表现，生成平衡准确率、排名与详细记录。
- 关键参数来源：`eval/configs/config.yaml`（`paths/files/run_params/output/model` 等）。
- 典型输出：见“结果位置”。

### 2. 图像注释生成（python -m eval.tools.feature_annotation）
- 功能：为真实/伪造图像生成结构化解释；可结合传统检测器打分生成评分提示与支持语句。
- 参数来源：`eval/configs/config.yaml` 的 `annotations` 与 `weak_supply` 段。
- 典型输出：见“结果位置”。

## 结果位置（统一）

- QA 评估
  - 根目录：`eval/outputs/qa_runs/`
  - 产物：`run_*/qa_summary.json`、`metrics/*`、`questions/*`、`responses/*`
  - 指针：`eval/outputs/qa_runs/latest_run.json`

- 快速推理
  - 根目录：`eval/outputs/infer/runs/infer_*/datasets/`
  - 产物：`*_result.json`（每个输入 JSON 的结果，`image` 为绝对路径）
  - 指针：`eval/outputs/infer/latest_run.json`

- 注释生成
  - 根目录：`eval/outputs/annotations/runs/ann_*/`
  - 产物：`datasets/*_annotations.json`、`datasets/*_annotations_scored.json`、`all_annotations_merged.json`
  - 进度：`eval/outputs/annotations/progress.json`

- 训练流水线（端到端）
  - 根目录：`train/outputs/pipeline/runs/pipeline_*/`
  - 产物：`train/…`、`test/…`、阶段标记 `.annotations.done/.merge.done/.training.done/.test.done`
  - 记录：`pipeline_info.json`

## 数据与路径规则
- Dataset 风格 JSON：顶层提供 `Description`（图片根前缀），条目位于 `images[].image_path`。
- 对话风格 JSON：列表条目包含绝对 `image` 路径与 `conversations`。
- 写入路径（很重要）：所有生成的结果文件（评估/推理/注释）的 `image` 字段均写入为操作系统上的绝对路径；不会再拼接任何全局前缀，也不会在结果里保留 `Description`。

最小 JSON 示例（Dataset 风格）
```
{
  "Description": "/abs/path/to/datasets/raw/images/FaceForensics++",
  "images": [
    { "image_path": "manipulated_sequences/Deepfakes/c23/00001.png" },
    { "image_path": "/abs/path/to/another/image.png" }
  ]
}
```
- 若缺少 `Description`，则 `images[].image_path` 必须全部为绝对路径；否则会报错。
- 对话风格 JSON（注释/推理可读）：`[{"id":"1","image":"/abs/img.png","conversations":[{"from":"human","value":"<image>\n..."},{"from":"gpt","value":""}]}]`，其中 `image` 必须为绝对路径。

## 依赖环境
- Python 3.8+
- PyTorch
- LLaVA 模型依赖
- tqdm（进度条）
- PyYAML（配置管理）

## 开发状态
- ✅ QA 准确性评估系统完整实现
- ✅ 图像注释生成系统完整实现
- ✅ LLaVA 模型集成完成
- ✅ 配置管理和结果持久化完成
- ✅ 进度跟踪和错误处理完成

## 训练/测试 JSON 规范
- 放置位置
  - 训练集 JSON：`/data/250010183/workspace/X2DFD/datasets/raw/data/train/`（由 `train/configs/config.yaml: paths.data_dir` 指定）
  - 测试/推理 JSON：`/data/250010183/workspace/X2DFD/datasets/raw/data/test/`（由 `eval/configs/infer_config.yaml` 或 `train/configs/config.yaml: infer.inputs` 指定）

- 文件命名与分组
  - 训练集分组通过配置列表划分标签：
    - `train/configs/config.yaml: annotations.real_files` 列表内文件视为 real
    - `train/configs/config.yaml: annotations.fake_files` 列表内文件视为 fake
  - 测试集用于 AUC 统计时，建议测试 JSON 基名包含 `real` 或 `fake` 以便分组：
    - 例如：`Real.json` → 结果 `Real_result.json` 归入 real；`Deepfakes_tiny.json` 含 `fake` 子串 → 归入 fake
    - 若基名不含 `fake`（如 `Faceswap_tiny.json`），其结果会被 AUC 分组忽略（不影响其它数据集参与计算）

- 字段要求（Dataset 风格，推荐）
  - 顶层 `Description`：必须是图片根目录的绝对路径，例如：`/data/250010183/Datasets/FaceForensics++` 或 `/data/250010183/Datasets`
  - `images[].image_path`：可以是相对 `Description` 的相对路径，或直接为绝对路径；必须指向存在的文件
  - 列表非空；建议每个 JSON 表示单一标签的数据集（real 或 fake）

- 写出行为（适用于训练注释、测试推理与 QA 评估）
  - 结果 JSON 中的 `image` 字段写为绝对路径，不再保留 `Description`
  - 单次运行的所有产物集中写入统一的 run 目录，并提供 `latest_run.json` 指针
