# 重构迁移清单

## 顶层目录规划
- utils/
- src/
- datasets/
- train/
- eval/
- weights/
- demo.py
- README.md (精简)

## 文件/目录映射
| 当前 | 目标 | 备注 |
| --- | --- | --- |
| README.md | README.md | 更新内容描述 |
| data/ | datasets/raw/ | 原始数据 JSON/图片 |
| datasets_explain/ | datasets/annotations/ | 运行时生成的注释（仓库内提供空目录） |
| Prompt/ | datasets/prompts/ | prompt 模板等 |
| feature_annotation.py | eval/tools/feature_annotation.py | 并拆分 API |
| question_evaluation.py | eval/qa/run.py | 重命名为包入口 |
| quick_infer.py | eval/infer/runner.py | ｜
| quick_infer.sh | eval/scripts/quick_infer.sh | 调整命令 |
| weak_supply.py | eval/tools/weak_supply.py | |
| utils/* | utils/* | 保留，新增 paths.py |
| model_train.py | train/lorra/train.py | LoRA 训练主脚本 |
| pipeline_unified.py | train/pipeline.py | 流水线入口 |
| sanity_train_setup.py | train/tools/sanity_setup.py | |
| run_train.sh | train/scripts/run_train.sh | |
| run_all.sh | train/scripts/run_all.sh | |
| run1.sh, run2.sh | train/scripts/run_stage_{1,2}.sh | 重命名 |
| run_qa*.sh | eval/scripts/run_qa*.sh | 全部指向 -m 方式 |
| config.yaml | eval/config.yaml | QA/注释配置 |
| config_small.yaml | eval/config_small.yaml | 小样本配置 |
| infer_config.yaml | eval/infer_config.yaml | 快速推理配置 |
| uniconfig.yaml | train/config.yaml | 流水线+训练配置 |
| uniconfig_small.yaml | train/config_small.yaml | 小样本训练 |
| infer_debug/ | eval/debug/ | |
| ckpt/ | weights/checkpoints/ | 若存在则移动 |
| results/ | eval/outputs/ | 统一输出根 |
| weights/ | weights/ | 结构内细化 (如 blending_models) |

## 额外任务
- 新建 demo.py，硬编码 eval 演示流程。
- README 精简结构。
- 新增 utils/paths.py，提供 PROJECT_ROOT / DATASETS_ROOT / OUTPUT_ROOT / WEIGHTS_ROOT。
- 阶段跳过：train/pipeline 与 eval/qa 在运行前检测上一阶段 `.done` 标记。
- Shell 脚本更新，以 `python -m eval.qa.run` 等形式调用。
