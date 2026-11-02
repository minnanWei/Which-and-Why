# Qwen3 LoRA：CWE 与证据行加权训练

直接读取 ../dataset/dataresult/traindataset。Qwen3-0.6B、Qwen3-8B、Qwen3-14B 三个基座及对应 LoRA 结构不变，适配新的七字段 joint-evidence，新增字段加权损失和 head:middle:tail=5:3:1 训练调度。

## 仓库与数据放置

本仓库根目录对应本地的 train 目录。克隆代码：

    git clone https://github.com/minnanWei/Which-and-Why.git
    cd Which-and-Why

将处理好的 train.jsonl、validation.jsonl、test.jsonl 放到仓库父目录下的 dataset/dataresult/traindataset/，默认相对路径即可使用。目录关系为：

    工作目录/
      Which-and-Why/                 本仓库代码
      dataset/dataresult/traindataset/
        train.jsonl
        validation.jsonl
        test.jsonl

数据位于其他位置时，用 --train-file 和 --eval-file 指定真实路径；独立测试用 --data-file。当前仓库只发布代码、配置和说明，数据集、基座权重和训练输出分别在运行环境中准备。

## 1. 环境、模型与启动

本地项目目录为 D:\Scholar\joint evidence LVD-VTP\train；克隆后则以仓库根目录为准。所有相对路径以代码所在目录为基准。固定 Python 3.11、torch 2.8.0、Transformers 4.56.2、TRL 0.23.0、PEFT 0.17.1、Accelerate 1.10.1，见 requirements.txt。先安装与实际 GPU/驱动匹配的 PyTorch，再安装其余依赖。

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

仅 CPU 调试时将 torch 安装源改为 https://download.pytorch.org/whl/cpu。需要 QLoRA 时另外安装 requirements-qlora.txt。Linux/WSL 使用 .venv/bin/python，并单独创建环境。

| 模型 | 本地目录 | 配置 |
| --- | --- | --- |
| Qwen3-0.6B | models/Qwen3-0.6B | configs/qwen3_0.6b.json |
| Qwen3-8B | models/Qwen3-8B | configs/qwen3_8b.json |
| Qwen3-14B | models/Qwen3-14B | configs/qwen3_14b.json |

    python download_models.py --sizes 0.6B 8B 14B
    python train_lora.py --config configs/qwen3_0.6b.json --preflight
    python train_lora.py --config configs/qwen3_0.6b.json
    python train_lora.py --config configs/qwen3_8b.json
    python train_lora.py --config configs/qwen3_14b.json

上述 python 使用所建环境的解释器。已有 uv 环境时也可使用 uv run --no-project --python .venv/Scripts/python.exe python。

默认普通 LoRA；--load-in-4bit 启用 NF4 QLoRA。量化不改变 LoRA 参数结构。正式训练要求 CUDA，--device cpu --dtype fp32 仅适合流程调试。训练脚本只从本地加载基座。

## 2. 新数据与过滤

默认数据路径为 ../dataset/dataresult/traindataset 下的 train.jsonl、validation.jsonl 和 test.jsonl。不再读取 train/dataset 中的旧数据。

新字段为 func、cwe_ids、cwe_group 和对象类型 joint-evidence。教师对象含 sample_valid、rejection_reason、evidence_lines、code_evidence、vulnerability_mechanism、cwe_rationale、final_cwe。

按用户确认，训练、验证、测试都跳过 sample_valid=false 的教师记录，原始三个文件不修改：

| 集合 | 原条数 | 无效教师 | 有效样本 |
| --- | ---: | ---: | ---: |
| train | 10,108 | 2,231 | 7,877 |
| validation | 1,263 | 280 | 983 |
| test | 1,264 | 289 | 975 |

这些是长度过滤前的有效数。data_stats.json 记录原文件有效/无效索引，以及显式长度过滤丢弃的索引。训练长度丢弃索引对应有效样本序列，可通过 valid_indices_1based 映射回原文件；评估的丢弃索引直接对应原文件。

有效类别覆盖分别为训练 68 类、验证 66 类、测试 64 类。CWE-19 在训练集中没有有效教师样本，因此按已确认的过滤策略不会有该类别的训练监督；原数据仍保留。valid_cwe_counts 会记录每类的有效数量。

学生输入仅包含完整 func。已有 0001 | 行号保持不变，不重复编号；无行号代码才从 1 编号。真实 CWE、cwe_group、补丁、提交信息、漏洞描述和教师答案不进入输入。cwe_group 仅用于采样。

助手学习同一份七字段 JSON，先输出 final_cwe 和 evidence_lines，再输出其他说明。evidence_lines 的每个元素包含 lines 行号数组和 code 原代码；lines 逐项枚举，不解释为区间端点。括号、else、空行和注释可属于完整证据块。有效教师的 final_cwe 必须与原始 cwe_ids 一致。

预览转换：

    python prepare_data.py --input ../dataset/dataresult/traindataset/train.jsonl --output data/processed/train_messages.jsonl

预览包含 messages、loss_spans 和 cwe_group，不是 --train-file 所需的原始格式。旧的 vulnerability lines / cweid 字符串协议及 --line-format 已移除。

## 3. 加权损失

| token 位置 | 默认权重 |
| --- | ---: |
| final_cwe 的 JSON 值域 | 3.0 |
| evidence_lines 整个数组，含行号、代码和结构符号 | 2.0 |
| 其余助手答案及 EOS | 1.0 |
| system、user、padding | 0 |

用 fast tokenizer 的字符 offset 对齐 JSON 值域，跨边界 token 取覆盖区域的最大权重。不会对输入或解释里偶然出现的 CWE 字符串误加权。

    L = sum(w_t * CE(logits_(t-1), token_t)) / sum(w_t)

分母覆盖整个梯度累积窗口。每个微批贡献分子，labels 和权重与因果预测右移对齐，避免不同长度微批被错误等权，也避免再次除以梯度累积步数。

    python train_lora.py --config configs/qwen3_0.6b.json --cwe-loss-weight 4 --evidence-loss-weight 3

两项都为 1 可做不加权对照。loss 是加权目标，不能与旧版未加权 loss 直接比较数值。生成式验证不使用教师 loss 选择模型。

## 4. 5:3:1 调度

默认 group_ratio=5,3,1。每个连续 9 样本周期固定包含 5 个 head、3 个 middle、1 个 tail，并打乱周期内部顺序。比例指样本曝光数，不是 token 数、loss 比重或仅有期望保证的随机概率。

每组先无放回遍历打乱的样本池，耗尽才重洗循环。一整轮确保每条有效样本至少曝光一次。seed + epoch 控制顺序，续训可重建序列。验证/测试不重采样。

| 组 | 有效唯一样本 | 默认每轮曝光 |
| --- | ---: | ---: |
| head | 6,022 | 6,030 |
| middle | 1,360 | 3,618 |
| tail | 495 | 1,206 |
| 合计 | 7,877 | 10,854 |

尾部曝光占 1/9=11.11%，高于有效训练池的约 6.28%。重采样只发生在加载器，不生成数据副本。

默认 batch_size=1、gradient_accumulation_steps=27，每次参数更新包含 head 15、middle 9、tail 3。完整一轮为 402 次更新。有效 batch 大小必须是周期长度 9 的整数倍；改变 ratio 时也要按新周期对齐。

当前沿用单进程训练及生成式评估；上述更新数按单卡默认配置计算。

如果用 max_steps 或小于一轮的 epochs 提前结束，不保证覆盖整个样本池，但完整更新窗口仍保持比例。显式丢弃超长样本后，调度按保留的各组数量重算。统计写入 data_stats.json。

## 5. 长度与参数

使用官方 Qwen3-0.6B tokenizer 实测，当前有效记录的完整序列最大长度如下：

| 集合 | 提示、模板与答案合计 | 最长教师答案 |
| --- | ---: | ---: |
| train | 12,029 | 2,992 |
| validation | 10,293 | 1,226 |
| test | 13,712 | 5,641 |

默认 max_length=16,384。原 4,096 会使 637 条有效训练记录超长。long_policy=error，不静默截断或丢弃；显存不足时可明确缩短长度并使用 --long-policy drop，但这会改变分布，统计会披露丢弃数量。QLoRA 与 gradient checkpointing 仍可用。

训练验证的生成预算为 2,048。独立预测/测试为 6,144，以容纳较长代码证据；独立测试默认以基座原生上下文作为总上限，可用 --max-length 覆盖。max_new_tokens 是上限，遇 EOS 会提前停止。

| 参数 | 默认值 |
| --- | --- |
| train_file / eval_file | traindataset 下的 train / validation |
| cwe_loss_weight / evidence_loss_weight | 3.0 / 2.0 |
| group_ratio | 5,3,1 |
| gradient_accumulation_steps | 27 |
| max_length | 16384 |
| eval_max_new_tokens | 2048 |
| batch_size / eval_batch_size | 1 / 1 |
| num_train_epochs | 1 |
| LoRA r / alpha / dropout | 32 / 64 / 0.05 |
| learning_rate | 0.6B 为 2e-4；8B/14B 为 1e-4 |

三个已有 JSON 配置保持各模型路径、学习率、LoRA 结构和量化设置。TrainSettings 所有字段均可通过同名连字符 CLI 参数覆盖。布尔参数支持 --no- 前缀。未新增 v1/v2 目录、旧版本备份或训练数据副本。

## 6. 生成式评估与预测

仍以 eval_joint_f1 选择最佳已保存 checkpoint：

    joint_f1 = (CWE micro F1 + evidence line micro F1) / 2

从预测 JSON 读取 final_cwe 和全部 evidence_lines[*].lines 的集合。只评分 CWE/行号，不评价解释文字的语义。参考教师仅用于评分，不进入生成提示。

报告 cwe_accuracy / cwe_exact_match、CWE 和行号 precision/recall/F1、joint_exact_match、format_valid_rate、truncated_samples。单标签有效子集上的 cwe_accuracy 就是 CWE 完全匹配率。无效/截断预测仍留在评分分母。

    python evaluate.py --model-path models/Qwen3-0.6B --adapter-path outputs/Qwen3-0.6B-LoRA/final_adapter --output-dir outputs/Qwen3-0.6B-LoRA/test_results
    python predict.py --model-path models/Qwen3-0.6B --adapter-path outputs/Qwen3-0.6B-LoRA/final_adapter --code-file data/raw/function_to_check.c

最终测试默认使用 traindataset/test.jsonl 的 975 条有效教师记录，不参与 checkpoint 选择。8B/14B 替换对应路径。教师 sample_valid=false 意味着其证据不足，不能直接当作无漏洞负例。

## 7. 日志、续训及验证范围

保持 checkpoint、final_adapter、run_config.json、data_stats.json、TensorBoard、JSONL/CSV loss 日志和逐次验证预测。恢复时保持模型、数据、LoRA、加权系数、采样比例和有效 batch 一致。

    python train_lora.py --config configs/qwen3_0.6b.json --resume-from-checkpoint latest
    python plot_loss.py --log-file outputs/Qwen3-0.6B-LoRA/logs/metrics.jsonl --output outputs/Qwen3-0.6B-LoRA/loss.png
    tensorboard --logdir outputs --port 6006

preflight 不加载模型权重，核对教师过滤、CWE、行号、train/validation 重叠、长度、assistant mask、加权区间和实际曝光统计。原 JSONL 只读。

本次用真实 Qwen3 tokenizer 检查全量有效记录，并用随机初始化的小型 Qwen3 + LoRA 完成手工梯度对照及一次实际参数更新。未启动 0.6B/8B/14B 正式训练；实际显存与任务效果须在训练硬件上测量。

实现对接固定版本的 [Transformers 4.56.2 Trainer](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py)、[TRL 0.23.0 SFTTrainer](https://github.com/huggingface/trl/blob/v0.23.0/trl/trainer/sft_trainer.py) 和 [Accelerate 1.10.1 DataLoader set_epoch](https://github.com/huggingface/accelerate/blob/v1.10.1/src/accelerate/data_loader.py)。
