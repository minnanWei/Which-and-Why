# Qwen3 + PEFT LoRA：行级漏洞预测与 CWE 类型识别

本项目把 `func-before` 作为用户提示中的函数代码，将 `joint-evidence`、`vulnerability lines` 和 `cwe-id` 拼成助手答案，用 PEFT LoRA 和 TRL SFTTrainer 微调。只对助手答案及结束标记计算 loss。提供模型下载、数据检查、训练、断点恢复、日志保存、loss 绘图和适配器推理。

## 1. 使用的模型

根据后续要求，选择 **Qwen3-0.6B、Qwen3-8B、Qwen3-14B**。在官方 Qwen3 稠密文本模型系列中，14B 是大于 10B 的最小档。这里明确使用 Qwen3，不将最初的 `Qwen3.8 / 17B` 当作下载仓库名，也不替换成 1.7B。

| 模型 | 官方下载页 | 本地目录 | 配置 |
| --- | --- | --- | --- |
| 0.6B | https://huggingface.co/Qwen/Qwen3-0.6B | `models/Qwen3-0.6B` | `configs/qwen3_0.6b.json` |
| 8B | https://huggingface.co/Qwen/Qwen3-8B | `models/Qwen3-8B` | `configs/qwen3_8b.json` |
| 14B | https://huggingface.co/Qwen/Qwen3-14B | `models/Qwen3-14B` | `configs/qwen3_14b.json` |

官方模型系列说明：https://qwenlm.github.io/blog/qwen3/

这三个配置默认都是普通 LoRA：基座权重保持冻结、以浮点精度加载，训练新增的 LoRA 参数。加上 `--load-in-4bit` 会改为 **QLoRA**：冻结基座使用 NF4 量化，仍训练 LoRA 参数。量化模式需要额外安装 `requirements-qlora.txt`。

14B 的 BF16 基座权重本身约 28 GB（十进制估算），训练还需要激活、梯度及优化器等内存。8B 基座约 16 GB。不能用权重体积直接推断训练显存；没有足够显存时降低 batch/长度，或启用 QLoRA。本项目没有声称在你的 GPU 上验证过显存占用。

## 2. 文件与目录

```text
train/
  requirements.txt              固定直接依赖版本
  requirements-qlora.txt        加装 bitsandbytes
  download_models.py           下载模型，支持断点缓存和版本记录
  prepare_data.py               原始数据 -> messages JSONL 预览
  data_utils.py                 字段校验、提示词、掩码与长度检查
  train_lora.py                 LoRA/QLoRA 训练入口
  chat_template.jinja           Qwen3 非 thinking 模板与助手 loss 范围
  logging_utils.py              Trainer callback，实时保存 JSONL/CSV
  plot_loss.py                  由日志绘制 train/eval loss
  predict.py                    加载基座和适配器进行预测
  evaluate.py                   独立验证/测试 CWE 和漏洞行
  evaluation.py                 生成、解析及任务指标计算
  task_trainer.py               训练中按任务指标验证和选择 checkpoint
  configs/                     三档模型参数
  models/                      下载的原始模型；不是训练输出
  dataset/                     train.jsonl、validation.jsonl、test.jsonl
  data/raw/                    可选的待预测函数文件
  data/processed/              可选的转换预览输出
  outputs/                     每次实验单独一个输出目录
```

项目位置：`D:\Scholar\joint evidence LVD-VTP\train`。脚本中的相对数据、模型、配置和输出路径均相对于项目根目录解析，不受启动目录影响；也支持绝对路径。

## 3. 使用 uv 安装环境

本项目固定 Python 3.11、PyTorch 2.8.0、Transformers 4.56.2、TRL 0.23.0、PEFT 0.17.1。固定版本用于保持本项目接口一致，不代表它们是最新版。不要单独升级 TRL 或 Transformers 后继续假设模板/数据整理行为不变。

Windows PowerShell：先按 [uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/) 安装 uv，也可以用 Windows 包管理器：

```powershell
winget install --id astral-sh.uv -e
# 安装后打开新 PowerShell
Set-Location 'D:\Scholar\joint evidence LVD-VTP\train'
uv python install 3.11
uv venv --python 3.11 .venv

# NVIDIA GPU，示例使用 PyTorch 官方 CUDA 12.8 wheel
uv pip install --python .venv/Scripts/python.exe torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
uv pip check --python .venv/Scripts/python.exe

# 显式指定本项目解释器；不依赖激活虚拟环境
uv run --no-project --python .venv/Scripts/python.exe python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

CUDA wheel 应与显卡/驱动兼容；另有 cu126/cu129，选择见 [PyTorch 2.8 官方安装命令](https://pytorch.org/get-started/previous-versions/#v280)。训练前确认 `torch.cuda.is_available()` 为 `True`。单纯安装 CUDA Toolkit 不会让 CPU 版 torch 变成 CUDA 版。

只运行 CPU 小规模调试时，第一条 torch 安装命令改成：

```powershell
uv pip install --python .venv/Scripts/python.exe torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

需要 QLoRA 时再执行：

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-qlora.txt
```

Linux/WSL 中把命令里的 `.venv/Scripts/python.exe` 换为 `.venv/bin/python`。WSL 中此目录路径为 `/mnt/d/Scholar/joint evidence LVD-VTP/train`；Linux/WSL 环境和 Windows 环境分别安装，不能共用 `.venv`。大规模训练优先在 Linux/WSL CUDA 环境运行。

`requirements.txt` 固定直接依赖；第一次安装后，如需复现实验的全部传递依赖，可保存环境快照：

```powershell
uv pip freeze --python .venv/Scripts/python.exe > requirements-lock-local.txt
```

该快照受操作系统、CUDA wheel 和 Python 版本影响；在相同平台恢复时需保留相同的 PyTorch 安装源。官方 uv 锁定说明：https://docs.astral.sh/uv/pip/compile/

## 4. 下载模型

```powershell
# 先用 0.6B 验证自己的完整流程
uv run --no-project --python .venv/Scripts/python.exe python download_models.py --sizes 0.6B

# 另外两档；也可只选择其中一个
uv run --no-project --python .venv/Scripts/python.exe python download_models.py --sizes 8B 14B

# 如需同时下载三档
uv run --no-project --python .venv/Scripts/python.exe python download_models.py --sizes 0.6B 8B 14B
```

下载的是原始 Transformers/safetensors 模型，不是 GGUF/AWQ/GPTQ 包。脚本自动创建模型子目录，重复运行会复用 Hugging Face 下载缓存；成功后记录 `download_manifest.json`，含仓库名和 commit SHA。可以用 `--revision <commit SHA>` 固定版本。已完成目录的版本不同会报错，需另选目录，避免混入旧权重。

其他选项：`--workers 4`、`--models-dir models`、`--tokenizer-only`。`--tokenizer-only` 仅用于数据预检查，无法直接训练正式模型；之后不带该参数重复下载可补齐权重。

`--model-id owner/repo` 可下载自定义仓库，目录名为 `owner--repo`，随后修改 `model_path`。本项目训练路径仅支持 Qwen3 稠密文本架构；新模型架构不能仅靠替换仓库名保证兼容。

如直连失败，可通过 `--endpoint <你信任的 HF 兼容端点>` 配置下载端点。私有仓库使用 `HF_TOKEN` 环境变量；不要把 token 写进代码或配置。正式模型权重不随本项目附带。

## 5. 数据格式与提示词

### 当前 dataset 检查结果（2026-10-22）

| 文件 | 样本数 | joint-evidence | 漏洞行格式 |
| --- | --- | --- | --- |
| `dataset/train.jsonl` | 7758 | 全部存在，numbers 模式字段检查全部通过 | 整数行号列表 |
| `dataset/validation.jsonl` | 1156 | 不需要此字段 | 整数行号列表 |
| `dataset/test.jsonl` | 1156 | 不需要此字段 | 整数行号列表 |

三个文件均为有效 JSONL。训练集需要 `joint-evidence`，用于三字段助手答案的 SFT；验证集和测试集只需要 `func-before`、`cwe-id` 与 `vulnerability lines`。评估时只把函数和任务提示交给模型，不向模型提供真实标签；生成后仅比较 CWE 和漏洞行，忽略 evidence。当前三个文件的相应字段检查均已通过，不需要为验证或测试补写 evidence。

按函数源码归一化空白后的哈希检查：train/validation 重叠 23 个不同函数，train/test 重叠 23 个，validation/test 重叠 6 个。各文件内部额外重复样本数分别为 1057、2、5。这不是近似克隆检查。训练脚本会拒绝 train/validation 重叠，需要在原始数据中重新划分或去重；原始数据保持不变。

本次训练集所有行号均通过函数内从 1 开始的边界检查；边界合法不能独立证明原始标注采用同一行号定义，仍应按数据来源确认。尚未加载 tokenizer 检查正式数据的 token 长度，也未启动训练。


默认读取 `dataset/train.jsonl` 和 `dataset/validation.jsonl`；`dataset/test.jsonl` 留作最终测试，不参与训练或 checkpoint 选择。也支持 UTF-8 JSON 数组文件；JSONL 必须每行一个完整 JSON 对象，字符串中的换行写作 `\n`，不能把一个对象直接跨物理行书写。空文件、缺失字段和无效标注会报错，不会静默丢弃。

下面是代码行文本模式的格式示例；你当前数据实际使用整数行号，已将训练和预处理的默认模式改为 `numbers`。整数行号模式的同一示例应将 `vulnerability lines` 写为 `[2]`：

```json
{"git-url":"https://example.invalid/repo","cwe-id":"['CWE-476']","func-before":"int read_value(int *p) {\n    return *p;\n}","vulnerability lines":["    return *p;"],"joint-evidence":"参数 p 可能为空，直接解引用可能导致空指针访问。"}
```

字段处理规则：

| 原始字段 | 用途 |
| --- | --- |
| `func-before` | 原函数代码，放入 user 提示词；不执行代码 |
| `joint-evidence` | 仅训练集必需；形成训练答案的 `evidence`，验证/测试不读取或评分 |
| `vulnerability lines` | 列表，形成 `vulnerability lines` |
| `cwe-id` | 支持真正的列表和 `"['CWE-476']"` 这类列表字符串，形成 `cweid` |
| `git-url` | 不放入模型输入；在原始文件中保留，可用于项目级划分 |

system 提示词：

> 你是一名代码安全审计专家。请基于给定函数完成行级漏洞预测和 CWE 漏洞类型识别，并给出依据。

user 提示词包含以下任务要求和原始函数：

```text
请分析以下函数，为我完成行级漏洞预测和漏洞类型识别任务。
严格输出：evidence:<依据>;vulnerability lines:<JSON列表>;cweid:<JSON列表>。
vulnerability lines 为漏洞代码行的字符串列表，逐字保留原代码行。 cweid 为 CWE 编号列表；无漏洞时两个列表均为空。
evidence 为 JSON 字符串，以转义换行和引号。不要输出额外标题或 Markdown。
<source_code>
int read_value(int *p) {
    return *p;
}
</source_code>
```

对应 assistant 答案：

```text
evidence:"参数 p 可能为空，直接解引用可能导致空指针访问。";vulnerability lines:["    return *p;"];cweid:["CWE-476"]
```

保留你要求的三个标签及顺序；依据用 JSON 字符串编码，列表用 JSON 编码，避免引号/换行破坏 JSONL。不要简单按所有分号切分答案：依据和源码本身也可能有分号。`evaluation.py` 只定位并 JSON 解码末尾的两个任务字段，不比较 evidence 的文字、JSON 编码或是否出现。

无漏洞样本使用 `"cwe-id": []`、`"vulnerability lines": []`，训练样本的依据仍应是字符串；验证和测试无需依据。脚本不会凭空为数据增加正负标签，也不改写 evidence 内容。

### 两种漏洞行表示

- `line_format: "code"`（可选）：每项是原函数内的代码行字符串，包含原有缩进；检查每项是否确实存在。相同代码行在函数中多次出现时，仅靠文本不能唯一定位，建议改成行号标注。
- `line_format: "numbers"`（当前默认）：每项是函数内从 1 开始的整数行号，例如 `[2, 5]`；user 中自动给函数添加行号。空行计数，不删除函数开头空行。禁止零基、负数、越界行号。

如果你的标签是源文件绝对行号或从 0 开始的行号，先根据函数起始位置转换。脚本不猜测偏移。模型输入与输出使用同一份函数、同一套行号定义；不要在转换前单独去掉空行/缩进。

预览转换结果（不需要模型权重，也不需要 torch）：

```powershell
uv run --no-project --python .venv/Scripts/python.exe python prepare_data.py --input dataset/train.jsonl --output data/processed/train_messages.jsonl
# 当前默认是 numbers；若标签改为代码行字符串，添加 --line-format code
```

`train_lora.py` 直接读取**原始字段格式**并内部转换；不要把这个预览输出当作 `--train-file`，它只有 `messages` 字段。

正式实验请预先划分 train/validation/test。验证集用于选择 checkpoint；测试集留待最终评测，不能把 test 当验证集。脚本检测 train/validation 中空白归一化后完全相同的函数；近似克隆和同项目泄漏仍需在数据划分阶段处理，推荐按仓库/项目或时间划分。

## 6. 训练命令

先下载模型，并检查 `dataset/train.jsonl` 和 `dataset/validation.jsonl` 的必需字段与数据重叠。以下全部从项目根目录执行：

```powershell
# 仅进行字段、重叠、token 长度和 assistant mask 检查，不加载权重
uv run --no-project --python .venv/Scripts/python.exe python train_lora.py --config configs/qwen3_0.6b.json --preflight

# 正式训练三档，分别执行
uv run --no-project --python .venv/Scripts/python.exe python train_lora.py --config configs/qwen3_0.6b.json
uv run --no-project --python .venv/Scripts/python.exe python train_lora.py --config configs/qwen3_8b.json
uv run --no-project --python .venv/Scripts/python.exe python train_lora.py --config configs/qwen3_14b.json

# 显存有限时启用 QLoRA，并另设输出目录
uv run --no-project --python .venv/Scripts/python.exe python train_lora.py --config configs/qwen3_14b.json --load-in-4bit --output-dir outputs/Qwen3-14B-QLoRA
```

先处理前述 train/validation 重叠，再跑短训练确认硬件/真实数据可用（与正式实验分开保存）：

```powershell
uv run --no-project --python .venv/Scripts/python.exe python train_lora.py --config configs/qwen3_0.6b.json --max-steps 10 --logging-steps 1 --eval-steps 5 --save-steps 5 --output-dir outputs/Qwen3-0.6B-debug
```


所有 dataclass 配置字段都可通过命令行覆盖，JSON 中下划线改为连字符，例如 `--learning-rate 1e-4`、`--gradient-accumulation-steps 12`、`--line-format numbers`。布尔参数支持 `--gradient-checkpointing` 和 `--no-gradient-checkpointing`、`--load-in-4bit` 和 `--no-load-in-4bit`。命令行优先于 JSON，JSON 优先于代码默认值。查看全部参数：

```powershell
uv run --no-project --python .venv/Scripts/python.exe python train_lora.py --help
```

### 主要参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `model_path` | `models/Qwen3-0.6B` | 本地模型目录；训练使用 local_files_only |
| `train_file` / `eval_file` | `dataset/train.jsonl` / `dataset/validation.jsonl` | JSON/JSONL；JSON 配置中 `eval_file: ""` 可关闭验证 |
| `output_dir` | `outputs/Qwen3-0.6B-LoRA` | 新实验用新目录，非空目录不允许直接覆盖 |
| `line_format` | `numbers` | `code` 或 `numbers`，与标注一致 |
| `max_length` | 4096 | **system + user + assistant + 模板**的 token 总上限，不是字符数 |
| `long_policy` | `error` | 超长立即报错；可选 `drop` 并记录丢弃样本序号；不支持静默截断 |
| `r` / `lora_alpha` | 32 / 64 | LoRA 秩与缩放系数；alpha=2r 是本项目起点，不是强制公式 |
| `lora_dropout` | 0.05 | LoRA dropout |
| `target_modules` | `all-linear` | 也可写 `q_proj,v_proj` 或完整的逗号分隔投影层名称 |
| `batch_size` / `eval_batch_size` | 1 / 1 | 训练 batch / 验证生成 batch；长输入建议从 1 开始 |
| `eval_max_new_tokens` | 1024 | 验证生成输出预算，包括模型生成的 evidence；若答案截断则提高预算 |
| `gradient_accumulation_steps` | 24 | 有效 batch = 每卡 batch × 累积次数 × GPU 数 |
| `num_train_epochs` | 1 | 轮数 |
| `max_steps` | -1 | 正数时覆盖 epoch 训练长度，指 optimizer steps |
| `learning_rate` | 0.0002 | 0.6B 起点；8B/14B 配置用 0.0001；应结合验证集调整 |
| `warmup_ratio` | 0.1 | cosine 学习率调度的 warmup 比例 |
| `weight_decay` | 0.0 | AdamW 权重衰减 |
| `logging_steps` | 10 | 每多少 optimizer steps 记录一次，第一步也记录 |
| `eval_steps` / `save_steps` | 100 / 100 | save_steps 必须是 eval_steps 整数倍，便于最佳模型选择 |
| `save_total_limit` | 3 | checkpoint 保留上限；HF 可能额外保留最佳 checkpoint |
| `gradient_checkpointing` | true | 以额外计算换显存；使用 non-reentrant 模式 |
| `load_in_4bit` | false | true 启用 NF4 + double quant 的 QLoRA |
| `dtype` | auto | CUDA 上优先 bf16，否则 fp16；CPU 调试 fp32 |
| `device` | auto | auto 要求 CUDA；cpu 只建议用于小规模流程验证 |
| `seed` | 42 | 固定随机种子；不同硬件/分布式条件仍可能存在数值差异 |

代码固定启用 `assistant_only_loss=True`、`packing=False`、SDPA、`adamw_torch`、`max_grad_norm=1.0`；这些实现参数若需变更，在 `train_lora.py` 的 `SFTConfig` 中修改。未启用 FlashAttention/Liger/offloading，避免安装额外内核作为基本运行条件。

长度超限时应优先审查完整样本和 token 统计，再提高 `max_length`；若显存允许也可增大长度。`--long-policy drop` 会改变训练/验证分布，丢弃样本索引写入 `data_stats.json`，正式报告应披露丢弃数量。默认 4096 不保证覆盖任意长函数。

## 7. 训练实现与原示例的对应关系

1. 初始化 `AutoTokenizer` 与本地模型配置，补齐原示例中未定义的 `tokenizer`。
2. 校验原始数据、构建 system/user/assistant；不用“其他字段一律当 assistant”的映射。
3. 用 `chat_template.jinja` 编码，每条样本生成 `input_ids` 和 `assistant_masks`。
4. 模板在 assistant 正文及 `<|im_end|>` 周围标记 `{% generation %}`，不对 system/user/padding 计算 loss。逐条解码被掩码选中的 token，要求恰好等于完整答案和 EOS。
5. 以 Qwen3 非 thinking 形式使用固定空 `<think>...</think>` 前缀，它属于提示，不训练隐藏推理文本；`joint-evidence` 是正常可见答案。预测脚本复用保存的模板及提示词。
6. `LoraConfig` + `get_peft_model` 包装基座；QLoRA 时先调用 `prepare_model_for_kbit_training`。
7. SFTTrainer 使用预编码数据和 `skip_prepare_dataset=True`，避免再次套模板、截断或覆盖已验证的掩码。`messages` 仅保留用于 conversational 格式识别，collator 使用 token IDs 和 masks。
8. `TaskSFTTrainer.evaluate()` 用函数提示进行贪心生成，只计算 CWE/漏洞行指标；`LossLoggerCallback` 每次 on_log 保存训练 loss 和任务指标，TensorBoard 同步保存 events。
9. 训练结束保存适配器、tokenizer、状态和评估结果。

因此不需要另外设置 `TENSORBOARD_LOGGING_DIR`：真实事件目录由 `SFTConfig(logging_dir=...)` 指定。

## 8. 输出、最佳模型与续训

```text
outputs/Qwen3-0.6B-LoRA/
  checkpoint-100/              adapter + optimizer/scheduler/RNG/trainer 状态
  final_adapter/               最终导出的 LoRA 权重、adapter_config 和 tokenizer
  logs/metrics.jsonl           全部 on_log 指标、step、epoch、UTC 时间和 run_id
  logs/loss.csv                训练 loss/lr/grad_norm 及验证任务指标
  predictions/                 每次验证的原始答案、标签、解析状态与截断标记
  tensorboard/                TensorBoard event 文件
  run_config.json             初始配置
  resume_config.json          最近一次续训配置（若有）
  data_stats.json             token 数与丢弃样本索引
  versions.json               训练主要库版本
  selection.json              最佳 checkpoint 及对应指标
  trainer_state.json          Trainer 状态与 log_history
  log_history.json            额外保存的日志历史
  train_results.json          训练汇总
  eval_results.json           最终评估汇总（有验证集时）
```

最佳模型按 `eval_joint_f1 = (eval_line_f1 + eval_cwe_f1) / 2` 选择，越大越好，启用 `load_best_model_at_end`。`final_adapter` 保存任务分数最高的**已保存并评估**的 checkpoint；若短训练未到保存间隔，则保存末尾权重。`selection.json` 记录指标名称和最佳 checkpoint。最终 evaluate 评估的是已恢复的最佳权重。验证不计算 evidence loss，也不计算 eval_loss。

`final_adapter` 不是完整基座模型；推理必须同时加载原始基座。请保留原基座下载目录与 commit 记录。`trainer_state.json` 本身也不是完整 checkpoint；优化器等状态在 `checkpoint-N` 中。

```powershell
# 使用同一配置、原输出目录，恢复最近的 checkpoint
uv run --no-project --python .venv/Scripts/python.exe python train_lora.py --config configs/qwen3_0.6b.json --resume-from-checkpoint latest

# 指定 checkpoint；max_steps 表示总目标步数，不是再训练的步数
uv run --no-project --python .venv/Scripts/python.exe python train_lora.py --config configs/qwen3_0.6b.json --resume-from-checkpoint outputs/Qwen3-0.6B-LoRA/checkpoint-100 --max-steps 500
```

续训保持原模型/数据/LoRA 结构等参数一致，并勿更改原数据内容。按 Ctrl+C 普通中断时，已追加的日志会保留，finally 会尝试写 Trainer 状态；进程被强杀/断电时不能保证最后一条日志落盘。恢复只能回到最近的完整 checkpoint；尚未保存的优化步骤会丢失。

每次启动 callback 有新的 `run_id`，续训追加原 JSONL/CSV。画图时同一 step、同一指标取最后一条记录，适合恢复后覆盖重跑区间；要比较不同实验，应使用独立输出目录。

## 9. 保存日志与绘制 loss

日志脚本 `logging_utils.py` 已自动注册。`loss` 是完整三字段训练答案的区间平均交叉熵；验证只记录 CWE/漏洞行任务指标，因此当前实验没有 eval_loss，loss 图只有训练曲线。验证指标可在同一 JSONL/CSV 或 TensorBoard 查看；旧版日志的 eval_loss 列保留兼容但新记录为空。续训已有旧 CSV 时保留其表头，新任务指标始终完整保存在 JSONL/TensorBoard。

```powershell
# 从 JSONL 画图（也可传 loss.csv 或 trainer_state.json）
uv run --no-project --python .venv/Scripts/python.exe python plot_loss.py --log-file outputs/Qwen3-0.6B-LoRA/logs/metrics.jsonl --output outputs/Qwen3-0.6B-LoRA/loss.png

# 最近 5 个观测值的滑动平均，只改图，不改原日志
uv run --no-project --python .venv/Scripts/python.exe python plot_loss.py --log-file outputs/Qwen3-0.6B-LoRA/logs/loss.csv --window 5 --output outputs/Qwen3-0.6B-LoRA/loss_smoothed.png

# 实时看 TensorBoard
uv run --no-project --python .venv/Scripts/python.exe tensorboard --logdir outputs --port 6006
```

浏览器打开 http://localhost:6006 。PNG 可用于汇报，`--output .../loss.pdf` 可输出 PDF 图。平滑窗口按观测点数计算。验证采用自回归生成，耗时高于一次前向 loss 评估，可适当增大 eval_steps。

正式任务效果由下面的生成式任务评测给出，不以训练 loss 替代任务指标。

## 10. 加载适配器预测

将待检测的单个函数保存成 UTF-8 文件，例如 `data/raw/function_to_check.c`：

```powershell
uv run --no-project --python .venv/Scripts/python.exe python predict.py --model-path models/Qwen3-0.6B --adapter-path outputs/Qwen3-0.6B-LoRA/final_adapter --code-file data/raw/function_to_check.c
```

推理脚本从 adapter 父目录的 `run_config.json` 读取行号模式，并从 adapter 目录加载训练时保存的 tokenizer/template；因此移动适配器时保留项目的输出结构。QLoRA 导出的 adapter 也可配原始浮点基座加载；显存不足时推理增加 `--load-in-4bit`。生成采用确定性贪心策略，`--max-new-tokens 1024` 控制输出上限；输出被截断时要提高该值并检查上下文预算。


### 任务指标与最终测试

验证/测试数据只用三项：`func-before`、`vulnerability lines`、`cwe-id`，无需 `joint-evidence`。模型输入与训练提示一致，因此可能先输出 evidence，但这部分完全不评分；两个任务字段也可直接输出为 `vulnerability lines:[...];cweid:[...]`。严禁把真实 CWE、漏洞行或真实 evidence 拼入评估输入。

- `line_precision/recall/f1`：对每条函数的预测行集合与真实行集合计数，再在全数据集累加 TP/FP/FN，计算 micro 指标。不同函数的同一行号不会混为同一标签。
- `cwe_precision/recall/f1`：每条样本的 CWE 集合比较，累加后计算 micro 指标，支持多标签。
- `line_exact_match`、`cwe_exact_match`：对应预测集合与真实集合完全相同的样本比例；忽略元素顺序和重复值。
- `joint_exact_match`：两项都完全正确的比例；`joint_f1` 为两项 micro F1 的算术平均，用于选择 checkpoint。
- `format_valid_rate`：两个任务字段均可解析且标签合法的比例；无效行字段按空预测计，仍可单独评分合法的 CWE，反之亦然。无法解析完整任务字段时两项均无效，不会丢弃这些样本。
- `truncated_samples`：生成达到预算且未生成 EOS 的样本数。答案若未完整输出，字段可能解析失败。

所有比例范围是 [0,1]。micro precision/recall/F1 分母为零时取 0；因此全为无漏洞样本的数据可出现 exact_match=1、micro F1=0，应同时报告二者。这里只提供 micro 与集合完全匹配，不将 CWE exact_match 误称为单标签 accuracy。

每次验证将 JSONL 预测保存到 `outputs/<实验>/predictions/`，包含原始模型答案、真实标签、解析标签、错误及截断标记；这些属于正式实验结果。为避免不正确的分布式评估，当前训练/生成评估仅支持单进程；单卡启动即可。

最终测试在选好模型后单独运行，默认读取 `dataset/test.jsonl`，不影响 checkpoint 选择：

```powershell
uv run --no-project --python .venv/Scripts/python.exe python evaluate.py --model-path models/Qwen3-0.6B --adapter-path outputs/Qwen3-0.6B-LoRA/final_adapter --data-file dataset/test.jsonl --output-dir outputs/Qwen3-0.6B-LoRA/test_results
```

8B/14B 替换对应模型和实验路径；量化推理加 `--load-in-4bit`。可设 `--batch-size 1 --max-new-tokens 1024 --max-length 4096`。输出为 `metrics.json` 和 `predictions.jsonl`；目录须为空，避免覆盖已有结果。

验证和测试的长度检查针对“完整输入提示 + 生成预算”，必须不超过 max_length 和模型上下文上限。默认遇到超长样本报错，显式 `--long-policy drop` 时记录原始样本序号，并披露评测样本数，防止静默改变测试集。训练配置对应参数为 `eval_max_new_tokens`。训练中的验证长度策略仍由 `long_policy` 控制。

## 11. 故障排查

常见问题：

| 现象 | 处理 |
| --- | --- |
| tokenizer 未定义 | 本项目已初始化并传给 `processing_class` |
| 缺少 config/权重 | 先下载；tokenizer-only 不能代替权重下载 |
| 无 CUDA | 检查 torch wheel 和驱动；CPU 仅用于流程调试 |
| CUDA out of memory | batch=1，打开 gradient checkpointing，审查 max_length，必要时启用 QLoRA |
| bf16 不支持 | 使用 `--dtype fp16` 或 auto |
| assistant mask 错误 | 确保使用本项目模板及固定版本；不删除 generation/endgeneration 标签 |
| 标注代码行匹配失败 | 检查缩进、CRLF、是否对应 func-before，或是否实际是整数行号 |
| 超过 max_length | 检查包含答案的总 token 数，提高长度或明确 drop；不截断答案 |
| save/eval 参数错误 | save_steps 应为 eval_steps 整数倍 |
| 输出目录非空 | 新实验换目录；续训使用原目录和 checkpoint |
| 没有 eval_loss 曲线 | 当前验证仅计算任务指标；在 TensorBoard 查看 eval_line_f1/eval_cwe_f1 等 |
| 评估结果大量无效/被截断 | 查看 predictions 及 eval_truncated_samples，必要时增大 eval_max_new_tokens 与 max_length |
| train/validation 重叠 | 按项目/函数重新划分或去重，脚本不会自动删除数据 |

## 12. 参考接口

- Qwen3 官方模型与非 thinking 模式：https://huggingface.co/Qwen/Qwen3-0.6B
- TRL SFTTrainer / assistant_only_loss（本项目版本）：https://huggingface.co/docs/trl/v0.23.0/en/sft_trainer
- PEFT 量化训练与 prepare_model_for_kbit_training：https://huggingface.co/docs/peft/v0.17.0/en/developer_guides/quantization
- Hugging Face 模型下载：https://huggingface.co/docs/huggingface_hub/guides/download
- uv 安装与环境：https://docs.astral.sh/uv/

核对日期：2026-10-22。真实训练时间、显存和任务精度需在你的 GPU 与正式数据上测量。
