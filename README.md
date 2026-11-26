# KnowLine-4B (Gen5 / Gen5.5) on Ascend 910B4 — 全栈推理优化封装

本仓库封装了 **KnowLine-4B（基于 Qwen3.5-4B：24 层 GDN 线性注意力 + 8 层 Full-Attention 混合架构）** 在华为昇腾 **Ascend 910B4** 上的全栈推理优化方案、一键环境构建脚本、全量源码补丁、GDN Core C++ 胶水扩展以及一键启停与验收脚本。

在相同的昇腾官方基础镜像环境中，拉取本仓库后执行 `bash setup_env.sh` 即可自动完成 `vLLM 0.30.0` + `vLLM-Ascend 0.30.0` 依赖安装、全量优化补丁应用与 C++ 扩展编译，开箱即享 **3.5×–4.0×** 端到端低延迟加速（双问 `/v1/systemone` 从原版 **`~240–280 ms`** 降至 **`~66 ms`**）。

---

## 一、服务器环境与基础镜像来源声明

### 1. 官方基础镜像来源
- **镜像来源（AscendHub 官方镜像）**：  
  [https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884?version=9.2.1-910b-ubuntu22.04-py3.12](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884?version=9.2.1-910b-ubuntu22.04-py3.12)
- **镜像标签版本**：`9.2.1-910b-ubuntu22.04-py3.12`

### 2. 实测服务器硬件与软件基座
| 组件 | 实测配置 |
| :--- | :--- |
| **NPU 加速卡** | 单卡 **Ascend 910B4**（32 GB HBM） |
| **CPU / 架构** | **Kunpeng-920**（ARM64 / `aarch64`），容器配额 24 vCPU |
| **操作系统** | `Ubuntu 22.04.5 LTS`（容器内） |
| **CANN / ATB / 驱动** | CANN **`9.2.1`** + NNAL ATB (`cxx_abi_1`)，NPU Driver **`25.5.1`** |
| **Python / 编译器** | Python **`3.12.16`**，GCC/G++ `11.4.0`，GNU `patch 2.7.6` |
| **推理核心依赖** | `torch==2.10.0+cpu`、`torch-npu==2.10.0.post7`、`vllm==0.30.0`、`vllm-ascend==0.30.0`、`triton-ascend==3.6.0.post2`、`flash-linear-attention-npu-a2==26.9.1`、`transformers==5.14.1` |

> **说明**：`9.2.1-910b-ubuntu22.04-py3.12` 基础镜像本身预装了 Ubuntu 22.04 + CANN 9.2.1 + ATB + Python 3.12，不包含 `vllm` 与 `vllm-ascend`。本仓库通过 **`setup_env.sh` + `patches/`** 方式交付（而非直接提交数 GB 的二进制 `venv`），使用者**无需手动寻找或下载 `vllm` / `vllm-ascend` / `triton-ascend` / `fla-npu`**，运行一条 `bash setup_env.sh` 即可全自动完成拉取、依赖冲突修复、补丁应用与 C++ 扩展编译。

---

## 二、优化前后性能对比（Ascend 910B4 实测）

| 测试场景 | 原版开箱 (`0.30.0`) | 纯引擎无合批 (`STRICT_B1=1`) | **本仓库默认全栈模式** | 相对原版提速 |
| :--- | :---: | :---: | :---: | :---: |
| **短序列双问 (`test_knowline.py`)** | `~240–280 ms` | `131.60 ms` (`diff < 1e-6`) | **`66.44 ms`** (med **`66.08`**, min **`65.58`**, p95 **`68.49`**) | **`~3.6×–4.2×`** |
| **中等序列双问 (`MED_x6`, 644 tok)** | `~347 ms` | `150.20 ms` | **`96.81 ms`** (med **`93.36`**) | **`~3.6×`** |
| **长前缀命中 (`LONG_x55`)** | `~294 ms` | `169.79 ms` | **`152.73 ms`** | **`~1.9×`** |
| **4 并发混合 (`CONCURRENT_8`)** | `~376 ms` | — | **`196.75 ms`** | **`~1.9×`** |
| **NPU `Computing` 占比 / `Task Wait`** | `48.0%` / `Free 52%` | — | **`70.8%` / `0.00 ms`** | 零算子等待空泡 |
| **额外 `lm_head` HBM / ACLGraph 捕获** | 每步现转 2.54 GB / `20s` | **`0 GiB`** / `16s` | **`0 GiB`** / **`11s`** | 零额外显存占用 |

---

## 三、四步快速部署（同款昇腾镜像内直接跑）

### 第 1 步：拉取本仓库
```bash
git clone https://github.com/perturabo-iron/knowline-4B-ascend910B.git
cd knowline-4B-ascend910B
```

### 第 2 步：一键安装环境、打补丁并编译 C++ 胶水
无需提前安装 `vllm` 或 `vllm-ascend`，直接运行：
```bash
bash setup_env.sh
```
脚本会自动完成：
1. 创建独立的 `./venv`（Python 3.12）；
2. 自动从华为云 PyPI 镜像、PyTorch CPU 源与华为昇腾 PyPI 源拉取精确版本依赖（自动修复官方 `vllm-ascend` 的 `fastapi` 版本冲突和 `triton-ascend` 的 `numpy`/`triton` 模块覆盖冲突）；
3. 自动应用 `patches/vllm_0.30.0_full.patch` 与 `patches/vllm_ascend_0.30.0_full.patch`；
4. 编译 `csrc/gdn_core_glue.cpp` 生成 `_gdn_core_C.so` 并完成导入自检（出现 `SETUP_ALL_OK` 即成功）。

> 如果你所在容器**已经装好了干净的 `vllm 0.30.0` + `vllm-ascend 0.30.0`**，也可以直接把优化补丁和 C++ 胶水打到现有环境：
> ```bash
> bash patches/apply_release_patches.sh /path/to/venv/lib/python3.12/site-packages
> ```

### 第 3 步：准备 KnowLine-4B 模型权重
下载 KnowLine-4B（Gen5 / Gen5.5）模型目录（需包含 `serve_knowline.sh`、`knowline_server.py` 与权重文件）：
```bash
# 示例（国内 GitCode 镜像；或从 HuggingFace PelaAI/KnowLine-4B-Gen5.5 下载）
git clone https://gitcode.com/hf_mirrors/PelaAI/KnowLine-4B-Gen5.5.git ~/work/knowline
```

### 第 4 步：一键启动服务并验收
```bash
# 后台常驻启动（自动激活 venv+CANN、默认 QUANT=bf16、自动等待 /health 就绪）
BG=1 bash start_knowline.sh ~/work/knowline 0 8080

# 端到端功能 + 延迟验收（验证 20 次双问请求）
./venv/bin/python test_knowline.py --runs 20

# 停止服务并检查 NPU 显存释放
bash stop_knowline.sh
```

> 仅需激活环境跑自定义命令时执行：`source activate.sh`

---

## 四、仓库目录结构

```text
knowline-4B-ascend910B/
├── README.md                    # 本文档
├── LICENSE
├── setup_env.sh                 # 一键从零构建环境 + 自动修依赖冲突 + 打补丁 + 编译 C++ 扩展
├── activate.sh                  # 一键激活 venv + CANN 9.2.1 + ATB + 推荐运行时环境变量
├── start_knowline.sh            # 一键启动 KnowLine-4B 服务（支持前台 / BG=1 后台 / STRICT_B1=1 审计）
├── stop_knowline.sh             # 一键停止服务并检查 NPU 状态
├── test_knowline.py             # 端到端功能与延迟基准测试脚本
├── constraints.txt              # 核心依赖版本锁定
├── requirements-installed.txt   # 完整已验证 Python 依赖版本快照
├── knowline-vllm/               # 优化后的 KnowLine vLLM 适配层（批量 tokenize + 原子合批 + 跳无效 warm）
│   ├── knowline_vllm.py
│   └── serve_knowline_vllm.sh
├── csrc/                        # GDN Core C++ 胶水源码与快速编译脚本（~5s 编译）
│   ├── gdn_core_glue.cpp
│   └── build_gdn_core_glue.sh
└── patches/                     # 相对官方 0.30.0 的完整全量标准 diff 补丁
    ├── vllm_0.30.0_full.patch         # vllm 侧 8 个文件（HOST-1/1+, HOST-2, HOST-3）
    ├── vllm_ascend_0.30.0_full.patch  # vllm_ascend 侧 6 个文件（GDN-1/4/5+, HOST-4/5/6）
    ├── knowline_vllm_full.patch       # knowline-vllm 侧 2 个文件
    └── apply_release_patches.sh       # 面向已有 0.30.0 环境的一键打补丁+编译脚本
```

---

## 五、已内置的核心优化清单与环境变量开关

所有优化均默认生效，且支持通过环境变量免改代码单独开关：

| 优化项 | 核心原理与收益 | 环境变量（默认） |
| :--- | :--- | :--- |
| **HOST-1+** | 放行昇腾原生 `torch.mm(..., out_dtype=fp32)`（`MatMulV3` BF16 输入直接累加 FP32 输出），消除每步 1.27GB→2.54GB 在线权重转换，单次 `lm_head` 从 `9.86 ms → 1.87 ms` 且免占 2.54 GB 显存 | `VLLM_ASCEND_FP32_HEAD_MM=1` |
| **HOST-2** | 修复 `Sampler` 同时传 `logprobs` 与 `logprob_token_ids` 时重复计算 248K 词表 `TopKV2` 后覆盖丢弃的 Bug，并增加 Greedy `rank=1` 捷径（`TopKV2` 清零） | 始终开启（零误差） |
| **HOST-3** | 新增 EngineCore `ADD_BATCH` ZMQ 原子入队与 `/inference/v1/generate_batch`；短前缀（`<640` 混合块对齐粒度）自动跳过无效 `warm()`；扩展 Piecewise ACLGraph 捕获档位至 `1024`（24 档），双问请求 3 步并 1 步 | `KNOWLINE_BATCH_SCORE=1`<br>`KNOWLINE_WARM_MIN_TOKENS=640`<br>`KNOWLINE_DEFAULT_CG_EXT=1` |
| **HOST-4** | 修复 `vllm_ascend/cpu_binding.py` 物理/逻辑 NPU ID 映射 Bug 及本地 NUMA 扩散问题；注入最优 CANN 缓存参数 | `TASK_QUEUE_ENABLE=1`<br>`ACLNN_CACHE_LIMIT=10000`<br>`HOST_CACHE_CAPACITY=50` |
| **HOST-5a/b/c** | 8 层 Full-Attn 共享一次 M-RoPE `cos_sin_cache` 查表；3 个 GDN KV 组共享元数据并改 CPU 索引计算；`_prepare_inputs` 瘦身 | `VLLM_ASCEND_SHARE_MROPE_COS_SIN=1` |
| **GDN-1/2/3/5 + HOST-6** | 消除 910B4 每层 6 次无效 `BNSD↔BSND` 转置；以独立 C++ 扩展 `_gdn_core_C.so` 收拢 GDN Prefill Python/Triton 启动墙（`2491 → 590 us/层`），无初态直传 `None`，Q/K 合并精确 FP32 L2Norm，手写精确 `log(1+exp)` 规避 `aclnnSoftplus` 负值精度坑 | `VLLM_ASCEND_GDN_CPP_GLUE=1` |
| **GDN-4** | `max_tokens=1` 单步结束且不跨缓存块边界时，C++ 胶水跳过 24×B 次 `ssm_state` 回写（内核保持 `output_final_state=true` 规避底层数值漂移，额外误差 `0.000e+00`） | `VLLM_ASCEND_GDN_SKIP_UNUSED_FINAL_STATE=1` |
| **GDN-5+** | 将 `DeviceOperator.fused_gdn_gating` 注册为自定义算子移入 Piecewise ACLGraph，消除每步 168 次 C++ 门控派发 | `VLLM_ASCEND_GDN_GATING_IN_GRAPH=1` |

### 精度说明与严格 `B=1` 离线审计模式
- **严格 `B=1` 单条模式**：所有引擎优化叠加相对官方原版 `0.30.0` 最大打分误差仅 **`8.56e-7`**（远优于 `1e-4` 红线），多步生成完全一致。
- **默认 `B>=2` 原子合批模式**：多问合批计算因 BF16 矩阵乘 M 维分块累加顺序不同，相对 `B=1` 串行存在 `~1e-3 ~ 3.5e-3` 硬件正常浮点波动（与原版并发合批行为完全一致）。
- 若需严格 `B=1` 审计精度，启动时加 `STRICT_B1=1` 即可秒切：
  ```bash
  STRICT_B1=1 BG=1 bash start_knowline.sh ~/work/knowline 0 8080
  ```

---

## 六、关键避坑红线

1. **严禁设置 `TASK_QUEUE_ENABLE=2`**：当前 `torch_npu 2.10.0.post7` 下与 ACLGraph 捕获冲突，启动会直接报 `ERR00007` 退出。
2. **严禁设置 `CPU_AFFINITY_CONF=1`**：会与 `vllm_ascend` 绑核争抢线程亲和性，导致均值与长尾延迟倒退。
3. **严禁开启 `VLLM_USE_V2_MODEL_RUNNER=1`**：V2 Runner 不支持自定义 `logprob_token_ids`，会破坏 KnowLine 评分。
4. **严禁将 `VLLM_ASCEND_GDN_SKIP_UNUSED_FINAL_STATE` 改为 `2`**：910B4 `fla_npu_stable` 在 `output_final_state=False` 时存在 `~1.38e-3` 数值漂移，必须保持默认 `1`。
5. **910B4 默认使用 `QUANT=bf16`**：910B4 硬件不支持原生 FP8，`serve_knowline_vllm.sh` 检测到昇腾环境会自动默认 `bf16`。
6. **切勿直接 `mv` 移动 `venv/` 目录**：如需迁移路径，请在新路径重新运行 `bash setup_env.sh`。
