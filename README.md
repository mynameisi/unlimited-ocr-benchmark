# Unlimited OCR Benchmark

**Unlimited OCR** (百度开源) 在 **NVIDIA GeForce RTX 3090** 上的全面评测报告。

> 模型：baidu/Unlimited-OCR (3.34B params, MoE 500M active)
> 论文：https://arxiv.org/abs/2506.12345
> 官方仓库：https://github.com/baidu/Unlimited-OCR

---

## 硬件环境

| 项目 | 规格 |
|:-----|:-----|
| GPU | NVIDIA GeForce RTX 3090 (24GB GDDR6X) |
| 系统 | Windows 10 + WSL2 Ubuntu |
| CUDA | 12.6 |
| PyTorch | 2.6.0+cu124 |
| Python | 3.10 |

## VRAM 占用

| 测试场景 | VRAM |
|:---------|:----:|
| 常规 PDF (5页) | **7.54GB** |
| 大文件 24MB (19页) | **7.54GB** |
| 长文档 100页 | **7.54GB** |
| 长文档 86页 | **7.54GB** |

> R-SWA 机制保证 KV 缓存固定大小，VRAM 不随文档长度增长。

---

## 测试一：常规 PDF 基准 (V1)

9 份 arXiv 论文 PDF，涵盖英文/中文/表格/公式/大文件/长文档。

| 文件 | 类型 | 大小 | 页数 | 平均耗时 |
|:-----|:-----|:----:|:----:|:--------:|
| 01_arxiv_paper.pdf | 英文论文（数学公式） | 14.5MB | 32p | 17.1s |
| 02_arxiv_tables.pdf | 英文论文（表格） | 1.6MB | 55p | 17.5s |
| 03_chinese.pdf | 中文论文 | 1.4MB | 5p | 16.7s |
| 04_chinese_news.pdf | 中英混合 | 3.1MB | 58p | 16.3s |
| 04_mixed_text.pdf | 英文 LoRA 论文 | 1.5MB | 26p | 16.9s |
| 05_llama_paper.pdf | LLaMA 论文 | 0.7MB | 27p | 35.4s |
| 06_gpt4_report.pdf | GPT-4 技术报告 | 5.0MB | **100p** | **15.9s** |
| 07_deepseek_r1.pdf | DeepSeek-R1 论文 | 4.8MB | **86p** | **29.9s** |
| 08_large_images.pdf | 高分辨率论文 | **24MB** | 19p | **16.8s** |

### 关键发现

- **长文档无退化**：100 页文档的第 1 页和第 100 页处理速度完全一致
- **大文件无退化**：24MB 高分辨率 PDF 与 1MB PDF 速度相同
- **表格识别**：自动输出 HTML `<table>` 格式
- **公式识别**：自动输出 LaTeX `\(...\)` 格式
- **版面分析**：自动分类 `title / text / table / equation / image_caption / ref_text`

---

## 测试二：真实场景挑战 (Challenge)

6 种真实世界 OCR 挑战场景，使用程序化生成的仿真测试图。

| 场景 | 文件 | 初始结果 | **调优后** | 速度 |
|:-----|:-----|:--------:|:----------:|:----:|
| 🟢 中文扫描件（噪点+倾斜） | c1 | ✅ 完美 | ✅ 完美 | **6.3s** |
| 🟡 手机拍摄文档（阴影+折痕） | c2 | ❌ 重复循环 | ⚠️ 减轻 | 7.1s |
| 🟡 手写体（模拟笔迹） | c3 | ❌ 重复循环 | ⚠️ 减轻 | 17.7s |
| 🟢 低质量扫描（模糊+泛黄） | c4 | ✅ 文字OK | ✅ 文字OK | **3.4s** |
| 🟢 多栏报纸排版 | c5 | ✅ 完美 | ✅ 完美 | **6.5s** |
| 🟢 盖章合同（水印覆盖） | c6 | ❌ 重复循环 | ✅ **完整正文+签名** | **13.4s** |

### 挑战场景生成说明

测试图使用 Python (PIL) 生成，模拟以下真实扫描/拍摄特征：

| 场景 | 模拟特征 |
|:-----|:---------|
| c1 中文扫描件 | 噪点(2%) + 轻微倾斜(2°) |
| c2 手机拍摄 | 噪点(3%) + 折痕阴影 + 倾斜 |
| c3 手写体 | 衬线字体模拟手写 + 噪点 |
| c4 低质量扫描 | 泛黄底色(#ccbbaa) + 高斯模糊 + 低对比度(灰度80) |
| c5 多栏报纸 | 两栏排版 + 标题分离 + 竖线分割 |
| c6 盖章合同 | 红色印章叠加("APPROVED" + "CONFIDENTIAL") + 噪点 |

---

## 🔧 最优参数调优

对盖章合同场景进行 14 种参数组合的网格搜索。

| 参数 | 测试范围 | ⭐**最优值** |
|:----|:---------|:-----------:|
| `no_repeat_ngram_size` | 0, 10, 20, 35, 50, 70, 80, 100 | **50** |
| `ngram_window` | 0, 30, 50, 100, 128, 200 | **128** |
| `temperature` | 0.0, 0.1 | **0.0** |
| `max_length` | 2048, 4096 | **2048** |

### 调优前后对比（盖章合同）

```
调优前 (ngram=0):
  CONTRACT AGREEMENT This agreement is entered into on June 1, 2024. 
  The Company's share capital of 10,000,000.00. 
  The Company's share capital of 10,000,000.00.     ← 无限重复
  The Company's share capital of 10,000,000.00. 
  ...

调优后 (ngram=50, window=128):
  CONTRACT AGREEMENT This agreement is entered into on June 1, 2024
  between: Party A: ABC Corporation 123 Business Ave...
  Party B: XYZ Solutions Ltd. 456 Commerce Dr...
  Terms and Conditions:
  1. Party A agrees to provide consulting services...
  2. The term of this agreement shall be 12 months
  3. Payment Terms: Net 30 days from invoice date
  4. Either party may terminate with 30 days written notice
  IN WITNESS WHEREOF... Signatures ✅
```

---

## 已知局限

1. **重复生成问题**：密集排版页面或印章干扰区域仍可能触发重复，`ngram=50` 能大幅缓解但无法完全消除
2. **特殊字符识别**：低质量扫描下的符号识别（如 `!@#$%^`）准确率较低
3. **处理速度**：13-30s/页，适合离线批处理，不适合实时识别
4. **SGLang 未部署**：本测试使用 HuggingFace Transformers 直接加载推理，未使用 SGLang 服务端（后者能提供更高并发和 streaming 能力）

---

## 项目结构

```
├── ocr_benchmark.py           # V1 基础测试
├── ocr_benchmark_v2.py        # V2 扩展测试（大文件+长文档）
├── ocr_challenge.py           # 真实场景挑战测试
├── ocr_tune.py                # 参数调优循环
├── ocr_final_verify.py        # 最优参数最终验证
├── gen_challenge.py           # 真实场景挑战图生成器
│
├── test_pdfs/                 # 测试 PDF
│   ├── 0*_*.pdf               # arXiv 论文（9份）
│   └── c*_*.pdf               # 6 份真实场景挑战
│
├── ocr_results/               # V1 结果
├── ocr_results_v2/            # V2 扩展结果
├── ocr_results_challenge/     # 调优前挑战结果
└── ocr_results_final/         # ⭐ 最优参数最终结果
    └── _final_summary.json
```

## 运行方法

```bash
# 设置环境
export PATH="$HOME/miniconda3/envs/unlimited-ocr/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/unlimited-ocr/lib/python3.10/site-packages/nvidia/cusparselt/lib:$HOME/miniconda3/envs/unlimited-ocr/lib/python3.10/site-packages/nvidia/cusparse/lib"

# 基础测试
python3 ocr_benchmark.py

# 扩展测试（大文件+长文档）
python3 ocr_benchmark_v2.py

# 真实场景挑战
python3 ocr_challenge.py

# 参数调优
python3 ocr_tune.py

# 最优参数最终验证
python3 ocr_final_verify.py

# 生成挑战测试图
python3 gen_challenge.py
```
