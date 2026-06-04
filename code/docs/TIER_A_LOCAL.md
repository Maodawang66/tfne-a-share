# 档位 A 本机基线

> **独立项目文档（完整流程 + 实验报告）**：[`tier_a_project/README.md`](../../tier_a_project/README.md)、[`tier_a_project/实验报告.md`](../../tier_a_project/实验报告.md)

| 路径 | 绝对地址 |
|------|----------|
| 项目根 | `C:\Users\Lenovo\Desktop\deep learning\pro` |
| 代码 | `C:\Users\Lenovo\Desktop\deep learning\pro\code` |
| 数据 | `C:\Users\Lenovo\Desktop\deep learning\pro\data` |
| 配置 | `C:\Users\Lenovo\Desktop\deep learning\pro\code\config.tier_a_local.yaml` |
| 一键脚本 | `C:\Users\Lenovo\Desktop\deep learning\pro\code\scripts\run_tier_a_local.py` |

硬件：NVIDIA GeForce RTX 3060 Laptop GPU 6GB，CPU 4 核，conda 环境 `ai25`。

---

## 一键运行

```powershell
conda activate ai25
cd "C:\Users\Lenovo\Desktop\deep learning\pro\code"
python "C:\Users\Lenovo\Desktop\deep learning\pro\code\scripts\validate_tier_a_local.py"
python "C:\Users\Lenovo\Desktop\deep learning\pro\code\scripts\run_tier_a_local.py"
```

### 分阶段

```powershell
python "C:\Users\Lenovo\Desktop\deep learning\pro\code\scripts\run_tier_a_local.py" --prep-only
python "C:\Users\Lenovo\Desktop\deep learning\pro\code\scripts\run_tier_a_local.py" --train-eval-only
```

---

## 产物绝对路径

```text
C:\Users\Lenovo\Desktop\deep learning\pro\code\artifacts\checkpoints\tier_a\gru\seed_42\best.pt
C:\Users\Lenovo\Desktop\deep learning\pro\code\artifacts\checkpoints\tier_a\mlp\seed_42\best.pt
C:\Users\Lenovo\Desktop\deep learning\pro\code\artifacts\checkpoints\tier_a\news_hash\seed_42\best.pt
C:\Users\Lenovo\Desktop\deep learning\pro\code\artifacts\predictions\tier_a\ensemble_scores.parquet
C:\Users\Lenovo\Desktop\deep learning\pro\code\artifacts\orders\tier_a\orders_20241231.csv
```

---

## 备份 C smoke panel（跑 A 前）

```powershell
$bak = "C:\Users\Lenovo\Desktop\deep learning\pro\code\artifacts\_backup_c_smoke_prep"
New-Item -ItemType Directory -Force -Path "$bak\features", "$bak\news" | Out-Null
Copy-Item "C:\Users\Lenovo\Desktop\deep learning\pro\code\artifacts\panel_daily.parquet" $bak\
Copy-Item "C:\Users\Lenovo\Desktop\deep learning\pro\code\artifacts\features\panel_features.parquet" "$bak\features\"
Copy-Item "C:\Users\Lenovo\Desktop\deep learning\pro\code\artifacts\news\panel_news.parquet" "$bak\news\" -ErrorAction SilentlyContinue
```

---

## 服务器提交

见 `C:\Users\Lenovo\Desktop\deep learning\pro\code\docs\USTC_SUBMIT_SIMPLE.md`。
