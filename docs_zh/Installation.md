# 安装

socialverse 是一个纯 Python 包。其数值核心仅依赖 `numpy`、`scipy` 和 `pandas` — **无需 R、Stata 或 SPSS 运行时**即可使用任何移植的方法。

## 从 PyPI 安装

```bash
pip install socialverse
```

```python
import socialverse as sv
print(sv.__version__)     # 0.7.2
```

## 依赖项

| 依赖项 | 用途 |
|---|---|
| Python ≥ 3.9 | 运行时 |
| `numpy`、`scipy` | 每个 R 包端口的数值核心 |
| `pandas` | 表格研究数据 / `StudyState` 框架 |
| `matplotlib` | 绘图 (`sv.pl.*`) |

可选的加速器在存在时会自动检测（例如用于心理测量学的 `factor_analyzer`），纯 `numpy`/`scipy` 实现始终作为备用方案，因此无论是否安装可选包，结果都相同。

## 验证安装

```python
import socialverse as sv

# 列出所有已注册的分析，按类别分组
for category, fns in sv.list_functions().items():
    print(category, "→", len(fns), "functions")

# 奇偶门控的 R 包端口位于 socialverse.external 下
from socialverse.external import pymetafor, pysurvey, pysurvival   # etc.
```

## 对于贡献者 — 重现奇偶门控

每个端口都包含 R 参考驱动程序和奇偶性测试。重现这些门控需要 **R**（仅用于重新生成参考值 — 从不在运行时）：

```bash
# R 4.5.x 且已安装参考包，例如
Rscript -e 'install.packages(c("metafor","survey","survival","lavaan"))'

# 重新生成参考值并运行一个端口的奇偶性测试
Rscript socialverse/external/pymetafor/tests/r_reference_driver.R
pytest  socialverse/external/pymetafor/tests/ -q
```

提交的 `reference.json` 文件使任何人都可以**不使用 R** 运行奇偶性测试；R 仅在从头重新生成它们时才需要。请参阅任何 [教程](tutorials/external/index.md) 以获取特定包的奇偶性证据。

## 本地构建这些文档

```bash
pip install mkdocs-material mkdocs-glightbox
mkdocs serve            # http://127.0.0.1:8000
```
