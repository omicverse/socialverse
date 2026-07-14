# pyrobumeta — Python中的robumeta

> 用于依赖性效应量的稳健方差估计（RVE）元回归——CORR/HIER工作模型、Tipton (2015) CR2小样本修正，以及`impute_covariance_matrix`——可从Python调用，与R无误差（1e-6），无需R运行时。

## `robumeta`的功能

`robumeta`（Hedges、Tipton和Johnson）在效应量在研究内部统计相关时拟合元回归——例如同一研究报告的多个结果、时间点或亚组——无需研究人员指定精确的研究内协方差结构。它提供两种工作模型，相关效应（"CORR"，用于在研究内共享通用测量/结果的效应量）和分层效应（"HIER"，用于在多个层级嵌套的效应量），通过矩量法估计方差分量，并——当`small=TRUE`时（默认值）——应用Tipton (2015) CR2偏差缩减三明治估计器和Satterthwaite自由度，使得p值和置信区间在研究较少时仍保持良好校准。社会科学家在教育、心理学和公共卫生元分析中经常使用它，其中"每个效应量一行，每个研究多行"是常见的情况，并且按研究聚类不能忽略。`clubSandwich::impute_covariance_matrix`和`coef_test(vcov="CR2")`是配套工具，用于构建基于相关性的工作V矩阵，并在任何拟合模型上重新运行CR2检验。

## Python端口

- `robu(effect_size, var_eff_size, studynum, covariates, modelweights="CORR", rho=0.8, small=True)` — 拟合RVE元回归：第一阶段加权的加权最小二乘法、矩量法τ²（CORR）或τ²+ω²（HIER）、用方差分量调整权重重新拟合，然后（如果`small=True`）应用CR2三明治协方差和Satterthwaite df。返回一个包含`b`、`SE`、`t`、`dfs`、`prob`、`CI_L`、`CI_U`、`tau_sq`（HIER时加`omega_sq`，CORR时加`I2`）、`N`、`M`、`p`的字典。
- `impute_covariance_matrix(vi, cluster, r, return_list=True)` — 在常数簇内相关性`r`下构建块对角工作协方差矩阵（`clubSandwich::impute_covariance_matrix`产生的）（对角线 = `vi`，非对角线 = `r * sqrt(vi_i * vi_j)`）；返回每个簇块的列表或完整的`(M, M)`矩阵。
- `coef_test(fit, vcov="CR2")` — 将`robu()`拟合的已计算CR2 `SE`/`dfs`打包到`clubSandwich::coef_test()`返回的每系数检验表中（`beta`、`SE`、`tstat`、`df`、`p_val`）。

该端口是纯numpy/scipy实现——无R运行时、无rpy2、无子进程调用。它主要通过`sv.tl.robu`（`socialverse/tl/_meta_rve.py`）连接到socialverse，后者调用`pyrobumeta.robu`作为其忠实后端（如果端口引发异常则回退到内部三明治估计器），其次调用`pyrobumeta.coef_test`获取CR2 Satterthwaite检验表，以及当调用者提供`impute_r`/`within_corr`时调用`pyrobumeta.impute_covariance_matrix`。在`sv.tl.ma_robust` / `sv.tl.ma_che`中存在一个独立实现的三明治估计器，用于在robumeta工作模型框架之外的CR0/CR1/CR2推断——它不调用本端口。

:::{admonition} 等价性门控
:class: note

该端口固定于R `robumeta` 2.1 / `clubSandwich` 0.7.0，在4个确定性等价性测试中`max_abs_err < 1e-6`。
:::

## 快速开始

```python
import numpy as np
from socialverse.external.pyrobumeta import robu, coef_test, impute_covariance_matrix

# --- a tiny dependent-effect-size dataset: 6 rows nested in 3 studies -------
effectsize = np.array([0.20, 0.35, 0.10, 0.50, 0.42, 0.28])
var_eff    = np.array([0.04, 0.05, 0.03, 0.06, 0.05, 0.04])
studyid    = np.array([1, 1, 2, 2, 3, 3])
males      = np.array([0.5, 0.5, 0.4, 0.4, 0.6, 0.6])   # study-level moderator
college    = np.array([0.3, 0.3, 0.6, 0.6, 0.2, 0.2])

# --- CORR working model, rho=0.8, CR2 small-sample correction (default) ----
fit = robu(
    effect_size=effectsize, var_eff_size=var_eff, studynum=studyid,
    covariates=[males, college], modelweights="CORR", rho=0.8, small=True,
)
print("coefficients (intercept, males, college):", fit["b"])
print("robust SE:                               ", fit["SE"])
print("Satterthwaite df:                         ", fit["dfs"])
print("tau^2, I^2:                               ", fit["tau_sq"], fit["I2"])

# --- Tipton (2015) CR2 coefficient test table -------------------------------
ct = coef_test(fit, vcov="CR2")
print("t-stats:", ct["tstat"], " p-values:", ct["p_val"])

# --- impute a block-diagonal working V under an assumed within-study r=0.7 -
blocks = impute_covariance_matrix(var_eff, studyid, r=0.7, return_list=True)
print("per-study V blocks:", [b.shape for b in blocks])
```

通过有线socialverse管道函数（`sv.tl.robu`）进行相同拟合，该函数从持有`meta_effects`模型的`StudyState`驱动相同的`pyrobumeta.robu`后端：

```python
import socialverse as sv

state = sv.pp.meta_effects(state, ...)          # produces yi/vi/study columns
state = sv.tl.robu(state, model="CORR", rho=0.8)  # dispatches to pyrobumeta.robu
print(state.models["meta_rve"]["coefs"])
print(state.models["meta_rve"]["coef_test_cr2"])  # CR2 Satterthwaite test table
```

## R ↔ Python词典

| R (`robumeta` / `clubSandwich`) | socialverse | 注释 |
|---|---|---|
| `robu(effectsize ~ x1 + x2, data=d, studynum=, var.eff.size=, modelweights="CORR", rho=0.8, small=TRUE)` | `socialverse.external.pyrobumeta.robu(effect_size, var_eff_size, studynum, covariates, modelweights="CORR", rho=0.8, small=True)` / `sv.tl.robu(state, model="CORR", rho=0.8)` | 协变量以数组列表形式传递（无截距）；截距在内部前置，匹配R公式的隐式截距。 |
| `robu(..., modelweights="HIER")` | `robu(..., modelweights="HIER")` / `sv.tl.robu(state, model="HIER")` | HIER还返回`omega_sq`（无`I2`）。 |
| `mc$reg_table$b.r`, `$SE`, `$dfs`, `$prob`, `$CI.L`/`$CI.U` | `fit["b"]`, `fit["SE"]`, `fit["dfs"]`, `fit["prob"]`, `fit["CI_L"]`/`fit["CI_U"]` | 每个系数一项，截距优先。 |
| `mc$mod_info$tau.sq`, `$I.2`, `$omega.sq` | `fit["tau_sq"]`, `fit["I2"]`, `fit["omega_sq"]` | `I2`仅用于CORR，`omega_sq`仅用于HIER。 |
| `clubSandwich::coef_test(mc, vcov="CR2")` | `socialverse.external.pyrobumeta.coef_test(fit, vcov="CR2")` | 重用`robu(..., small=True)`已计算的CR2 `SE`/`df`。 |
| `clubSandwich::impute_covariance_matrix(vi, cluster, r, return_list=TRUE)` | `socialverse.external.pyrobumeta.impute_covariance_matrix(vi, cluster, r, return_list=True)` / `sv.tl.robu(state, impute_r=0.7)` | 仅标量`r`（常数簇内相关路径）。 |

## 等价性证据

`socialverse/external/pyrobumeta/tests/test_parity.py`中的4个确定性等价性测试，在`socialverse/external/pyrobumeta/tests/reference.json`处门控在`max_abs_err < 1e-6`（由`r_reference_driver.R`针对R的`robumeta::corrdat`和`robumeta::hierdat`夹具生成）：

- `test_corr` — 在`corrdat`上的CORR工作模型（N=39项研究，M=172行，p=3个协变量）：系数`b`、稳健`SE`、`t`、Satterthwaite `dfs`、`prob`、`CI_L`/`CI_U`、`tau_sq`、`I2`。
- `test_hier` — 在`hierdat`上的HIER工作模型（5个协变量）：相同的量加上`tau_sq`和`omega_sq`（HIER无`I2`）。
- `test_impute_covariance_matrix` — 在`corrdat`上的`impute_covariance_matrix`块对角V（r=0.7）：每个簇块的条目和块大小向量。
- `test_coef_test_cr2` — 在CORR拟合上的`coef_test(fit, vcov="CR2")`：`beta`、`SE`、Satterthwaite `df`、`tstat`、`p_val`。

所有四个量都是完全确定的（本端口中无自举或MCMC步骤），因此1e-6门是严格的数值相等性检查，而非随机容差检查。

:::{admonition} 范围限制
:class: warning

仅`small=TRUE`（默认值，也是RVE的统计相关路径）为CORR/HIER工作模型移植；`small=FALSE`（HC0风格、未修正）分支和用户提供权重的工作模型未进行等价性测试。`impute_covariance_matrix`仅支持标量（常数）簇内`r`，不支持clubSandwich的每簇向量或`ar1`相关结构。
:::

重现：

```bash
Rscript socialverse/external/pyrobumeta/tests/r_reference_driver.R
pytest socialverse/external/pyrobumeta/tests/
```

## 在socialverse工作流中

日常工作中，在`sv.pp.meta_effects`之后调用`sv.tl.robu(state, model="CORR"|"HIER", rho=0.8)`——它直接分派到本端口，并额外将Tipton CR2系数检验表（`coef_test_cr2`）写入`state.models["meta_rve"]`。注册表强制执行契约（`requires={"models": ["meta_effects"]}`、`produces={"models": ["meta_rve"]}`）；如果代码有所移动，请随时通过`sv.list_functions()`或`registry_lookup("robu")`而非相信本页来确认实时签名和别名。
