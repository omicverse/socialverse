# socialverse benchmarks

每个 benchmark 同时是**演示案例**和**回归测试**:给一句用户会对 omicos agent 说的自然语言 prompt,跑对应的 socialverse 契约链,并**断言结果复现了发表值或已知真值**。既能拿来做案例演示(prompt + 预期数字),也能当 socialverse 的端到端正确性守卫。

```bash
python benchmarks/run_benchmarks.py            # 全部 11 个(会下载 4 份公开数据到 data/)
python benchmarks/run_benchmarks.py --offline  # 只跑 7 个内置玩具数据案例(秒级、免下载)
python benchmarks/run_benchmarks.py --only qca  # 单个案例(按 id 子串匹配)
```

需要 `pip install socialverse`(或把仓库根加进 `PYTHONPATH`)。

---

## A. 旗舰复现(真实公开数据 → 发表级数值)

| id | 能力 | agent · skill | prompt | 预期(已核对) |
|---|---|---|---|---|
| `did_fect_hh2015` | 交错采纳 DiD + 反事实插补 | `social_science_econometrician` · causal-identification + modern-did | 「用这份瑞士市镇面板,估计切换到间接归化程序对归化率的效应:声明面板设计、检验平行趋势、跑 TWFE 双重差分,再用反事实插补做异质稳健对照 + placebo。」 | TWFE ATT **+1.339**、FEct ATT ~**+1.50**、HC1 SE **0.161**、placebo 不显著 |
| `dml_cate_401k` | 双重机器学习 + 因果森林 | `causal_data_scientist` · causal-dag + causal-machine-learning | 「估计 401(k) 资格对净金融资产的效应:用双重机器学习去混杂估平均效应,再用因果森林看效应随收入怎么变、谁获益最多。」 | 朴素差 **\$19,559** → DML **~\$9,900**(发表 ~\$9,000)、CATE 跨度 ~\$23k、头号修饰变量=收入 |
| `mediation_jpsp2023` | 中介效应(bootstrap) | `social_science_econometrician` · causal-identification | 「检验'英雄之旅重述 → 意义感'是否经由英雄之旅感知(HJS)中介,给 5000 次 bootstrap 的间接效应。」 | ACME **0.309** [0.083, 0.532](发表 .31 [.08,.53]) |
| `survival_rossi` | Cox 比例风险生存 | `social_science_econometrician` · survival-analysis | 「做累犯的 Cox 比例风险:经济资助、年龄、前科对再次被捕风险的效应,查比例风险假设。」 | fin/prio/age logHR **−0.379 / +0.091 / −0.057**(Allison 2014 逐位吻合) |

## B. 现代因果三大件(0.3.0 新,内置数据零门槛)

| id | 能力 | agent · skill | prompt | 预期 |
|---|---|---|---|---|
| `dag_identify_refute` | 因果图识别 + 四步反驳 | `causal_data_scientist` · causal-dag | 「把'混杂同时影响处理和结果'画成 DAG,d-分离找最小充分调整集,估效应后做安慰剂/随机共因/子样本/隐藏混杂反驳。」 | 后门策略、调整集 {Z}、ATE **1.5**(朴素偏到 2.4)、placebo≈0、verdict robust |
| `modern_did` | Sun-Abraham / did2s / LP-DiD | `causal_data_scientist` · modern-did | 「交错采纳、效应随时间增长,别用经典 TWFE——用交互加权、两步、局部投影三种异质稳健估计量给动态路径。」 | did2s ATT 复原真值、TWFE 明显偏低、前导期≈0、动态效应逐期增长 |

## C. 补齐的量化方法(内置玩具数据)

| id | 能力 | agent · skill | prompt | 预期 |
|---|---|---|---|---|
| `multilevel_hlm` | 多层线性模型 HLM | `social_science_econometrician` · multilevel-modeling | 「学生嵌套在学校里,普通回归违反独立性。跑分层线性模型:随机截距+斜率,给方差成分和 ICC。」 | ICC **0.52**、x 斜率 ~**2.0** |
| `survival` *(见 A 的 Rossi)* | — | — | — | — |
| `spatial_moran_sar` | Moran 自相关 + SAR | `social_science_econometrician` · spatial-analysis | 「检验空间聚集:算全局 Moran's I 和局部 LISA,再跑空间滞后回归 SAR。」 | Moran I **0.337**(p=0.001)、SAR ρ **0.52** |
| `network_ergm` | 社会网络 + ERGM | `causal_data_scientist` · network-analysis | 「从边列表构网,给中心性和密度,再用 ERGM 检验结构项。」 | 24 节点 / 69 边、密度 0.25、ERGM 系数估出 |
| `qca_fsqca` | 模糊集 QCA | `qualitative_researcher` · qca | 「用模糊集 QCA 找导致结果的条件组态:校准、真值表、布尔最小化,给一致性和覆盖度。」 | 解 **C + A\*B**、一致性 0.97、覆盖度 0.98 |
| `demography_kitagawa` | 生命表 + Kitagawa 分解 | `social_science_econometrician` · demographic-analysis | 「构建周期生命表给预期寿命,再用 Kitagawa 把两组率差拆成'构成'和'率'两部分。」 | 预期寿命 e0 有限、率+构成 = 总差(精确加和) |

---

## 数据

- **A 组(4 份公开复现数据)**在首次运行时下载到 `benchmarks/data/`(缓存,已 `.gitignore`,不提交他人数据):HH2015(Harvard Dataverse)、401(k) SIPP1991(DMLonGitHub)、Rogers 2023(OSF)、Rossi(Rdatasets)。均为公开学术复现数据,benchmark 只做统计计算并与发表值对照。
- **B/C 组**用 socialverse 自带玩具数据(`socialverse.datasets.load_*`),内置于包、无需下载。

## 设计

每个案例 = `benchmarks/cases/c*.py` 里一个 `CASE = Case(...)`,声明 `prompt`(自然语言)、`agent`/`skill`(路由)、`run()`(跑 socialverse 链)、`check()`(逐项断言 vs 预期)。所有链都遵循 socialverse 的**依赖注册表契约**——`registry_lookup` 查而非猜、`did` 前置 `parallel_trends`、`dag_refute` 前置 `dag_identify`,前置未满足抛 `RegistryError`。
