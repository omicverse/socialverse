The report is a pure synthesis of the 6 research streams. Here it is.

---

# 人文社科计算分析生态调研综合报告 —— 指导 socialverse 下一步

调研截至 2026-07。综合 6 路原文(py-packages / r-packages / analysis-taxonomy / digital-humanities / agent-simulation / registry-prior-art)。核心判断:**socialverse 定位的空白是真实、结构性、且大的——可复刻的不是算法(已点状齐全),而是 omicverse 那张「注册表脊柱」。**

---

## 1. 现状全景:Python 与 R 各有什么,范式差异在哪

### 1.1 两边的主力包(点名)

| 领域 | R 主力(冠军包) | Python 主力 | 成熟度对比 |
|---|---|---|---|
| 面板/固定效应 | **fixest**(feols,C++ 定点吸收,业界最快)、plm | **pyfixest**(fixest 移植,336★)、linearmodels | R 领先,Py 追赶中且活跃 |
| DiD/事件研究 | **did**(Callaway-Sant'Anna)、didimputation、did2s、sunab | pyfixest 内建 + differences/pydid(零散) | R 有权威包,Py 无统一 DiD 包 |
| 因果(方法论) | dagitty、mediation、sensemakr、grf、DoubleML(R) | **DoWhy**(8.2k★)、EconML、DoubleML、causal-learn | Py 在 ML×因果侧反超(PyWhy 生态) |
| 合成控制/RDD | augsynth、gsynth、Synth、rdrobust 生态 | **CausalPy**(贝叶斯) | R 权威,Py 单薄 |
| 复杂抽样 | **survey**(Lumley,事实标准)、srvyr | **samplics**(唯一对标)→svy(预发布) | R 碾压;Py 无 svydesign 式统一设计对象 |
| SEM/CFA | **lavaan**(S4,504★)、blavaan、semTools | **semopy**(唯一成熟,lavaan 语法) | R 碾压 |
| IRT/心理测量 | **mirt**、psych、ltm、TAM | factor_analyzer(仅 EFA/CFA) | R 碾压,Py 无成熟 IRT |
| 多层/纵向 | **lme4**、brms、nlme | PyMC(手写)、statsmodels(弱) | R 碾压 |
| 事件史/生存 | **survival**(Therneau,基石)、flexsurv、mstate | lifelines | R 领先 |
| 文本即数据 | **quanteda**(corpus/tokens/dfm)、**stm**、tidytext、Wordfish | **BERTopic**(7.7k★)、spaCy、gensim、sentence-transformers | 各有侧重;Py 强嵌入,R 强 dfm+协变量主题(stm) |
| 网络 | **igraph**、statnet(**ergm**/tergm)、**RSiena**(SAOM) | networkx/igraph/graph-tool | ERGM/SAOM 是 R 独占,Py 无原生对等 |
| 空间 | **sf**、spdep、spatialreg、GWmodel | **PySAL**(元包,libpysal.W)、geopandas | 罕见地两边都成熟;Py 有唯一「领域元包」 |
| QCA | **QCA**(Dusa,v3.23)、SetMethods、cna | (无成熟 Python) | R 独占 |
| ABM 仿真 | NetLogoR | **Mesa**(3.7k★,Mesa 3)、agentpy | Py 领先 |
| 人口学 | **demography**、DemoTools、HMDHFDplus、oaxaca | (零散) | R 领先 |
| 文体计量/DH | **stylo**(事实标准,GUI)、TEI/lxml、CollateX、CATMA、Kraken、DraCor | faststylometry、pystyl(碎片化) | R/专用工具主导,Py 碎片 |

**一句话:每个子领域 R 都有「专业冠军包」,Python 是点状追赶——文本嵌入、ML×因果、ABM 领先,而抽样/SEM/IRT/多层/ERGM/QCA 明显落后甚至空白(只能调 R)。**

### 1.2 范式差异(根本分野)

**R = 语言层的公共契约(去中心化隐式注册表):**
- **formula 接口**:`y ~ x1 + x2 + (1|group)` 是一等公民,换包(lm→feols→lmer→plm)公式几乎不变,认知迁移成本极低。
- **data.frame 唯一通货**:带 factor(有序/无序因子=社科分类变量原生表达),base/tibble/data.table 可互换。
- **富 model 对象(S3/S4)+ generic 分派**:`lm()` 返回装着系数/残差/vcov/调用的对象;`summary/predict/confint/vcov` 是泛型,靠 class 分派。**任何新包只要实现这几个 generic 就自动接入整个下游工具链**——这是可组合性的机制核心。
- **broom(tidy/glance/augment)**:三泛型把 100+ 种富对象拍平成整洁表 → 可 ggplot / 可出论文表。easystats 的 `insight` 是同思想的适配层(抹平 45+ 种对象,被 45 个 CRAN 包依赖)。

**Python = 各自为政的对象(无公共契约):**
- BERTopic 吃 `List[str]`;spaCy 吃自定义 `Doc/Token`;gensim 吃 `Dictionary`+BoW;DoWhy 吃 DataFrame+因果图;EconML 走 sklearn `X,y`;DoubleML 强制自建 `DoubleMLData`;pyfixest 吃 pandas+R 公式串;semopy 用 lavaan 公式对象;networkx/igraph/graph-tool 三套图对象互不兼容。
- 科学栈以 **ndarray/DataFrame + OO fit/predict** 为中心,为**ML 预测**优化;R 以 **formula + 富对象 + generic** 为中心,为**估计一个参数并解释它**优化。社科要的是稳健 SE、边际效应、效应量、论文表格——恰是 R 沉淀最厚处。

**结论:R 的「统一」不是靠巨包,而是靠 formula+S3/S4+broom/insight 四层契约 + tidyverse(数据前端)/easystats(报告后端)两个互补元包 + CRAN Task Views(人工策划目录)。Python 缺的正是这套公共契约层。**

---

## 2. 有没有「统一/agent-oriented 社科总包」?——空白 = 护城河

**明确结论:没有前人做出「注册表脊柱 + StudyState 统一容器 + registry_lookup grounding」三位一体。omicverse 在社科的直接对应物是空位。**

### 2.1 「统一总包」层面

- **R**:无单一总包(`socsci` 只是教学便利封装)。统一靠语言契约 + 两元包 + Task Views。
- **Python**:唯一「领域元包」是 **PySAL**(空间聚合十几个子包,靠 libpysal.W + GeoDataFrame 统一)——但**只管空间**,不跨文本/因果/网络。最接近「总包」的只是**同组织松散生态**:PyWhy(DoWhy↔EconML 真互通,仅因果)、py-econometrics(围绕 fixest 公式,共享作者非统一 runtime)、samplics→svy(仍单领域)。**没有一个包「一个 import 打通文本→因果→网络→抽样」。**

### 2.2 「注册表 / 契约 / grounding」层面(socialverse 的真正差异点)

前人四类竞品**各触及一角,无人三位一体**:

| 竞品 | 触及什么 | 缺什么(vs socialverse) |
|---|---|---|
| **gojiplus/rmcp**(R 计量 MCP,205★) | 52 个统计工具暴露给 agent | **无契约**:429 包 CRAN 白名单是静态清单;工具一字排开,无 requires/produces/prerequisites,无 planner,无统一分析态对象 |
| **ORCA**(arXiv 2508.21304,电商因果) | 有共享中间态、Config Selector | 方法选择靠**prompt 塞选项让 LLM 挑**,非机器可读契约;因果单点,非社科通用 |
| **S-Researcher/YuLan-OneSim**(人大,2026) | 40+ 工具 registry、动态选择 | 重心是 **ABM 仿真+报告生成**(造仿真数据),registry 无正式 requires/produces 契约,无统一研究态容器 |
| **ToolGate / Contract2Tool**(2026 通用) | Hoare 式契约(pre/postcondition)形式化 | **纯通用无领域**:无研究态槽位词汇表、无 auto_fix、无社科方法 prerequisites |
| **targets+Quarto+renv** | 可复现 DAG | **文件/对象级构建系统**,非 agent 方法级契约;可叠加非替代 |

**agent-simulation 赛道(8 平台)全部做「造模拟人」不做「分析真实数据」**:Generative Agents、Concordia、AgentSociety、OASIS、SocioVerse、GenSim、AgentTorch、S3。真实数据永远只出现在两个位置——(a) 对齐/初始化 agent 画像,(b) 验证模拟是否逼真。**没有一个把真实微观数据(CGSS/CFPS/ANES/面板)当分析对象去跑因果识别+加权推断+稳健性+证据链。**

**socialverse 独有的三点组合(护城河):**
1. **注册表即脊柱**(非容器优先):`@register` 的 requires/produces/prerequisites/**auto_fix** 依赖图当根,方法是二等公民由契约挂载。
2. **社科专属研究态槽位词汇表(StudyState)**:定义 DiD/RDD/IV/面板/调查加权/固定效应/聚类 SE 等前置条件为机器可读槽位,让 fixest/DoWhy/semopy 按契约注册。**社科完全空白——无人定义过社科统一研究态词汇表。**
3. **registry_lookup grounding(查而非猜)**:agent 查注册表决定合法下一步(前置是否满足、缺什么触发 auto_fix),而非 prompt 塞方法描述让 LLM 挑。

> **命名风险(务必留意)**:**SocioVerse**(复旦 DISC,arXiv 2504.10157,社会仿真 world model)已占用同名+同赛道话语权,强烈建议区隔;`socialverse` 的 PyPI 名目前大概率可注册,但与复旦 SocioVerse 读音/联想混淆。**建议叙事上强调「真实数据/证据/工作台」而非「verse/世界」,与整片「造世界」的模拟赛道切割。**

---

## 3. 社科分析方法全景表(输入形态 → 方法 → 代表软件 → socialverse 是否覆盖)

覆盖状态基于现有 34 函数(sv.pp/tl/pl/gov/lit)。标注: ✅已覆盖 / 🟡部分或邻近 / ❌缺口。

| 家族 | 输入形态 | 代表方法 | 代表软件 Py / R | socialverse |
|---|---|---|---|---|
| **计量·面板** | unit×time long | FE/RE、TWFE、聚类稳健 SE | pyfixest / **fixest** | ✅(因果计量) |
| **计量·IV** | 截面+工具 | 2SLS/GMM、弱工具诊断 | linearmodels / fixest·ivreg | ✅ |
| **计量·DiD** | 交错处理面板 | Callaway-Sant'Anna、Sun-Abraham、Borusyak | pyfixest·differences / **did** | 🟡(需确认交错估计量) |
| **计量·RDD** | 断点+running var | 局部多项式、最优带宽、稳健 CI | CausalPy / **rdrobust** | ❌ **缺口** |
| **计量·合成控制** | 处理单元+对照池 | SCM、增广 SCM、gsynth | CausalPy / **augsynth·gsynth** | ❌ **缺口** |
| **计量·分位数** | 截面/面板 | QR、面板分位数 | statsmodels / **quantreg** | 🟡 |
| **抽样统计** | 设计信息微观数据(strata/PSU/权重) | 加权估计、Taylor/BRR/jackknife、raking、后分层 | samplics / **survey·srvyr** | ✅(复杂抽样) |
| **心理测量·信度/因子** | 被试×题项 | α/ω、EFA/CFA | factor_analyzer / **psych·lavaan** | 🟡 |
| **心理测量·SEM** | 潜变量结构 | 路径/CFA/成长曲线 | **semopy** / lavaan | ❌ **缺口(SEM)** |
| **心理测量·IRT** | 二分/有序作答 | Rasch/2PL/3PL、GRM、DIF | (无) / **mirt·TAM** | ❌ **缺口** |
| **多层/纵向** | 嵌套 long | HLM、随机斜率、增长曲线 | PyMC / **lme4·brms** | 🟡 |
| **事件史/生存** | 生存对象 | Cox PH、离散时间、竞争风险 | lifelines / **survival** | ❌ **缺口** |
| **文本·主题** | 语料/DFM | LDA、STM(协变量)、BERTopic | **BERTopic** / **stm·quanteda** | ✅(文本) |
| **文本·标度** | 政治文本 | Wordfish/Wordscores | / **quanteda.textmodels** | 🟡 |
| **文本·嵌入** | 原始文本 | sentence-transformers、NER | **sentence-transformers·spaCy** / text | ✅ |
| **网络·描述** | 边列表/邻接阵 | 中心性、社群(Louvain/Leiden) | networkx·igraph / **igraph** | 🟡(需确认) |
| **网络·ERGM** | 网络+tie 变量 | 指数随机图、TERGM | (无) / **ergm·tergm** | ❌ **缺口** |
| **网络·SAOM** | 网络面板快照 | 随机 actor 导向、网络-行为共演化 | (无) / **RSiena** | ❌ **缺口** |
| **因果·DAG** | 数据+因果假设 | 调整集、后门/前门、d-分离 | **DoWhy·dagitty** | ✅(因果) |
| **因果·中介/敏感性** | 处理-中介-结果 | 中介效应、E-value | / **mediation·sensemakr** | 🟡 |
| **因果·ML** | 高维协变量 | DML、因果森林、TMLE、CATE | **DoubleML·EconML** / grf | 🟡 |
| **空间** | 几何+权重 W | Moran/LISA、SAR/SEM、GWR | **PySAL** / **sf·spdep·GWmodel** | ❌ **缺口** |
| **仿真 ABM** | 规则/参数 | 主体建模、微观模拟 | **Mesa** / NetLogoR | ❌(超范围?) |
| **质性·QCA** | 案例×条件表 | csQCA/fsQCA、真值表最小化 | (无) / **QCA·SetMethods** | ✅(质性编码?需确认) |
| **质性·编码** | 访谈/语料 | 主题分析、standoff 标注 | qualcoder / (CAQDAS) | ✅(质性编码) |
| **人口学** | 年龄别率 | 生命表、标准化、Kitagawa-Oaxaca 分解 | / **demography·oaxaca** | ❌ **缺口** |
| **数字人文·文体** | 数字语料/TEI | Delta 距离、作者归属、bootstrap 树 | faststylometry / **stylo** | ✅(校勘?需确认覆盖 stylo) |
| **数字人文·校勘** | 多见证本 | variant graph、对齐 | collatex / (CollateX) | ✅(校勘) |
| **治理** | — | (socialverse 特有 gov 模块) | — | ✅ |
| **文献** | 引文/文库 | 文献综合 | — / — | ✅(lit) |

---

## 4. socialverse 覆盖 vs 缺口(对照现有 34 函数)

### 4.1 已覆盖(现有 sv.pp/tl/pl/gov/lit 5 模块推断)

- **因果计量**:面板 FE/IV、因果 DAG/DoWhy 式识别-估计-反驳 ✅
- **复杂抽样**:设计基础加权推断(对标 survey/samplics)✅
- **文本**:主题建模 + 嵌入(对标 BERTopic/stm)✅
- **质性编码**:CAQDAS 式编码/standoff ✅
- **校勘**:variant graph 校勘(对标 CollateX)✅
- **理论透镜 / 治理(gov) / 文献(lit)**:socialverse 特有,无外部直接对标(差异化)✅

### 4.2 明显缺口(需补,按社科使用频率与护城河价值排序)

**P0(社科高频、审稿硬需求、且外部有权威包可契约挂载):**
1. **SEM/CFA**(semopy/lavaan)—— 心理测量/社会学/传播学核心,无之则量表分析全断
2. **IRT**(mirt)—— 教育测量/量表,Python 侧本就空白,补上即领先
3. **RDD**(rdrobust)—— 因果四大准实验之一,现仅有 DiD
4. **合成控制**(augsynth/gsynth)—— 政策评估旗舰方法
5. **多层/HLM**(lme4/brms)—— 嵌套数据(学生嵌套学校)基础设施

**P1(领域刚需,部分 Python 空白正是机会):**
6. **事件史/生存**(survival)—— 政治学(政权存续)、社会学(职业流动)
7. **空间分析**(PySAL/spdep)—— Moran/LISA/SAR,地理/城市/政治
8. **ERGM**(ergm)—— 社会网络推断黄金标准,**Python 无原生对等 = 高护城河**
9. **RSiena/SAOM**—— 网络-行为共演化,同样 Python 空白

**P2(补全全景):**
10. **QCA/fsQCA**(QCA)—— 若现有「质性编码」未含集合论最小化则需补,Python 空白
11. **人口学**(生命表/Kitagawa-Oaxaca 分解)—— 人口学/健康
12. **网络描述**(中心性/社群)—— 若现有网络覆盖不足需确认
13. **文体计量 stylo**(作者归属/Delta)—— 若校勘模块未含则补 DH 另一半

> **注**:上述「是否已覆盖」有若干需对现有 34 函数逐一核对(网络描述、QCA、stylo、DiD 交错估计量)。建议下一步先做一次 34 函数 × 本全景表的逐格 gap audit,产出精确的覆盖矩阵。ABM 仿真属「造模拟人」赛道,建议**不纳入** socialverse(与定位「分析真实数据」切割)。

---

## 5. 给 notebook/教程的建议

### 5.1 设计原则

- **每本 notebook = 一条真实分析链**(不是函数罗列),从「拿到数据」走到「可复核结论 + 论文级输出」,全程展示 **StudyState 槽位如何被填充、registry_lookup 如何 grounding、auto_fix 如何触发**——这是与 rmcp/AgentSociety 切割的叙事核心:**别只演示「能调方法」,要演示「注册表如何决定下一步合法性」**。
- **玩具数据用经典公开数据集**(可 `sv.datasets` 内置,端到端可跑、有稳定输出),避免真实受限数据的合规问题。
- **每本结尾产出一个「证据链卡片」**(数据→识别策略→估计→稳健性→结论 provenance),这是 socialverse 独一份的能力,每本都要展示。

### 5.2 建议的 notebook 清单(每本一条链,覆盖全部功能)

| # | Notebook 标题 | 分析链 | 玩具数据 | 演示函数(现有+待补) | 预期输出 |
|---|---|---|---|---|---|
| **01** | **快速上手:注册表脊柱与 StudyState** | 载入数据→注册表自省→查 registry_lookup→看 auto_fix | 内置小面板 | 核心 registry API、StudyState 12 槽位、@register 契约 | 依赖图可视化、槽位填充表、「查而非猜」演示 |
| **02** | **因果计量:面板固定效应 + DiD 政策评估** | 面板→设计声明(处理/时点)→TWFE→交错 DiD 稳健性→event-study 图 | 内置政策面板(类 fixest 数据) | sv.tl 因果/面板、DiD、聚类 SE、(待补 Callaway-SA) | 系数表+稳健 SE、event-study 图、证据链卡片 |
| **03** | **准实验三件套:IV + RDD + 合成控制** | 同一政策问题三种识别→反驳测试→对比 | 内置断点/工具变量数据 | IV(已有)、**RDD(待补 rdrobust)**、**合成控制(待补)**、DoWhy refute | LATE/局部效应、断点图、合成对照轨迹、识别策略对照 |
| **04** | **复杂抽样:设计基础加权推断** | 声明 svydesign(strata/PSU/权重)→加权均值/回归→BRR 方差 | 内置调查微观数据(类 NHANES) | sv.pp 抽样、加权估计、replicate weights | 设计一致总体估计+正确 SE、加权列联表 |
| **05** | **心理测量:量表从 EFA 到 SEM 到 IRT** | 题项矩阵→信度→EFA→CFA→结构模型→IRT 参数 | 内置量表作答矩阵 | **SEM(待补 semopy)**、**IRT(待补 mirt)**、信度、测量不变性 | 载荷、拟合指数(CFI/RMSEA)、题项信息函数、不变性层级 |
| **06** | **多层与纵向:HLM + 生存分析** | 嵌套数据→随机截距/斜率→方差成分;事件史→Cox→风险比 | 内置嵌套/生存数据 | **多层(待补 lme4 式)**、**生存(待补 survival)** | 方差成分、随机效应、生存曲线、HR |
| **07** | **文本即数据:主题模型 + 立场标度 + 嵌入** | 语料→DFM→STM(协变量)/BERTopic→Wordfish 标度→嵌入聚类 | 内置政治/新闻语料 | sv.tl 文本、主题、Wordfish、sentence-transformers | 文档-主题分布、一维立场、嵌入聚类图 |
| **08** | **网络分析:描述 + ERGM 生成机制** | 边列表→中心性/社群→ERGM 声明网络统计量→MCMC-MLE | 内置社交网络 | 网络描述、**ERGM(待补,Py 空白高护城河)**、(可选 SAOM) | 中心性排名、社群划分、ERGM 参数(互惠/传递/同质) |
| **09** | **空间分析:Moran + 空间回归** | 几何+W→全局/局部 Moran(LISA)→SAR/SEM→溢出分解 | 内置县级空间数据 | **空间(待补 PySAL/spdep 式)** | LISA 地图、空间效应直接/间接分解、系数地图 |
| **10** | **质性 + QCA:编码到组态分析** | 语料 standoff 编码→案例×条件表→fsQCA 真值表最小化 | 内置访谈/案例数据 | sv.gov 质性编码、**QCA(待补,Py 空白)** | 编码体系、充分/必要条件组态、真值表解 |
| **11** | **数字人文:校勘 + 文体计量** | 多见证本→variant graph 对齐→MFW+Delta→作者归属 bootstrap 树 | 内置多版本文本 | sv.gov 校勘、**stylo 式 Delta(待补/确认)** | 异文图、共识树、作者归属概率 |
| **12** | **治理 + 文献 + 理论透镜:研究闭环** | 文献综合→理论透镜选识别策略→治理/合规审计→证据链汇总 | 承接前 notebook 产物 | sv.lit 文献、sv.gov 治理、理论透镜 | 文献综述段落、透镜映射、完整 provenance 报告 |

> 12 本覆盖全部现有 34 函数 + 明确标出「待补」处对应的 5 大 P0 缺口。若缺口未补,可先出 8 本(01/02/04/07/08/10/11/12)覆盖现有能力,SEM/IRT/RDD/合成控制/多层/生存/空间随功能补齐再出 03/05/06/09。

### 5.3 教程整体结构(建议 5 部)

1. **Part 0 — 为什么是注册表**(叙事):社科生态「点状繁荣、彼此不互通」的问题 → R 的隐式契约 vs socialverse 的显式注册表 → StudyState 词汇表总览。对标 easystats/insight「事后适配层」的思路讲清 socialverse 是「前置契约层」。
2. **Part 1 — 核心机制**(notebook 01):registry / @register 契约 / StudyState 12 槽位 / registry_lookup / auto_fix。所有后续 notebook 的地基。
3. **Part 2 — 定量分析链**(notebook 02–09):因果计量→准实验→抽样→心理测量→多层/生存→文本→网络→空间。每本一条真实链 + 证据链卡片。
4. **Part 3 — 质性与人文链**(notebook 10–11):QCA/质性编码 + 校勘/文体计量。展示 socialverse 跨「定量-质性-人文」的独有广度(无竞品覆盖)。
5. **Part 4 — 研究闭环与治理**(notebook 12):文献→理论透镜→治理/证据链。收束到「可复核研究」这一差异化定位。
   - **横切附录**:每个方法给一张「契约卡」(requires/produces/prerequisites/auto_fix + 对标的 Py/R 权威包),既是文档也是 registry 的人类可读视图(对标 CRAN Task Views 的「策划目录」思路)。

---

## 核心 takeaway(给决策)

1. **空白真实**:无人做过「注册表脊柱 + StudyState + grounding」三位一体;算法点状齐全但无公共契约层——这正是可填的护城河。
2. **护城河 = 契约,不是方法数量**:rmcp(205★,增长中)会覆盖「暴露方法」,socialverse 必须落在 requires/produces/prerequisites/**auto_fix** + 研究态词汇表,而非工具数量。
3. **P0 缺口**:SEM、IRT、RDD、合成控制、多层——社科审稿硬需求,补齐才能撑起 notebook 05/06 与 03。**ERGM/SAOM/QCA 是 Python 原生空白 = 高护城河区**,值得优先自研或桥接 R。
4. **叙事切割**:强调「分析真实世界」对抗整片「造模拟世界」赛道;命名注意与复旦 **SocioVerse** 区隔。
5. **下一步先做**:34 函数 × 全景表逐格 gap audit(确认网络描述/QCA/stylo/DiD 交错估计量的真实覆盖),再据此定 8 本 or 12 本 notebook 的发布顺序。