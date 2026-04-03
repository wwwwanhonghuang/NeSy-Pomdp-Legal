# Title: Towards a Neuro-Symbolic POMDP Framework for Sequential Legal Judgment with Structured Brief Particles

## Abstract

Neural-symbolic AI is ...., [it can used in computational law can do tasks like xxx]. the problem is xx. in this work we propose A Neuro-Symbolic POMDP Framework, that integrated brief and evidence based decision making framework with neuro-symbolic model. 我们主张我们的优势是 (1)**优势 1：序贯证据累积下的决策质量**, (2) **优势 2：信息收集行动的价值**,(3) **优势 3：不确定性校准（Calibration）**. 

[本工作将如何评估]，[本工作的期待成果以及简要贡献]



> ------
>
> Neuro-symbolic (NeSy) AI has shown promise in computational law, enabling legal judgment prediction, charge classification, and argument mining by combining neural pattern recognition with symbolic reasoning. However, existing systems treat legal decision making as a single-pass inference problem — consuming all evidence at once and producing a verdict in one forward pass. Real legal proceedings are inherently sequential: evidence accumulates incrementally, epistemic uncertainty evolves at each step, and a rational agent should defer judgment or actively request further information when uncertainty is too high to act justly. No existing framework formalizes this process.
>
> We propose a Neuro-Symbolic POMDP framework in which a neural encoder maps evidence onto a continuous embedding manifold, and a conditional structured sampler generates a particle set ${(\text{brief}*i, w_i)}*{i=1}^N$ over structured argument graphs — each particle a formal brief $(C, R^+, R^-, \lambda)$ of claims, support/attack relations, and evidence anchors. This particle representation preserves genuine argument-level uncertainty across competing legal interpretations. A POMDP planning layer treats the weighted particle set as observations, maintains a belief state $b(s)$ over hidden legal states, and optimizes a policy that may verdict, defer, or issue information-gathering actions.
>
> We claim three quantifiable advantages over static NeSy baselines: (1) higher accuracy under partial evidence, measured by area under the Acc@k curve; (2) positive value of information-gathering actions, quantified as the EVPI gap between constrained and unconstrained policies; and (3) better-calibrated uncertainty, measured by ECE and Brier Score against LLM baselines.
>
> We evaluate on a sequential evidence simulation protocol derived from CAIL2018 and compare against four baselines: static NeSy-LJP, one-shot LLM, POMDP without structured observations, and NeSy without sequential planning.
>
> Contributions: (i) the first POMDP framework with structured argument particles as legal observations; (ii) a particle-based brief representation bridging neural embeddings and abstract argumentation theory; (iii) a reproducible sequential evidence benchmark from CAIL2018.

# 1 Introduction

# 2 Overview



# 3 Method

The overview is

``` 
Evidences -> Neuron -> embeddings (smooth manifold \mathcal M) -> [brief update model -> brief -> action -> new evidences -> ..
```

![image-20260328130821996](C:\Users\Micro\AppData\Roaming\Typora\typora-user-images\image-20260328130821996.png)



## 3.1 Evidence

### 3.1.1 The 认识论 of Evidences

This section describe what is evidences for us in our task, and why.



### 3.1.2 The 表现论 of Evidence

How us present evidence

#### 3.1.2.1 Spatial Structure

#### 3.1.2.2 Temporal Structure



### 3.1.3 The 实践论 of Evidence

What evidence in our dataset, where it comes from, etc.



## 4 Law and Formal Logic





## 5 The NeSy-POMDP Framework

## 5.1 Neural Enc.

## 5.2 Neural Update Model

## 5.3 Symbolic Mediation

## 5.4 Belief Update and Policy π(b)

## 5.5 Action Space and Termination





## 6 Law and Logics (Symbolic Order) Mediation





# Q/A

1. 和POMDP整合，我们主张的优势是什么呢？我们如何定量评估证明我们的主张。

> 与现有方法相比，POMDP 整合带来三个**可区分的、可量化的**优势：
>
> **优势 1：序贯证据累积下的决策质量**
>
> 现有 NeSy 法律 AI（Prolog-CoT、LJP 系统）是**单次推断**——给定所有证据，输出判决。但真实法律程序是序贯的：证据逐步呈现，每步都可能改变推断。POMDP 的主张是：在证据不完整的中间状态下，我们的策略比贪心推断更优。
>
> 可量化指标：**在证据子集上的判决准确率曲线**（accuracy vs. evidence completeness ratio）。我们的系统应在低完整度时显著优于静态方法，在高完整度时收敛到相同水平。
>
> **优势 2：信息收集行动的价值**
>
> POMDP 策略可以输出"请求更多证据"而不是立即判决。这在法律上对应申请鉴定、传唤证人等。主张：当允许信息收集行动时，最终决策质量（准确率 + 校准度）应显著提升。
>
> 可量化指标：**Value of Information（VOI）**——$\text{VOI} = V^*(\text{with info action}) - V^*(\text{without info action})$，在模拟法律案例中测量。
>
> **优势 3：不确定性校准（Calibration）**
>
> 现有方法输出一个判决，不输出置信度，或输出的置信度未经校准。POMDP 的信念状态 $b(s)$ 是显式的概率分布，应当是**校准的**——即声称 70% 有罪时，真实有罪率接近 70%。
>
> 可量化指标：**Expected Calibration Error（ECE）** 和 **Brier Score**，与 LLM baseline 对比。
>
> 三、关键的实验设计
>
> **核心实验：证据完整度 vs. 准确率曲线**
>
> 将 CAIL 案件的证据拆成有序序列（按呈现顺序：起诉书 → 证人证词 → 物证 → 鉴定意见），在每个节点 $k$ 处让各方法做出判决：
>
> $$\text{Acc@k} = \frac{1}{|D|} \sum_{d \in D} \mathbf{1}[\hat{y}^{(k)}_d = y_d]$$
>
> 我们的系统在 $k$ 小（证据稀疏）时应显著优于 B1/B2，因为 POMDP 策略会选择延迟判决（defer action），而静态方法被迫猜测。在 $k$ 大时所有方法收敛。这条曲线的 **AUC 差异**就是可量化的主要主张。
>
> **信念校准实验**
>
> 与 LLM baseline 相比，用可靠性图（reliability diagram）展示 $b(s)$ 的校准性。法律 AI 的一个严重问题是过度自信——模型说"90% 有罪"但实际上只有 60% 的案件是有罪的。POMDP 信念状态通过贝叶斯更新天然具有比 LLM logit 更好的校准性，这是一个**可直接测量的理论预测**。
>
> **粒子集质量实验**
>
> 用 **Effective Sample Size（ESS）** 衡量粒子退化程度：
>
> $$\text{ESS} = \frac{(\sum_i w_i)^2}{\sum_i w_i^2}$$
>
> ESS 越低说明粒子集退化为单一假设（确定），ESS 越高说明保持了多样性。对于法律上真正模糊的案件，我们期望 ESS 保持高值直到关键证据出现，之后快速下降——这个形状本身就是一个可解释的信号。
>
> 四、一个诚实的警告
>
> 目前没有任何现成的 benchmark 直接支持"序贯证据呈现下的法律判决"这一设置——这本身就需要**数据集构建**作为论文贡献之一。可行路径是：从 CAIL 的案情描述中用 LLM 提取证据时间线，构建一个半自动的序贯证据数据集，作为新 benchmark 贡献，然后在上面评估所有方法。这样论文就有三个贡献：理论框架、数据集、实验评估。