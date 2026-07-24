**硬件约束驱动（hardware-constrained）、噪声感知（noise-aware）、R-GAT增强的强化学习量子路由框架**。

核心思想：
不要让RL/GNN学习硬件规则，而是利用coupling graph提供硬约束和合法动作空间；利用R-GAT学习线路状态、错误传播、串扰等复杂因素，在合法SWAP中选择最优动作。


# 1. 总体框架

目标：

解决超导量子计算中的：

[
\text{Logical Circuit}
\rightarrow
\text{Physical Hardware}
]

映射问题。

给定：

* 量子线路
* 当前logical-physical mapping
* 超导硬件拓扑
* 校准噪声信息

输出：

* SWAP插入策略

优化：

* SWAP数量
* circuit depth
* execution fidelity

---

# 2. 核心设计思想

整个系统分为三个部分：

```
              Quantum Circuit

                    |
                    |
          Noise-aware R-GAT Encoder
                    |
                    |
             State Embedding

                    |
                    |
     --------------------------------
     |                              |
Coupling Graph              Candidate SWAP
(Hardware Constraint)       Generation
     |                              |
     --------------------------------

                    |
              RL Decision

                    |
              Insert SWAP

                    |
          Executability Check

                    |
              Reward Update
```

---

# 3. 状态表示设计

状态：

[
s_t=(C_t,M_t,H)
]

其中：

| 符号    | 含义                       |
| ----- | ------------------------ |
| (C_t) | 当前未执行线路                  |
| (M_t) | logical-physical mapping |
| (H)   | hardware coupling graph  |

---

# 4. GNN编码方案

最终倾向采用：

[
\boxed{
Circuit\ R-GAT
+
Hardware\ Constraint
}
]

而不是完全双图融合。

原因：

* coupling graph本身承担硬约束；
* GNN主要学习线路和噪声特征；
* 降低模型复杂度。

---

# 5. Circuit R-GAT编码

## 5.1 图结构

采用：

Gate Dependency Graph

节点：

[
V={g_i}
]

每个节点表示一个量子门。

例如：

```
g1(CX)
 |
g2(RZ)
 |
g3(CX)
```

---

# 5.2 Node Feature

每个gate节点：

[
x_i=
[
x_i^{gate},
x_i^{noise},
x_i^{error},
x_i^{mapping}
]
]

---

## (1) Gate信息

包括：

* gate type
* layer
* duration
* qubit number

---

## (2) Calibration noise

来自硬件校准：

* gate error rate
* T1
* T2
* readout error

---

## (3) Error propagation

描述错误影响范围：

包括：

### 后继gate数量

[
F_{out}
]

### Error influence

[
I_i=
\sum_j
\epsilon_i
\gamma^{d(i,j)}
]

### Criticality

表示该门错误的重要程度。

---

## (4) Mapping信息

当前：

[
q_i\rightarrow Q_j
]

加入：

* physical distance
* CX fidelity
* SWAP cost
* executable probability

---

# 6. R-GAT关系设计

采用多关系attention：

[
R=
{
dependency,
error,
crosstalk
}
]

---

## Relation 1：Dependency

表示线路执行顺序：

[
g_i\rightarrow g_j
]

---

## Relation 2：Error propagation

表示错误传播：

[
g_i\rightarrow g_j
]

edge:

[
w_{ij}^{err}
]

考虑：

* error rate
* qubit overlap
* time interval

---

## Relation 3：Crosstalk

表示并行操作干扰：

[
g_i\leftrightarrow g_j
]

edge feature：

* crosstalk strength
* frequency difference
* spatial distance

---

# 7. Hardware Coupling Graph作用

硬件图：

[
G_H=(V_H,E_H)
]

其中：

节点：

physical qubit

边：

coupling relation

例如：

```
Q0 --- Q1 --- Q2
```

---

它承担两个作用：

---

## 作用1：判断gate是否可执行

对于：

[
CX(q_i,q_j)
]

当前：

[
M(q_i)=Q_a
]

[
M(q_j)=Q_b
]

检查：

[
(Q_a,Q_b)\in E_H
]

如果：

True：

执行gate。

如果：

False：

需要routing。

---

## 作用2：生成合法SWAP动作

动作空间：

[
A=
{SWAP(Q_i,Q_j)|(Q_i,Q_j)\in E_H}
]

即：

只允许hardware coupling上的SWAP。

避免RL产生非法动作。

---

# 8. RL决策方式

不是：

```
GNN → 直接输出SWAP
```

而是：

```
GNN → 表征当前状态

Coupling graph → 生成合法SWAP候选

RL → 在候选中选择最佳SWAP
```

---

对于每个candidate：

[
a_i=SWAP(Q_m,Q_n)
]

计算：

[
Q(s,a_i)
]

选择：

[
argmax Q(s,a_i)
]

---

# 9. 状态更新流程

每一步：

## Step 1

检查front layer：

[
Executable(front,M,H)
]

---

## Step 2

如果可执行：

执行gate：

[
C_t=C_t-\Delta C
]

---

## Step 3

如果不可执行：

生成：

[
SWAP\ candidates
]

---

## Step 4

RL选择：

[
a_t
]

更新mapping：

[
M_{t+1}
]

---

## Step 5

重新检查：

直到产生新的可执行gate。

---

# 10. Reward设计

目标：

同时考虑：

* 完成门数量
* SWAP代价
* 深度
* fidelity

推荐：

[
\boxed{
R=
\alpha\Delta N_{gate}
-\beta N_{SWAP}
-\gamma\Delta Depth
+\eta\Delta Fidelity
}
]

其中：

---

## Gate progress

[
\Delta N_{gate}
]

鼓励快速推进线路。

---

## SWAP cost

[
N_{SWAP}
]

减少额外操作。

---

## Fidelity

考虑：

* gate error
* calibration
* crosstalk

---

# 11. 与原方法相比的改进

原方法：

```
Circuit graph
      |
     GNN
      |
 RL chooses SWAP
      |
 executable check
```

你的方法：

```
Circuit graph
      |
Noise-aware R-GAT
      |
State representation


Hardware coupling graph
      |
Hard constraint
      |
Valid SWAP generation


RL chooses best SWAP
```

---

# 12. 最终模型名称可以定义为

例如：

**Hardware-Constrained Noise-Aware Relational Graph Attention Reinforcement Learning for Superconducting Quantum Routing**

简称：

[
\boxed{
HC-NA-RGAT-RL
}
]

---

# 13. 后续实验路线

建议逐步比较：

### Baseline 1

原始：

GraphSAGE + RL

---

### Baseline 2

GAT + RL

---

### Model 1

Gate-RGAT：

加入：

* dependency
* error propagation

---

### Model 2

Full：

Gate-RGAT

*

- crosstalk
- calibration
- hardware constraint

---

评价指标：

| 指标                  | 意义        |
| ------------------- | --------- |
| SWAP数量              | routing效率 |
| Circuit depth       | 执行时间      |
| Estimated fidelity  | 可靠性       |
| Success probability | 最终性能      |
| Runtime             | 算法效率      |

