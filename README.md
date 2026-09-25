# wayplan — 可分配缓存路数的性能隔离规划

离线体系结构优化工具库：对最多 6 个应用的合成访存块序列，逐路模拟
set-associative LRU miss 曲线，再用**精确动态规划**在路数预算内选择每应用的
隔离路数，最小化带正整数权重的 miss 总和，并在隔离缓存上重放统计。

v1.1 新增（沿原有数据流扩展，非替代实现）：

- **每应用可选 miss 硬上限**（`miss_caps`）：上限在 DP **内部**作为候选剪枝
  条件强制执行，任何超出任一硬上限的分配都不可行——不是先求无约束最优再
  局部修补（miss 曲线不保证凸，事后修补不可靠）；
- **可选付费停用**（`deactivation_penalties`）：显式声明可停用的应用可以被
  停用——支付给定整数惩罚、不占缓存路、在结果中单独报告；未声明的应用仍
  至少占一路；
- **不可行诊断**：`InfeasibleBudgetError` 携带各应用独立最低路数需求与总
  预算（仅为必要条件，见下文）；
- **重放核验**：`verify_plan` 对规划结果逐条重放确认每个硬约束。

- **Python 版本**：3.14.7（仅使用标准库，语法兼容 3.10+）
- **依赖**：无第三方依赖，无需网络、账号、数据库、Docker；安装 Python 即可运行
- **平台**：Linux 原生（纯 Python，任何平台均可）

## 快速开始

```bash
./.venv/bin/python -m unittest discover -s tests -v   # 运行全部测试（47 个）
./.venv/bin/python demo.py                            # 确定性演示（实测约 0.05 秒，远低于 90 秒上限）
```

```python
from wayplan import plan_allocation, verify_plan

traces = [("appA", [0, 1] * 4), ("appB", [7] * 5)]
result = plan_allocation(traces, weights=[3, 1],
                         num_sets=1, total_ways=3, budget=3,
                         miss_caps=[None, 2],            # appB 至多 2 次 miss
                         deactivation_penalties=[None, 10])  # appB 可付费停用
print(result.allocation_map())       # {'appA': 2, 'appB': 1}（0 表示已停用）
print(result.deactivated_app_ids())  # 被停用的应用（单独报告）
ver = verify_plan(result, traces)    # 实际重放确认每个硬约束
assert ver.ok, ver.failures
```

## 接口（固定契约）

| 名称 | 说明 |
|---|---|
| `AppTrace(app_id, blocks)` | 已验证的不可变轨迹；也接受 `(app_id, blocks)` 元组作为输入 |
| `simulate_lru_misses(blocks, num_sets, num_ways) -> int` | 单配置冷启动 LRU miss 数 |
| `miss_curve(blocks, num_sets, max_ways) -> tuple[int, ...]` | 下标 `w-1` 为 `w` 路的 miss 数 |
| `plan_allocation(traces, weights, num_sets, total_ways, budget, miss_caps=None, deactivation_penalties=None) -> PlanResult` | 精确 DP 最优分配（含硬上限与停用） |
| `replay_isolated(traces, allocation, num_sets) -> tuple[ReplayStats, ...]` | 隔离缓存重放统计（不含停用应用） |
| `verify_plan(result, traces) -> PlanVerification` | 对规划结果逐条重放核验全部硬约束 |
| `WayPlanError` / `ValidationError` / `InfeasibleBudgetError` | 错误分类 |
| `MAX_APPS` / `MAX_SETS` / `MAX_WAYS` | 接口上限常量 6 / 16 / 12 |

`PlanResult` 字段：`num_sets, total_ways, budget, app_ids（按 app id 排序）,
weights, curves, allocation（0=停用）, per_app_misses（停用的应用为 None）,
weighted_misses, used_ways`，v1.1 新增 `miss_caps, deactivation_penalties,
deactivated, deactivation_count, deactivation_cost, objective`；另有
`allocation_map()` 返回 `{app_id: ways}`（含 0），`deactivated_app_ids()`
单独报告被停用的应用。

`PlanVerification` 字段：`ok, failures, replay（活跃应用的 ReplayStats）,
deactivated（被停用应用 id，单独列出）`。

### 目标与优化次序

目标代价 `objective = 活跃应用加权 miss 总和 + 被停用应用的惩罚总和`。
优化次序为字典序：

1. **目标代价**最小；
2. **停用数**最小（代价相同时尽量少停用）；
3. **分配向量**字典序最小（应用按 app id 排序；停用记为 0，排在任何
   正路数之前）。

### 硬上限与停用语义

- `miss_caps` / `deactivation_penalties` 均为可选；整体为 `None` 表示所有
  应用都不设上限 / 都不可停用。否则须为与 `traces` **输入顺序**对齐的序列，
  每项为 `None`（该应用不设上限 / 不可停用）或非负 int。
- 硬上限是对**活跃状态** miss 数的约束（含等于上限即达标）；被停用的应用
  不运行、不产生 miss，上限对其无意义。
- 冷启动下非空轨迹至少 miss 一次，因此上限 0 对活跃应用不可达——它会迫使
  该应用被停用（若已声明可停用）或整体不可行；这是合法输入，不是校验错误。
- 停用惩罚为非负整数（0 合法，表示停用免费，但仍计入停用数）。

### 地址模型

输入为**块地址**序列（块内偏移已剔除）。块 `b` 映射到组 `b % num_sets`，
以完整块号为 tag。每次模拟/重放均为冷启动（空缓存）。只实现 LRU 替换策略，
不宣称兼容任何硬件或行业标准协议。

### 输入上限与约束

- 应用数 1..6，`app_id` 为非空字符串且唯一；
- `num_sets` 1..16，`total_ways` 1..12，`budget` 0..`total_ways`
  （分配总量不可能超过被划分缓存自身的路数）；
  **契约变更（v1.1）**：`budget` 下界由 1 放宽为 0——引入停用后零预算
  是合法输入，仅当所有应用都可停用时可行；`tests/test_validation.py`
  中对应一条旧断言已随之更新，其余旧测试原样保留；
- 块地址为非负 int，轨迹非空；
- 权重为正整数，按 `traces` 的**输入顺序**对齐（库内部再按 app id 排序，
  权重/上限/惩罚随应用走，不会错位）。

## 错误语义

- `ValidationError`（同时是 `ValueError`）：一切输入契约违规。消息中必定
  包含可定位的参数名（如 `weights[1]`、`miss_caps[0]`、`num_sets`）。
  - 整数参数拒绝 `bool`（尽管 `bool` 是 `int` 子类）；
  - 数值输入拒绝 NaN / Infinity；非整数值不做静默截断或纠正。
- `InfeasibleBudgetError`：输入合法，但不存在同时满足全部硬约束（每未声明
  可停用的应用至少一路、各硬上限、总预算）的分配，**可证明无解**。规划器是
  精确、完备的动态规划，不存在"搜索预算耗尽"这一结果，因此两种情形在类型
  上天然分开，绝不混淆。
  - 异常的 `diagnostics` 属性给出诊断：`budget`、`total_ways` 及每应用的
    `min_ways_if_active`（独立满足自身上限所需的最低路数；上限在
    1..total_ways 内不可达时为 `None`）、`deactivatable`、`penalty`。
    **这些最低需求只是必要条件**：它们用于定位瓶颈，不构成一般情形下的
    无解证明；可行性本身由精确完备的 DP 判定。
- 库不持有任何模块级可变状态：调用失败不会留下部分变更，后续调用不受影响。

## 精度口径（浮点结果）

- miss 数、分配路数、停用惩罚、加权目标值均为**精确整数**，无误差。
- 唯一的浮点结果是 `ReplayStats.hit_rate = hits / accesses`，为 IEEE-754
  双精度除法的一次舍入；对本题规模（访问次数为普通整数）相对误差不超过
  2^-52 量级。测试中以 `==` 与同一表达式 `hits / accesses` 比对，不使用
  宽容差掩盖边界。

## 设计取舍

- **精确 DP 而非贪心**：目标是最小化加权 miss 和（含停用惩罚），状态为
  （应用，剩余预算），规模 ≤ 6×12，一次递推即得全局最优。不假定 miss 曲线
  凸或单调，不用边际收益贪心替代最优——`demo.py` 第 2 节给出贪心劣于最优
  159 次 miss 的实际反例。
- **硬上限在 DP 内部执行**：每个应用的可选路数在递推时就被自身上限过滤，
  超出任一硬上限的分配根本不会成为候选。不采用"先求无约束最优再局部修补"
  ——曲线非凸时修补既没有最优性保证也没有可行性保证。
- **停用是 DP 的一个并列选项**：仅对显式声明（惩罚非 `None`）的应用开放，
  代价为声明的整数惩罚、占 0 路；未声明的应用保持至少一路的硬约束。
- **可核验的最终状态**：`PlanResult` 完整保留曲线、上限、惩罚、分配与目标
  分量；`verify_plan` 不信任规划器，逐应用在冷启动隔离缓存上实际重放，
  确认每条硬约束（上限、预算、停用声明、目标算术）后再给出 `ok`。
- **逐路独立模拟**：对每个路数 1..total_ways 分别重放轨迹（每次 O(访问数)），
  实现直白、易于手工核对；本题上限下总开销微不足道（demo 全程约 0.05 秒）。
  未采用单次扫描算全部路数的栈式算法，以换取可验证性。
- **先验证后计算**：所有参数在进入算法前集中校验，失败即抛，无部分结果。

## 已知限制

- 仅 LRU 替换策略、冷启动、块地址粒度；不建模写回、预取、共享/窃取路、
  组间异构或运行时动态重划分；没有时序阶段或业务管理界面。
- 硬上限是每应用独立约束，不建模应用间耦合的服务质量目标。
- 曲线按 `b % num_sets` 取组，不模拟物理地址哈希。
- 上限（6 应用 / 16 组 / 12 路）是接口边界而非性能极限；超出即拒绝。
  测试中的枚举参考实现只用于小规模对照（≤4 应用、≤5 路、40 个确定性
  随机实例），不是库的运行路径。
- 不宣称完整行业标准兼容；仅实现本 README 所述契约。

## 测试与验证

`tests/` 下 47 个用例：

- `test_lru.py`：手工推算的 LRU 曲线（含循环冲突 thrash、组映射隔离、
  LRU≠FIFO 判别）与模拟器输入校验；
- `test_planner.py`：手工推算的加权分配、无收益多余路不分配、字典序并列、
  贪心反例、不可行预算；以及**独立参考实现**（测试文件内自带的朴素
  列表式 LRU + 全枚举暴力搜索，不调用被测核心）在 60 个确定性随机小实例上
  与 DP 逐一比对分配向量与目标值；重放一致性为补充性往返核对；
- `test_constraints.py`（v1.1）：手工推算的硬上限绑定、停用优于保留、
  保留/停用代价平局（停用数决胜）、二选一停用平局（向量字典序决胜）、
  **零预算**、**恰好达标**、上限不可达强制停用或不可行、诊断内容；
  **独立枚举参考**（停用集合 × 全部路数分配，含非凸曲线）在 40 个确定性
  随机小实例上与 DP 比对目标值/停用数/分配向量，不可行实例逐一确认抛出
  `InfeasibleBudgetError`；`verify_plan` 对正常/被篡改结果的确认为与否认；
- `test_validation.py`：NaN/Infinity、bool、类型、边界、可定位错误消息、
  失败后无残留状态，以及 `miss_caps` / `deactivation_penalties` 的契约校验。

## 目录结构

```
wayplan/            # 可导入库（核心算法，纯标准库）
  __init__.py       # 公开接口导出
  core.py           # 校验 + LRU 模拟 + DP 规划 + 隔离重放 + 重放核验
  errors.py         # 错误分类（含不可行诊断）
tests/              # unittest 测试（含独立参考实现）
demo.py             # 固定输入演示：正常规划 / 贪心反例 / 硬上限+停用+核验 / 实际触发的拒绝
README.md
```
