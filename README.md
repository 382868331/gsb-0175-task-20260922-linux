# wayplan — 可分配缓存路数的性能隔离规划

离线体系结构优化工具库，提供两套**精确动态规划**能力：

1. **单阶段多应用路数分配**：对最多 6 个应用的合成访存块序列，逐路模拟
   set-associative LRU miss 曲线，在路数预算内选择每应用的隔离路数，
   最小化带正整数权重的 miss 总和，并在隔离缓存上重放统计。
2. **多阶段任务改路规划**（本次迭代新增）：对最多 3 个任务、每个任务最多
   4 个已知阶段、最多 8 路的离散性能（损失）曲线，在满足每阶段最小路数和
   最小性能要求的前提下，规划逐阶段路数，最小化"累计损失 + 阶段间迁移
   成本"；相邻阶段改路还受最大变化量限制。

- **Python 版本**：3.14.7（仅使用标准库，语法兼容 3.10+）
- **依赖**：无第三方依赖，无需网络、账号、数据库、Docker；安装 Python 即可运行
- **平台**：Linux 原生（纯 Python，任何平台均可）

## 快速开始

```bash
python -m unittest discover -s tests -v   # 运行全部测试（56 个）
python demo.py                            # 确定性演示（约 0.1 秒，远低于 90 秒上限）
```

```python
from wayplan import plan_allocation, replay_isolated

traces = [("appA", [0, 1] * 4), ("appB", [7] * 5)]
result = plan_allocation(traces, weights=[3, 1],
                         num_sets=1, total_ways=3, budget=3)
print(result.curves)           # 逐路 miss 曲线
print(result.allocation_map()) # {'appA': 2, 'appB': 1}
stats = replay_isolated(traces, result.allocation_map(), num_sets=1)
```

```python
from wayplan import Stage, Task, plan_task_stages

# 每个阶段: Stage(最小路数, 可接受最大损失 max_loss, 1..num_ways 路的离散损失曲线)
task = Task("svc", [
    Stage(1, 60, [80, 40, 10, 5]),   # 阶段0: 路数越多损失越低
    Stage(1, 30, [90, 8, 30, 70]),   # 阶段1: 2 路最优
    Stage(1, 30, [90, 8, 30, 70]),   # 阶段2: 2 路最优
    Stage(1, 60, [80, 40, 10, 5]),   # 阶段3: 路数越多损失越低
])
r = plan_task_stages([task], num_ways=4, migration_cost=6, max_change=1)
p = r.plans[0]
print(p.ways, p.total_cost)         # (3, 2, 2, 3) 48
```

## 接口（固定契约）

| 名称 | 说明 |
|---|---|
| `AppTrace(app_id, blocks)` | 已验证的不可变轨迹；也接受 `(app_id, blocks)` 元组作为输入 |
| `simulate_lru_misses(blocks, num_sets, num_ways) -> int` | 单配置冷启动 LRU miss 数 |
| `miss_curve(blocks, num_sets, max_ways) -> tuple[int, ...]` | 下标 `w-1` 为 `w` 路的 miss 数 |
| `plan_allocation(traces, weights, num_sets, total_ways, budget) -> PlanResult` | 精确 DP 最优分配 |
| `replay_isolated(traces, allocation, num_sets) -> tuple[ReplayStats, ...]` | 隔离缓存重放统计 |
| `Stage(min_ways, max_loss, loss_curve)` | 一个阶段：最小路数、性能上限（损失口径）、离散损失曲线 |
| `Task(task_id, stages, migration_cost=None, max_change=None)` | 一个任务的已知阶段序列；可选任务级迁移参数覆盖 |
| `plan_task_stages(tasks, num_ways, migration_cost=0, max_change=None) -> MultiStagePlanResult` | 多阶段精确 DP 改路规划 |
| `WayPlanError` / `ValidationError` / `InfeasibleBudgetError` / `InfeasibleStageError` | 错误分类 |
| `MAX_APPS` / `MAX_SETS` / `MAX_WAYS` | 旧接口上限常量 6 / 16 / 12 |
| `MAX_TASKS` / `MAX_STAGES` / `MAX_TASK_WAYS` | 多阶段接口上限常量 3 / 4 / 8 |

`PlanResult` 字段：`num_sets, total_ways, budget, app_ids（按 app id 排序）,
weights, curves, allocation, per_app_misses, weighted_misses, used_ways`，
另有 `allocation_map()` 返回 `{app_id: ways}`。

### 地址模型

输入为**块地址**序列（块内偏移已剔除）。块 `b` 映射到组 `b % num_sets`，
以完整块号为 tag。每次模拟/重放均为冷启动（空缓存）。只实现 LRU 替换策略，
不宣称兼容任何硬件或行业标准协议。

### 输入上限与约束

- 应用数 1..6，`app_id` 为非空字符串且唯一；
- `num_sets` 1..16，`total_ways` 1..12，`budget` 1..`total_ways`
  （分配总量不可能超过被划分缓存自身的路数）；
- 块地址为非负 int，轨迹非空；
- 权重为正整数，按 `traces` 的**输入顺序**对齐（库内部再按 app id 排序，
  权重随应用走，不会错位）。

### 多阶段改路规划（`plan_task_stages`）

- **输入规模**：任务数 1..3（`MAX_TASKS`），每任务阶段数 1..4
  （`MAX_STAGES`），路数 1..8（`MAX_TASK_WAYS`）。
- **阶段定义**：`Stage(min_ways, max_loss, loss_curve)`。
  - `loss_curve` 是长度恰为 `num_ways` 的非负整数序列，下标 `w-1` 为
    该阶段以 `w` 路运行时的离散损失（如 miss、延迟损失等，口径由调用方
    给定；本库不模拟曲线，只消费它）。
  - `min_ways` 是该阶段必须满足的最小路数。
  - `max_loss` 是该阶段的**最小性能要求**（以损失口径给出）：当且仅当
    `loss_curve[w-1] <= max_loss` 时 `w` 路在该阶段可接受。性能"至少
    达到下限"与损失"至多为对应上限"完全等价；损失等于上限也算合格。
- **迁移成本**：`migration_cost` 为非负整数；相邻阶段**路数发生变化**时
  在该边界收一次固定费，路数不变则不收费。阶段 0 无入边，费用为 0。
- **最大变化量**：`max_change` 为 1..`num_ways` 的整数（默认
  `num_ways`，即不限），相邻阶段必须满足 `|w_next - w_prev| <= max_change`。
- **任务级覆盖**：`Task(task_id, stages, migration_cost=..., max_change=...)`
  可只为该任务指定参数；为 `None`（默认）时继承 `plan_task_stages` 的值。
- **目标**：每任务独立最小化
  `Σ 阶段损失 + Σ 改路迁移成本`；任务之间不共享预算，
  `MultiStagePlanResult.total_cost` 为各任务总成本之和。
- **并列**：总成本相同的路径中，取逐阶段路数向量字典序最小者（任务按
  task id 排序后报告，与旧规划器约定一致）。
- `TaskPlan` 字段：`task_id, ways（逐阶段路数）, stage_losses,
  migration_costs（入阶段费用，首项恒为 0）, migration_cost, max_change,
  total_loss, total_migration_cost, total_cost`；另有 `stages()` 返回
  逐阶段 `(ways, loss, migration_paid)` 行。
- `MultiStagePlanResult` 字段：`num_ways, default_migration_cost,
  default_max_change, task_ids（按 id 排序）, plans, total_cost`，
  另有 `plan_map()` 返回 `{task_id: TaskPlan}`。

## 错误语义

- `ValidationError`（同时是 `ValueError`）：一切输入契约违规。消息中必定
  包含可定位的参数名（如 `weights[1]`、`num_sets`、`blocks of app 'a'`）。
  - 整数参数拒绝 `bool`（尽管 `bool` 是 `int` 子类）；
  - 数值输入拒绝 NaN / Infinity；非整数值不做静默截断或纠正。
- `InfeasibleBudgetError`：输入合法，但 `budget < 应用数`，**可证明无解**
  （每应用至少一路）。规划器是精确、完备的动态规划，不存在“搜索预算耗尽”
  这一结果，因此两种情形在类型上天然分开，绝不混淆。
- `InfeasibleStageError`：多阶段规划器的输入合法，但约束**可证明无解**——
  某阶段在 `min_ways..num_ways` 内没有任何路数满足 `loss <= max_loss`，
  或某阶段的可接受路数都无法在 `max_change` 内从上一阶段到达。它同样来自
  精确、完备的 DP，不是超时，也不是非法输入。
- 库不持有任何模块级可变状态：调用失败不会留下部分变更，后续调用不受影响。
  多任务规划在任何 DP 运行前先集中校验全部任务，失败即整体拒绝。

## 精度口径（浮点结果）

- miss 数、分配路数、加权目标值均为**精确整数**，无误差。
- 多阶段规划的损失、迁移成本、总成本同样全部为非负整数运算，结果精确。
- 唯一的浮点结果是 `ReplayStats.hit_rate = hits / accesses`，为 IEEE-754
  双精度除法的一次舍入；对本题规模（访问次数为普通整数）相对误差不超过
  2^-52 量级。测试中以 `==` 与同一表达式 `hits / accesses` 比对，不使用
  宽容差掩盖边界。

## 设计取舍

- **精确 DP 而非贪心**：单阶段分配的状态为（应用，剩余预算），规模 ≤ 6×12；
  多阶段改路的状态为（阶段，当前路数），规模 ≤ 4×8、迁移边 ≤ 8²。两者都是
  一次递推即得全局最优，并列时以元组比较天然取得字典序最小解。不假定曲线
  凸或单调，不用边际收益贪心替代最优——`demo.py` 第 2 节给出贪心劣于最优
  159 次 miss 的实际反例；第 4 节给出迁移费改变最优路径、以及约束不可达的
  实际例子。
- **逐路独立模拟**：对每个路数 1..total_ways 分别重放轨迹（每次 O(访问数)），
  实现直白、易于手工核对；本题上限下总开销微不足道（demo 全程 < 0.1 秒）。
  未采用单次扫描算全部路数的栈式算法，以换取可验证性。多阶段曲线由调用方
  直接给定（题面为已知离散性能曲线），库不负责生成。
- **先验证后计算**：所有参数在进入算法前集中校验，失败即抛，无部分结果。
- **任务间不共享预算**：题面约束以任务为单位，最优解可逐任务独立求得，
  库只负责一起报告与求和；不引入跨任务路数争抢这一未要求的语义。
- **固定迁移费而非按差值计费**：题面的迁移成本是"阶段间改路数"的代价，
  因此路数变化即收一次、与变化幅度无关；幅度另由 `max_change` 约束。

## 已知限制

- 仅 LRU 替换策略、冷启动、块地址粒度；不建模写回、预取、共享/窃取路、
  组间异构或运行时动态重划分。
- 曲线按 `b % num_sets` 取组，不模拟物理地址哈希。
- 旧接口上限（6 应用 / 16 组 / 12 路）与多阶段上限（3 任务 / 4 阶段 /
  8 路）都是接口边界而非性能极限；超出即拒绝。
- 多阶段模型只覆盖题面限定范围：阶段序列与曲线均**已知且确定**，迁移费
  固定、`max_change` 对称。不包含不确定场景树、非预知（在线）策略、鲁棒
  下界证书，也没有跨任务共享预算或按变化量计费的迁移成本。
- 不宣称完整行业标准兼容；仅实现本 README 所述契约。

## 测试与验证

`tests/` 下 56 个用例（旧 30 个全部保留未改断言）：

- `test_lru.py`：手工推算的 LRU 曲线（含循环冲突 thrash、组映射隔离、
  LRU≠FIFO 判别）与模拟器输入校验；
- `test_planner.py`：手工推算的加权分配、无收益多余路不分配、字典序并列、
  贪心反例、不可行预算；以及**独立参考实现**（测试文件内自带的朴素
  列表式 LRU + 全枚举暴力搜索，不调用被测核心）在 60 个确定性随机小实例上
  与 DP 逐一比对分配向量与目标值；重放一致性为补充性往返核对；
- `test_stages.py`（新增 26 个）：手算的迁移费改变最优路径、`max_change`
  阻断路径、最小路数与性能上限同时生效、等于上限即合格、保持路数省费、
  字典序并列、任务级覆盖与 id 排序、元组简写、3×4×8 规模边界、单路边界；
  两类可证明不可行（阶段无合格路数、跨边界不可达）；非法类型/范围/曲线
  长度/不可定位消息/不可变性；以及**独立全枚举参考**在 80 个确定性随机
  实例上（含可行与不可行两类）与 DP 逐一比对路径、损失与迁移费拆分；
- `test_validation.py`：NaN/Infinity、bool、类型、边界、可定位错误消息、
  失败后无残留状态。

## 目录结构

```
wayplan/            # 可导入库（核心算法，纯标准库）
  __init__.py       # 公开接口导出
  core.py           # 校验 + LRU 模拟 + 单阶段 DP 分配 + 隔离重放
  stages.py         # 多阶段输入结构 + 跨阶段 DP 改路规划
  errors.py         # 错误分类
tests/              # unittest 测试（含独立参考实现）
demo.py             # 固定输入演示：正常规划 / 贪心反例 / 拒绝 / 多阶段规划
README.md
```
