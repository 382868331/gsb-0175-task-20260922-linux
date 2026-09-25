# wayplan — 可分配缓存路数的性能隔离规划

离线体系结构优化工具库：对最多 6 个应用的合成访存块序列，逐路模拟
set-associative LRU miss 曲线，再用**精确动态规划**在路数预算内选择每应用的
隔离路数，最小化带正整数权重的 miss 总和。v1.1 起支持**每应用可选硬 miss
上限**（超出即不可行，而非罚分）与**显式声明可停用应用**（停用支付给定整数
惩罚、不占路、单独报告），并用独立重放核验最终方案的每条硬约束。

- **Python 版本**：3.14.7（仅使用标准库，语法兼容 3.10+）
- **依赖**：无第三方依赖，无需网络、账号、数据库、Docker；安装 Python 即可运行
- **平台**：Linux 原生（纯 Python，任何平台均可）

## 快速开始

```bash
./.venv/bin/python -m unittest discover -s tests -v   # 运行全部 52 个测试
./.venv/bin/python demo.py                            # 确定性演示（约 0.05 秒，远低于 90 秒上限）
```

```python
from wayplan import plan_allocation, replay_isolated, verify_plan

traces = [("appA", [0, 1] * 4), ("appB", [7] * 5)]
result = plan_allocation(traces, weights=[3, 1],
                         num_sets=1, total_ways=3, budget=3,
                         miss_caps=[None, 1],          # appB 硬上限 1 次 miss
                         disable_penalties=[None, 10]) # appB 可停用，惩罚 10
print(result.curves)           # 逐路 miss 曲线
print(result.allocation_map()) # {'appA': 2, 'appB': 1}（0 表示被停用）
print(result.disabled_app_ids) # 被停用的应用，单独报告
report = verify_plan(traces, result)   # 独立重放确认每条硬约束
assert report.ok
stats = replay_isolated(traces, {"appA": 2, "appB": 1}, num_sets=1)
```

## 接口（固定契约）

| 名称 | 说明 |
|---|---|
| `AppTrace(app_id, blocks)` | 已验证的不可变轨迹；也接受 `(app_id, blocks)` 元组作为输入 |
| `simulate_lru_misses(blocks, num_sets, num_ways) -> int` | 单配置冷启动 LRU miss 数 |
| `miss_curve(blocks, num_sets, max_ways) -> tuple[int, ...]` | 下标 `w-1` 为 `w` 路的 miss 数 |
| `plan_allocation(traces, weights, num_sets, total_ways, budget, miss_caps=None, disable_penalties=None) -> PlanResult` | 精确约束 DP 最优分配 |
| `replay_isolated(traces, allocation, num_sets) -> tuple[ReplayStats, ...]` | 隔离缓存重放统计（只接受 ≥1 路的活跃应用） |
| `verify_plan(traces, plan) -> VerificationReport` | 对方案做独立冷启动重放，逐条确认硬约束 |
| `WayPlanError` / `ValidationError` / `InfeasibleBudgetError` | 错误分类 |
| `MAX_APPS` / `MAX_SETS` / `MAX_WAYS` | 接口上限常量 6 / 16 / 12 |

`PlanResult` 字段：`num_sets, total_ways, budget, app_ids（按 app id 排序）,
weights, curves, allocation（0=停用）, per_app_misses（停用应用为 None）,
weighted_misses, used_ways, miss_caps, disable_penalties, disabled,
disabled_app_ids, disable_penalty_total, objective`，
另有 `allocation_map()` 返回 `{app_id: ways}`。

### 目标与约束语义

- **硬上限**：`miss_caps` 按 `traces` 输入顺序对齐，每项为 `None`（不限）或非负
  整数。活跃应用在所得路数下的 miss 数必须 ≤ 上限；违反任一上限的分配**不可行**，
  不是罚分。上限只约束活跃状态，被停用的应用不产生 miss。
- **停用**：`disable_penalties` 同样按输入顺序对齐，每项为 `None`（不可停用，
  仍至少占一路）或非负整数惩罚。被停用的应用占 0 路、支付惩罚、记入
  `disabled_app_ids` 单独报告。
- **优化顺序**（字典序，一次精确 DP 完成）：先最小化目标代价
  `objective = 加权 miss 总和 + 停用惩罚总和`，再最小化停用数，最后取按
  app id 排序的分配向量（停用为 0）的字典序最小者。
- **不先求无约束最优再局部修补**：miss 曲线不保证凸，无约束最优的邻域修补
  可能严格劣于约束最优（`demo.py` 第 3 节给出实际算例：修补最好 303，
  精确约束解 155）。约束直接编入 DP 状态转移，一遍递推得到全局最优。

### 地址模型

输入为**块地址**序列（块内偏移已剔除）。块 `b` 映射到组 `b % num_sets`，
以完整块号为 tag。每次模拟/重放均为冷启动（空缓存）。只实现 LRU 替换策略，
不宣称兼容任何硬件或行业标准协议。

### 输入上限与约束

- 应用数 1..6，`app_id` 为非空字符串且唯一；
- `num_sets` 1..16，`total_ways` 1..12，`budget` 0..`total_ways`
  （分配总量不可能超过被划分缓存自身的路数）；
  **v1.1 契约变化**：`budget=0` 由输入错误改为合法输入——仅当所有应用
  均可停用时可满足（全部停用，只付惩罚）；否则为可证明不可行
  （`InfeasibleBudgetError`），两类情形在类型上仍然分开；
- 块地址为非负 int，轨迹非空；
- 权重为正整数，`miss_caps` / `disable_penalties` 每项为 `None` 或非负整数，
  三者均按 `traces` 的**输入顺序**对齐（库内部再按 app id 排序，参数随应用走，
  不会错位）。

## 错误语义

- `ValidationError`（同时是 `ValueError`）：一切输入契约违规。消息中必定
  包含可定位的参数名（如 `weights[1]`、`miss_caps[0]`、`num_sets`）。
  - 整数参数拒绝 `bool`（尽管 `bool` 是 `int` 子类）；
  - 数值输入拒绝 NaN / Infinity；非整数值不做静默截断或纠正。
- `InfeasibleBudgetError`：输入合法，但没有任何分配能同时满足全部硬约束
  （预算、未声明停用应用的至少一路、各硬 miss 上限）。这是**可证明无解**：
  规划器是精确、完备的动态规划，不存在“搜索预算耗尽”这一结果，两种情形
  在类型上天然分开，绝不混淆。异常携带诊断属性：
  - `app_ids` / `min_ways`：各应用**独立**最低路数需求（可停用为 0；
    上限在 `total_ways` 内不可达为 `None`）；
  - `required_total`：独立最低需求之和（有任何不可达则为 `None`）；
  - `budget`：本次预算。
  该诊断只是**必要条件**对照，用于定位原因，本身并不构成一般无解的证明；
  无解的权威判定来自精确完备的 DP 本身。
- 库不持有任何模块级可变状态：调用失败不会留下部分变更，后续调用不受影响。

## 方案核验（verify_plan）

`verify_plan(traces, plan)` 用轨迹与方案重新冷启动重放每个活跃应用的隔离
缓存，返回 `VerificationReport`（`checks` 为逐条 `ConstraintCheck`，`ok`
为全部通过）：分配形状、预算、停用规则（只有声明可停用的应用可为 0 路，
其余至少一路）、**每条硬 miss 上限在重放下成立**、重放 miss 与计划曲线一致、
目标值算术（加权 miss + 停用惩罚 = objective）。被停用应用不重放，在
`disabled_app_ids` 中单独报告。参数结构性违约（非 PlanResult、app id 集合
不符）抛 `ValidationError`。

## 精度口径（浮点结果）

- miss 数、分配路数、惩罚、目标值均为**精确整数**，无误差。
- 唯一的浮点结果是 `ReplayStats.hit_rate = hits / accesses`，为 IEEE-754
  双精度除法的一次舍入；对本题规模（访问次数为普通整数）相对误差不超过
  2^-52 量级。测试中以 `==` 与同一表达式 `hits / accesses` 比对，不使用
  宽容差掩盖边界。

## 设计取舍

- **精确约束 DP 而非贪心/修补**：状态为（应用，剩余预算），每应用的候选
  为“停用（若声明）”或“满足硬上限的 1..total_ways 路”，按
  （代价， 停用数， 分配向量） 三元组比较，规模 ≤ 6×12×(12+1)，一次递推即得
  全局最优；不假定 miss 曲线凸或单调，不用边际收益贪心替代最优，也不做
  无约束最优后的局部修补（`demo.py` 第 2、3 节各有一个实际反例）。
- **逐路独立模拟**：对每个路数 1..total_ways 分别重放轨迹（每次 O(访问数)），
  实现直白、易于手工核对；本题上限下总开销微不足道（demo 全程约 0.05 秒）。
  未采用单次扫描算全部路数的栈式算法，以换取可验证性。
- **先验证后计算**：所有参数在进入算法前集中校验，失败即抛，无部分结果。
- **搜索/数值限制**：DP 本身无搜索预算、无超时分支；测试里的暴力枚举参考
  （2^n 停用子集 × total_ways^n 路数组合）只用于 n≤4、total_ways≤5 的小规模
  确定性实例，不是库的运行路径。

## 已知限制

- 仅 LRU 替换策略、冷启动、块地址粒度；不建模写回、预取、共享/窃取路、
  组间异构或运行时动态重划分。
- 曲线按 `b % num_sets` 取组，不模拟物理地址哈希。
- 上限（6 应用 / 16 组 / 12 路）是接口边界而非性能极限；超出即拒绝。
- 硬上限只约束活跃应用；被停用应用的访存行为不模拟（以惩罚代替）。
- 不宣称完整行业标准兼容；仅实现本 README 所述契约。

## 测试与验证

`tests/` 下 52 个用例：

- `test_lru.py`：手工推算的 LRU 曲线（含循环冲突 thrash、组映射隔离、
  LRU≠FIFO 判别）与模拟器输入校验；
- `test_planner.py`：手工推算的加权分配、无收益多余路不分配、字典序并列、
  贪心反例、不可行预算；以及**独立参考实现**（测试文件内自带的朴素
  列表式 LRU + 全枚举暴力搜索，不调用被测核心）在 60 个确定性随机小实例上
  与 DP 逐一比对分配向量与目标值；重放一致性为补充性往返核对；
- `test_caps.py`：硬上限与停用——手工推算的“修补无约束最优必劣于约束最优”
  用例、恰好达标（边界含等号）、不可达上限迫使停用、停用惩罚平局（代价平→
  停用少者优先；停用集平局→分配向量字典序）、零预算（全停用可行 / 含不可
  停用应用则不可行）、诊断属性；**独立暴力枚举**（2^n 停用子集 × 全部路数
  分配，自带朴素 LRU，不调用被测核心）在 50 个确定性随机小实例上与 DP
  逐一比对目标值、分配向量与停用数，覆盖非凸曲线与不可行分支；
  `verify_plan` 的通过与篡改检出（改上限、改目标值、改分配均被重放抓获）；
- `test_validation.py`：NaN/Infinity、bool、类型、边界、可定位错误消息、
  失败后无残留状态（含 `budget=0` 的新契约）。

## 目录结构

```
wayplan/            # 可导入库（核心算法，纯标准库）
  __init__.py       # 公开接口导出
  core.py           # 校验 + LRU 模拟 + 约束 DP 规划 + 隔离重放 + 方案核验
  errors.py         # 错误分类（含不可行诊断属性）
tests/              # unittest 测试（含独立参考实现）
demo.py             # 固定输入演示：正常规划 / 贪心反例 / 硬上限+停用+核验 /
                    # 零预算与不可行诊断 / 实际触发的拒绝
README.md
```
