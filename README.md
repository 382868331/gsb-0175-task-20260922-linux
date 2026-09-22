# wayplan — 可分配缓存路数的性能隔离规划

离线体系结构优化工具库：对最多 6 个应用的合成访存块序列，逐路模拟
set-associative LRU miss 曲线，再用**精确动态规划**在路数预算内选择每应用的
隔离路数，最小化带正整数权重的 miss 总和，并在隔离缓存上重放统计。

- **Python 版本**：3.14.7（仅使用标准库，语法兼容 3.10+）
- **依赖**：无第三方依赖，无需网络、账号、数据库、Docker；安装 Python 即可运行
- **平台**：Linux 原生（纯 Python，任何平台均可）

## 快速开始

```bash
python -m unittest discover -s tests -v   # 运行全部测试
python demo.py                            # 确定性演示（约 0.1 秒，远低于 8 秒上限）
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

## 接口（固定契约）

| 名称 | 说明 |
|---|---|
| `AppTrace(app_id, blocks)` | 已验证的不可变轨迹；也接受 `(app_id, blocks)` 元组作为输入 |
| `simulate_lru_misses(blocks, num_sets, num_ways) -> int` | 单配置冷启动 LRU miss 数 |
| `miss_curve(blocks, num_sets, max_ways) -> tuple[int, ...]` | 下标 `w-1` 为 `w` 路的 miss 数 |
| `plan_allocation(traces, weights, num_sets, total_ways, budget) -> PlanResult` | 精确 DP 最优分配 |
| `replay_isolated(traces, allocation, num_sets) -> tuple[ReplayStats, ...]` | 隔离缓存重放统计 |
| `WayPlanError` / `ValidationError` / `InfeasibleBudgetError` | 错误分类 |
| `MAX_APPS` / `MAX_SETS` / `MAX_WAYS` | 接口上限常量 6 / 16 / 12 |

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

## 错误语义

- `ValidationError`（同时是 `ValueError`）：一切输入契约违规。消息中必定
  包含可定位的参数名（如 `weights[1]`、`num_sets`、`blocks of app 'a'`）。
  - 整数参数拒绝 `bool`（尽管 `bool` 是 `int` 子类）；
  - 数值输入拒绝 NaN / Infinity；非整数值不做静默截断或纠正。
- `InfeasibleBudgetError`：输入合法，但 `budget < 应用数`，**可证明无解**
  （每应用至少一路）。规划器是精确、完备的动态规划，不存在“搜索预算耗尽”
  这一结果，因此两种情形在类型上天然分开，绝不混淆。
- 库不持有任何模块级可变状态：调用失败不会留下部分变更，后续调用不受影响。

## 精度口径（浮点结果）

- miss 数、分配路数、加权目标值均为**精确整数**，无误差。
- 唯一的浮点结果是 `ReplayStats.hit_rate = hits / accesses`，为 IEEE-754
  双精度除法的一次舍入；对本题规模（访问次数为普通整数）相对误差不超过
  2^-52 量级。测试中以 `==` 与同一表达式 `hits / accesses` 比对，不使用
  宽容差掩盖边界。

## 设计取舍

- **精确 DP 而非贪心**：目标是最小化加权 miss 和，状态为（应用，剩余预算），
  规模 ≤ 6×12，一次递推即得全局最优；并列时按 app id 排序的分配向量取
  字典序最小（元组比较天然完成）。不假定 miss 曲线凸或单调，不用边际收益
  贪心替代最优——`demo.py` 第 2 节给出贪心劣于最优 159 次 miss 的实际反例。
- **逐路独立模拟**：对每个路数 1..total_ways 分别重放轨迹（每次 O(访问数)），
  实现直白、易于手工核对；本题上限下总开销微不足道（demo 全程 < 0.1 秒）。
  未采用单次扫描算全部路数的栈式算法，以换取可验证性。
- **先验证后计算**：所有参数在进入算法前集中校验，失败即抛，无部分结果。

## 已知限制

- 仅 LRU 替换策略、冷启动、块地址粒度；不建模写回、预取、共享/窃取路、
  组间异构或运行时动态重划分。
- 曲线按 `b % num_sets` 取组，不模拟物理地址哈希。
- 上限（6 应用 / 16 组 / 12 路）是接口边界而非性能极限；超出即拒绝。
- 不宣称完整行业标准兼容；仅实现本 README 所述契约。

## 测试与验证

`tests/` 下 30 个用例：

- `test_lru.py`：手工推算的 LRU 曲线（含循环冲突 thrash、组映射隔离、
  LRU≠FIFO 判别）与模拟器输入校验；
- `test_planner.py`：手工推算的加权分配、无收益多余路不分配、字典序并列、
  贪心反例、不可行预算；以及**独立参考实现**（测试文件内自带的朴素
  列表式 LRU + 全枚举暴力搜索，不调用被测核心）在 60 个确定性随机小实例上
  与 DP 逐一比对分配向量与目标值；重放一致性为补充性往返核对；
- `test_validation.py`：NaN/Infinity、bool、类型、边界、可定位错误消息、
  失败后无残留状态。

## 目录结构

```
wayplan/            # 可导入库（核心算法，纯标准库）
  __init__.py       # 公开接口导出
  core.py           # 校验 + LRU 模拟 + DP 规划 + 隔离重放
  errors.py         # 错误分类
tests/              # unittest 测试（含独立参考实现）
demo.py             # 固定输入演示：正常规划 / 贪心反例 / 实际触发的拒绝
README.md
```
