# 可分配缓存路数的性能隔离规划（cache-isolation planner）

为离线体系结构优化提供一个可复用的纯 Python 库：对最多 6 个应用的合成
访存块序列，先**分别模拟**每个应用在每种路数下的 set-associative LRU
miss 数（逐路 miss 曲线），再用**精确动态规划**选择“每应用至少 1 路、
总数不超过预算”的路数分配，最小化带正整数权重的 miss 总和，并在隔离
分区缓存上重放核对。

- **Python 版本**：Python 3.14.7（使用 3.10+ 通用语法，标准库即可运行）。
- **依赖**：仅 Python 标准库；安装 Python 后**无需安装任何其他依赖**。
- **完全离线**：不访问网络、账号、密钥、硬件、数据库、中间件、云 API 或
  Docker；示例数据由仓库内确定性代码生成，无外部附件或随机种子状态。

## 准确的运行命令

```bash
# 必要测试（独立参考实现 + 手工例 + 边界）
python -m unittest discover -s tests -v

# 固定输入演示（正常结果 + 贪心反例 + 实际触发的异常边界），约 0.1 秒
python demo.py
```

终端命令仅用于以上两项。

## 公开接口与数据结构

```python
from cacheisolation import (
    AppTrace, MissCurve, AllocationPlan, SimResult,
    ReplayAppStats, ReplayReport, IsolationPlanResult,
    simulate_lru, build_miss_curves, optimize_allocation,
    plan_isolation, replay_partition,
    greedy_marginal_allocation,          # 仅作反例对照的基线，非核心算法
    InvalidInputError, InfeasibleAllocationError, SearchBudgetExhausted,
)
```

| 接口 | 作用 |
|---|---|
| `AppTrace(app_id, blocks, weight=1)` | 一个应用：非负整数 ID、块 ID 序列（可空）、正整数权重 |
| `simulate_lru(blocks, sets, ways) -> SimResult` | 单次 set-associative LRU 仿真 |
| `build_miss_curves(apps, sets, max_ways=12) -> tuple[MissCurve,...]` | 每应用 × 每路数（1..max_ways）的 miss 曲线，按 app_id 升序 |
| `optimize_allocation(curves, budget, *, state_budget=None) -> AllocationPlan` | **精确 DP** 最优分配；枚举全部并列最优向量 |
| `replay_partition(apps, allocation, sets) -> ReplayReport` | 在隔离分区缓存上按分配重放并统计 |
| `plan_isolation(apps, sets, budget, *, max_ways=12, state_budget=None)` | 上面三步的端到端流水线，含一致性自检 |

关键返回字段：

- `MissCurve.misses[k-1]`：`k` 路时的 miss 数；`MissCurve.at(k)`。
- `AllocationPlan.allocation`：按 app_id 升序的 `(app_id, ways)`；
  `weighted_total`（=Σ wᵢ·missᵢ，**精确整数**）、`ways_used`、`budget`、
  `ties`（其它并列最优向量）、`tie_count`、`iterations`（DP 实际转移数）、
  `budget_states`（本次允许的转移预算）。
- `ReplayReport.per_app` 为各分片 `ReplayAppStats`；汇总含
  `weighted_miss_total`、`total_ways`。`SimResult.miss_rate` /
  `ReplayAppStats.miss_rate` 是库中**唯一**的浮点输出。

### 最小示例

```python
from cacheisolation import AppTrace, plan_isolation

apps = [
    AppTrace(0, [0, 1, 0, 2, 0] * 4, weight=2),
    AppTrace(1, [8, 9, 8, 9], weight=1),
]
r = plan_isolation(apps, sets=2, budget=4)
print(r.plan.allocation, r.plan.weighted_total)
print(r.replay.weighted_miss_total == r.plan.weighted_total)  # True，精确相等
```

## 输入上限（接口边界）

| 参数 | 范围 |
|---|---|
| 应用数 | 1 ≤ n ≤ 6 |
| 缓存组数 `sets` | 1 ≤ sets ≤ 16 |
| 总路数预算 `budget` / `max_ways` / 单分片 `ways` | 1 ≤ … ≤ 12 |
| 块 ID | 非负整数（无上限以外的地址空间假设） |
| 权重 | 正整数（≥ 1） |
| app_id | 非负整数，同批唯一 |

短演示规模刻意远小于上限；上限是接口硬边界，超限直接报错。

## 模型与错误语义

**缓存模型（本题有限子集，非行业标准完整模型）**：地址 `b` 映射到组
`b % sets`，整地址为 tag；组间独立，每组严格 LRU（命中移到 MRU，组满
淘汰 LRU），冷启动；不建模预取、写策略、包含性/一致性。隔离分区即每个
应用独占 `sets` 组 × `ways_i` 路的私有 LRU，分片间零干扰。

- **数值卫生**：任何浮点入口拒绝 `NaN` 与正负 `Infinity`；所有整数参数
  拒绝 `bool`（`True/False` 虽是 `int` 子类也报错）；权重必须为正整数。
- **可定位错误**：非法输入抛 `InvalidInputError`（`ValueError` 子类），
  消息带字段路径，如 `apps[0].blocks[2]: 块 ID 必须是非负整数，得到 -7`；
  不做静默纠正、隐式截断或类型强转。
- **无部分变更**：库无内部可变状态，所有结果为不可变 dataclass；结构
  校验先于计算，批次调用失败时不存在“改了一半”的对外状态。
- **搜索预算耗尽 ≠ 证明无解**，二者是不同异常：
  - `InfeasibleAllocationError`：完整 DP 跑完无可达终点，或
    `budget < 应用数`（每应用至少 1 路）——**数学上已证明无可行分配**；
  - `SearchBudgetExhausted`：显式传入过小的 `state_budget`（DP 转移计数
    上限）导致搜索**在途中中断**，未证明任何结论，放宽预算可重试。
    默认预算取完整 DP 表转移数上界 `n·(budget+1)·max_ways`，在本题上限
    （n≤6, budget≤12）内必能跑完，正常路径不会触发该异常。

## 设计取舍

1. **精确 DP，不用贪心边际收益，也不假设曲线凸。** 对每个应用枚举其
   全部可行路数，DP 状态为“处理完前 i 个应用、恰好用 t 路”的最小加权
   miss，复杂度 O(n·budget·max_ways)，在 n≤6、budget≤12 下极小。
   通过并列父指针回溯枚举**全部**最优向量。
2. **并列裁决按分配向量字典序。** 应用按 app_id 升序排列，所有最优向量
   按普通整数元组字典序排序，主解取最小者（即尽量不给靠前应用多余路）；
   其余在 `AllocationPlan.ties`。
3. **允许用不满预算。** 只要求 Σways ≤ budget；多余路在平坦（无收益）
   区域不会被强行分配。
4. **核心与基线分离。** `cacheisolation/baselines.py` 的朴素边际贪心
   仅用于演示/测试反例，核心 `planner.py` 不导入、不依赖它。
5. **计数精确、比率派生。** 所有 miss/hit、加权和、最优值均为 Python
   任意精度整数，规划与重放之间是整数恒等断言（无容差）。

## 浮点结果的误差口径

库中唯一的浮点量是 `miss_rate = misses / accesses`（空序列定义为
`0.0`）。它是对两个**已精确实测的整数**做的一次 IEEE-754 binary64
除法，误差即一次双精度舍入（相对误差 ≤ 2⁻⁵³），不涉及迭代累加或
浮点优化。**所有正确性判定都在整数计数/整数加权和上进行（精确相等），
不要用对 miss_rate 的宽容差去掩盖计数层的边界错误。**

## 测试策略（独立参考）

`tests/reference.py` 是**独立**测试预言机：它不导入 `cacheisolation`，
用不同的数据结构（每组普通 `list`，LRU 在头/MRU 在尾）重写 LRU，并用
`itertools.product` 暴力枚举全部路数向量求最优。

- `tests/test_lru.py`：手工逐步 LRU 例（含逐次命中/缺失事件）、循环
  冲突、无收益的多余路、组间独立性，以及多种序列形状与独立 LRU 对照。
- `tests/test_planner.py`：精确 DP 与独立暴力枚举在多组非凸/平台/递减
  曲线、多预算下逐一对照（含全部并列向量与字典序主解）、不可行预算、
  搜索预算中断与不可行的类型区分、输入校验。
- `tests/test_integration.py`：NaN/Inf/bool 拒绝、错误可定位、隔离重放
  与曲线/规划的精确一致、端到端、贪心反例（贪心 122 vs DP 42）。

“规划值 == 重放值”的往返一致只作为补充自检，期望值来自独立参考。

## 已知限制

- 只实现本题契约的直接映射组索引（`b % sets`）与严格 LRU；不支持可配
  哈希/置换策略、伪 LRU、随机替换、写回/写直达、包含性、非包含层级。
- 不宣称兼容任何完整行业标准（如某种具体 CPU 缓存规范）。
- 应用在隔离分片中各自冷启动重放；不建模分片间动态路数迁移或共享时间
  分片（time-multiplexing）。
- 默认无搜索中断；`SearchBudgetExhausted` 仅在显式给定过小
  `state_budget` 时用于教学/测试该失败语义。
- 未提供 CLI 产品、Web 界面或报表看板；入口仅 `demo.py` 与 unittest。
