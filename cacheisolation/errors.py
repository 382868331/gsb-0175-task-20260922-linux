"""错误类型与错误语义。

所有由本库主动抛出的错误都继承 :class:`CacheIsolationError`，
错误消息以 ``"字段路径: ..."`` 形式给出可定位信息（非法输入），
或在消息中包含约束条件（不可行/预算耗尽）。

区分三类失败：

* :class:`InvalidInputError` —— 输入违反契约，调用方修正后重试；
* :class:`InfeasibleAllocationError` —— 输入合法但**已证明**无可行分配；
* :class:`SearchBudgetExhausted` —— 搜索预算耗尽，**未证明**无解
  （本题精确 DP 的完整表算法在契约上限内不会触发，见 ``planner`` 文档）。
"""


class CacheIsolationError(Exception):
    """本库所有自定义异常的基类。"""


class InvalidInputError(CacheIsolationError, ValueError):
    """非法输入。继承 ``ValueError``，消息可定位到具体字段/索引。"""


class InfeasibleAllocationError(CacheIsolationError):
    """已证明不存在满足约束的分配（每应用至少一路且总数不超预算）。"""


class SearchBudgetExhausted(CacheIsolationError):
    """算法搜索预算耗尽而停止；不代表无解（与“证明无解”严格分开）。"""
