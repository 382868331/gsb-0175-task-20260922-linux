"""输入校验原语。

契约：

* 数值输入（``float``）拒绝 ``NaN`` 与正负 ``Infinity``；
* 整数参数拒绝 ``bool``（``True``/``False`` 是 ``int`` 子类，但语义非法）；
* 所有错误抛 :class:`~cacheisolation.errors.InvalidInputError`，
  消息前缀为可定位的字段路径，不静默纠正、不做隐式截断。
"""

from __future__ import annotations

import math
from typing import Any

from .errors import InvalidInputError

# 本题接口边界（题面给出的上限）。
MAX_APPS = 6
MAX_SETS = 16
MAX_TOTAL_WAYS = 12


def _fail(path: str, message: str) -> None:
    raise InvalidInputError(f"{path}: {message}")


def reject_nonbool_int(value: Any, path: str) -> int:
    """要求真正的 ``int``（拒绝 ``bool``、浮点、NaN/Inf 无整数形态）。"""
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, f"必须是整数 int，得到 {type(value).__name__} ({value!r})")
    return value


def check_int(value: Any, path: str, *, minimum: int | None = None,
              maximum: int | None = None) -> int:
    """校验整数（拒绝 bool）并可选检查闭区间。"""
    value = reject_nonbool_int(value, path)
    if minimum is not None and value < minimum:
        _fail(path, f"必须 >= {minimum}，得到 {value}")
    if maximum is not None and value > maximum:
        _fail(path, f"必须 <= {maximum}，得到 {value}")
    return value


def check_positive_int(value: Any, path: str, *, maximum: int | None = None) -> int:
    """正整数（>=1，拒绝 bool）。"""
    return check_int(value, path, minimum=1, maximum=maximum)


def check_finite_number(value: Any, path: str) -> float:
    """要求有限实数：int（拒绝 bool）或非 NaN/非 Inf 的 float。"""
    if isinstance(value, bool):
        _fail(path, f"必须是数值，得到 bool ({value!r})")
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        if math.isnan(value):
            _fail(path, "拒绝 NaN")
        if math.isinf(value):
            _fail(path, "拒绝 Infinity（含正负无穷）")
        return value
    _fail(path, f"必须是数值，得到 {type(value).__name__} ({value!r})")


def check_positive_integer_weight(value: Any, path: str) -> int:
    """正整数权重（题面：正整数权重）。"""
    value = reject_nonbool_int(value, path)
    if value < 1:
        _fail(path, f"权重必须是正整数 >= 1，得到 {value}")
    return value


def check_block_id(value: Any, path: str) -> int:
    """访存块 ID：非负整数（拒绝 bool）。"""
    value = reject_nonbool_int(value, path)
    if value < 0:
        _fail(path, f"块 ID 必须是非负整数，得到 {value}")
    return value
