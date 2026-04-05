"""
Docstring 强制检查模块
确保函数在没有提供 description 时必须包含 docstring
"""
import functools
import inspect
from typing import Callable, Any


def require_docstring(func: Callable) -> Callable:
    """
    装饰器：强制要求函数必须有 docstring

    如果函数没有 docstring，会抛出 ValueError

    Args:
        func: 被装饰的函数

    Returns:
        装饰后的函数

    Raises:
        ValueError: 如果函数没有 docstring

    Example:
        >>> @require_docstring
        ... def my_function():
        ...     '''This is a docstring'''
        ...     pass
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # 检查函数是否有 docstring
        if not func.__doc__:
            raise ValueError(
                f"Function '{func.__name__}' must have a docstring. "
                f"Please add a docstring to describe what this function does."
            )
        return func(*args, **kwargs)

    # 在定义时立即检查（可选，根据需要启用）
    # if not func.__doc__:
    #     raise ValueError(f"Function '{func.__name__}' must have a docstring at definition time.")

    return wrapper


def ensure_docstring(description: str = None):
    """
    装饰器工厂：如果函数没有 docstring，使用提供的 description

    Args:
        description: 如果函数没有 docstring，使用此描述

    Returns:
        装饰器函数

    Example:
        >>> @ensure_docstring(description="计算两个数的和")
        ... def add(a, b):
        ...     return a + b
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        # 如果函数没有 docstring 且提供了 description，则自动添加
        if not func.__doc__ and description:
            func.__doc__ = description
            wrapper.__doc__ = description

        return wrapper
    return decorator


# ==================== 使用示例 ====================

@require_docstring
def calculate_area(length: float, width: float) -> float:
    """
    计算矩形的面积

    Args:
        length: 矩形的长度
        width: 矩形的宽度

    Returns:
        矩形的面积
    """
    return length * width


@ensure_docstring(description="计算两个数的和")
def add_numbers(a: int, b: int) -> int:
    # 这个函数没有 docstring，会自动使用 description
    return a + b


# 这会失败 - 没有 docstring
# @require_docstring
# def bad_function():
#     pass


if __name__ == "__main__":
    # 测试正常调用
    print(calculate_area(5.0, 3.0))  # 输出: 15.0
    print(add_numbers(2, 3))         # 输出: 5

    # 查看 docstring
    print("\n--- Docstring 信息 ---")
    print(f"calculate_area docstring: {calculate_area.__doc__}")
    print(f"add_numbers docstring: {add_numbers.__doc__}")
