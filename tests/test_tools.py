import pytest

from app.tools.calculator import CalculatorTool, UnsafeExpressionError, safe_eval
from app.tools.web_search import WebSearchTool
from app.tools.base import ToolRegistry


def test_safe_eval_basic_arithmetic():
    assert safe_eval("2 + 2") == 4
    assert safe_eval("25 * 18") == 450
    assert safe_eval("10 / 4") == 2.5
    assert safe_eval("-5 + 3") == -2


def test_safe_eval_rejects_function_calls():
    with pytest.raises(UnsafeExpressionError):
        safe_eval("__import__('os').system('ls')")


def test_safe_eval_rejects_names():
    with pytest.raises(UnsafeExpressionError):
        safe_eval("os.system('ls')")


def test_safe_eval_rejects_long_expressions():
    with pytest.raises(UnsafeExpressionError):
        safe_eval("1+" * 500 + "1")


def test_calculator_tool_returns_result():
    tool = CalculatorTool()
    result = tool.execute(expression="12 * 12")
    assert result.success
    assert result.output == "144"


def test_calculator_tool_handles_division_by_zero():
    tool = CalculatorTool()
    result = tool.execute(expression="1 / 0")
    assert not result.success
    assert result.error


def test_web_search_tool_fails_gracefully_without_key():
    tool = WebSearchTool(api_key=None)
    result = tool.execute(query="latest AI news")
    assert not result.success
    assert "unavailable" in result.error.lower()


def test_tool_registry_only_returns_registered_tools():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    assert registry.get("calculator") is not None
    assert registry.get("not_a_real_tool") is None
