"""Tests for tools module.

TODO:
- Add tests for each tool
- Add tests for registry resolution
- Add tests for version matching
"""

from app.tools.registry import registry


def test_registry_list_tools():
    """Test listing available tools."""
    tools = registry.list_tools()
    assert isinstance(tools, dict)
    assert "parse_docs" in tools
    assert "score_rubrics" in tools


def test_registry_resolve_stub():
    """Test tool resolution (stub)."""
    # This will fail until tools are properly registered
    # tool = registry.resolve("parse_docs", "1.x")
    # assert tool is not None


# TODO: Add tests for each tool
# - test_parse_docs_v1()
# - test_score_rubrics_v1()
# - test_gen_recs_v1()
# - test_rank_usecases_v1()
# - test_write_backlog_v1()
