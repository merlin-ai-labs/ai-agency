# Tool Calling Abstraction Implementation

**Estimated Effort**: 23-33 hours (3-4 working days)
**Priority**: HIGH (blocks multi-LLM tool usage)
**Status**: Not Started

## Overview

The weather agent currently bypasses `BaseAdapter` and directly uses OpenAI client for tool calling (lines 320-373 in `weather_agent.py`). This defeats the multi-LLM architecture. This plan adds tool calling support to the base adapter interface so all 3 providers (OpenAI, Vertex AI, Mistral) can handle tools uniformly.

**Key Files**:
- `app/core/base.py` - Base adapter interface
- `app/core/types.py` - Type definitions
- `app/adapters/llm_openai.py` - OpenAI implementation
- `app/adapters/llm_vertex.py` - Vertex AI implementation
- `app/adapters/llm_mistral.py` - Mistral implementation
- `app/flows/agents/weather_agent.py` - Consumer to refactor

---

## Phase 1: Type Definitions (2-3 hours)

### Step 1.1: Add Tool Type Definitions
**File**: `app/core/types.py`
**Estimated**: 30 minutes
**Dependencies**: None

- [ ] Add `ToolFunction` TypedDict after line 49
  ```python
  class ToolFunction(TypedDict):
      """Type for tool function definition."""
      name: str
      description: str
      parameters: dict[str, Any]  # JSON Schema
  ```

- [ ] Add `ToolDefinition` TypedDict
  ```python
  class ToolDefinition(TypedDict):
      """Type for complete tool definition (OpenAI/Mistral format)."""
      type: str  # "function"
      function: ToolFunction
  ```

- [ ] Add `ToolCall` TypedDict
  ```python
  class ToolCall(TypedDict):
      """Type for tool call from LLM response."""
      id: str
      type: str  # "function"
      function: dict[str, str]  # {"name": str, "arguments": str (JSON)}
  ```

- [ ] Add `ToolMessage` TypedDict
  ```python
  class ToolMessage(TypedDict):
      """Type for tool result message."""
      role: str  # "tool"
      content: str  # JSON string of tool result
      tool_call_id: str
  ```

- [ ] Add `LLMResponseWithTools` TypedDict
  ```python
  class LLMResponseWithTools(TypedDict):
      """Extended LLM response type that includes tool calls."""
      content: str
      model: str
      tokens_used: int
      finish_reason: str
      tool_calls: list[ToolCall] | None
  ```

---

## Phase 2: Update BaseAdapter Interface (1-2 hours)

### Step 2.1: Extend BaseAdapter
**File**: `app/core/base.py`
**Estimated**: 1 hour
**Dependencies**: Phase 1 complete

- [ ] Update `complete()` method signature (line 165)
  - Add `tools: list[dict[str, Any]] | None = None` parameter
  - Add `tool_choice: str | dict[str, Any] = "auto"` parameter
  - Update docstring with new parameters

- [ ] Update `complete_with_metadata()` method signature (line 198)
  - Add `tools: list[dict[str, Any]] | None = None` parameter
  - Add `tool_choice: str | dict[str, Any] = "auto"` parameter
  - Update return type to `LLMResponse | LLMResponseWithTools`
  - Update docstring

- [ ] Update existing tests to pass `tools=None` (backward compatible)

---

## Phase 3: OpenAI Adapter Implementation (3-4 hours)

### Step 3.1: Update OpenAIAdapter.complete()
**File**: `app/adapters/llm_openai.py`
**Estimated**: 1.5 hours
**Dependencies**: Phase 2 complete

- [ ] Update method signature at line 104
  - Add `tools` and `tool_choice` parameters

- [ ] Update API call parameters (around line 150)
  - Add tools to `api_params` if provided
  - Add tool_choice to `api_params` if tools provided

- [ ] Handle empty content when tool call requested
  - Return `response.choices[0].message.content or ""`

### Step 3.2: Update OpenAIAdapter.complete_with_metadata()
**File**: `app/adapters/llm_openai.py`
**Estimated**: 1.5 hours
**Dependencies**: Step 3.1 complete

- [ ] Update method signature at line 189
  - Add `tools` and `tool_choice` parameters
  - Update return type to `LLMResponse | LLMResponseWithTools`

- [ ] Add tool parameters to API call

- [ ] Extract tool calls from response
  ```python
  if message.tool_calls:
      result["tool_calls"] = [
          {
              "id": tc.id,
              "type": tc.type,
              "function": {
                  "name": tc.function.name,
                  "arguments": tc.function.arguments,
              },
          }
          for tc in message.tool_calls
      ]
  else:
      result["tool_calls"] = None
  ```

---

## Phase 4: Vertex AI Adapter Implementation (4-6 hours)

### Step 4.1: Research Vertex AI Tool Format
**Estimated**: 1 hour
**Dependencies**: Phase 3 complete

- [ ] Document Vertex AI tool format differences from OpenAI
- [ ] Identify conversion requirements
- [ ] Plan for tool call ID generation (Vertex doesn't provide IDs)

### Step 4.2: Add Tool Format Converter
**File**: `app/adapters/llm_vertex.py`
**Estimated**: 2 hours
**Dependencies**: Step 4.1 complete

- [ ] Add `_convert_tools_to_vertex_format()` method after line 45
  - Convert OpenAI format to Vertex `FunctionDeclaration`
  - Return `Tool` object
  - Add validation and error handling

- [ ] Add `_extract_tool_calls_from_response()` method
  - Extract function calls from Vertex response
  - Convert to OpenAI format
  - Generate deterministic IDs using hash

### Step 4.3: Update VertexAIAdapter.complete()
**File**: `app/adapters/llm_vertex.py`
**Estimated**: 1.5 hours
**Dependencies**: Step 4.2 complete

- [ ] Update method signature at line 75
  - Add `tools` and `tool_choice` parameters

- [ ] Convert tools to Vertex format
  - Call `_convert_tools_to_vertex_format()` if tools provided

- [ ] Update API call to include tools
  - Pass `tools=[vertex_tools]` parameter

### Step 4.4: Update VertexAIAdapter.complete_with_metadata()
**File**: `app/adapters/llm_vertex.py`
**Estimated**: 1.5 hours
**Dependencies**: Step 4.3 complete

- [ ] Update method signature
  - Add `tools` and `tool_choice` parameters
  - Update return type

- [ ] Extract tool calls from response
  - Call `_extract_tool_calls_from_response()`
  - Add to result dict

---

## Phase 5: Mistral Adapter Implementation (3-4 hours)

### Step 5.1: Update MistralAdapter.complete()
**File**: `app/adapters/llm_mistral.py`
**Estimated**: 1.5 hours
**Dependencies**: Phase 3 complete

- [ ] Update method signature at line 103
  - Add `tools` and `tool_choice` parameters

- [ ] Add tools to request payload
  - Mistral uses OpenAI-compatible format
  - Add `tools` and `tool_choice` to payload if provided

### Step 5.2: Update MistralAdapter.complete_with_metadata()
**File**: `app/adapters/llm_mistral.py`
**Estimated**: 1.5 hours
**Dependencies**: Step 5.1 complete

- [ ] Update method signature
  - Add `tools` and `tool_choice` parameters
  - Update return type

- [ ] Extract tool calls from response
  - Parse `tool_calls` from response JSON
  - Add to result dict

---

## Phase 6: Refactor Weather Agent (2-3 hours)

### Step 6.1: Remove Direct OpenAI Usage
**File**: `app/flows/agents/weather_agent.py`
**Estimated**: 2 hours
**Dependencies**: Phases 3, 4, 5 complete

- [ ] Delete `_call_llm_with_tools()` method (lines 320-373)

- [ ] Update `run()` method (around line 194)
  - Replace direct OpenAI call with adapter interface
  - Use `self.llm.complete_with_metadata()` with tools parameter
  - Update tool call handling to use new response format

- [ ] Remove OpenAI client import (lines 24-25)

- [ ] Test weather agent works with all 3 providers

---

## Phase 7: Comprehensive Testing (6-8 hours)

### Step 7.1: Unit Tests - OpenAI Adapter
**File**: `tests/test_adapters/test_llm_openai.py`
**Estimated**: 2 hours
**Dependencies**: Phase 3 complete

- [ ] Create `TestOpenAIAdapterToolCalling` test class

- [ ] Add test: `test_complete_with_tools_no_call`
  - Mock response without tool calls
  - Verify tools passed to API
  - Assert normal response

- [ ] Add test: `test_complete_with_metadata_tool_call`
  - Mock response with tool call
  - Verify tool_calls in result
  - Assert correct format

- [ ] Add test: `test_complete_without_tools`
  - Test backward compatibility
  - Verify tools NOT passed when None

- [ ] Add test: `test_tool_choice_none`
- [ ] Add test: `test_tool_choice_specific`
- [ ] Add test: `test_multiple_tool_calls`

### Step 7.2: Unit Tests - Vertex AI Adapter
**File**: `tests/test_adapters/test_llm_vertex.py`
**Estimated**: 2.5 hours
**Dependencies**: Phase 4 complete

- [ ] Add test: `test_convert_tools_to_vertex_format`
  - Test OpenAI → Vertex conversion
  - Verify FunctionDeclaration structure

- [ ] Add test: `test_extract_tool_calls_from_response`
  - Test Vertex → OpenAI format conversion
  - Verify ID generation

- [ ] Add test: `test_complete_with_tools`
- [ ] Add test: `test_complete_with_metadata_tool_call`
- [ ] Add test: `test_invalid_tool_definition`

### Step 7.3: Unit Tests - Mistral Adapter
**File**: `tests/test_adapters/test_llm_mistral.py`
**Estimated**: 2 hours
**Dependencies**: Phase 5 complete

- [ ] Add test: `test_complete_with_tools`
- [ ] Add test: `test_complete_with_metadata_tool_call`
- [ ] Add test: `test_tool_call_error_handling`

### Step 7.4: Integration Test - Weather Agent
**File**: `tests/integration/test_weather_agent.py`
**Estimated**: 1.5 hours
**Dependencies**: Phase 6 complete

- [ ] Add test: `test_weather_agent_openai`
- [ ] Add test: `test_weather_agent_vertex`
- [ ] Add test: `test_weather_agent_mistral`
- [ ] Verify all produce consistent results
- [ ] Test conversation flow with tool usage

---

## Phase 8: Documentation (2-3 hours)

### Step 8.1: Update API Documentation
**Estimated**: 1 hour

- [ ] Add tool calling examples to `BaseAdapter` docstring
- [ ] Document tool definition format
- [ ] Add example showing multi-turn tool conversation
- [ ] Update method docstrings with tool parameters

### Step 8.2: Create Migration Guide
**File**: `docs/tool-calling-migration.md`
**Estimated**: 1 hour

- [ ] Document how to use tool calling with adapters
- [ ] Add migration guide for existing direct API usage
- [ ] Include examples of tool definition formats
- [ ] Add troubleshooting section

### Step 8.3: Update Architecture Documentation
**Estimated**: 1 hour

- [ ] Update architecture docs to reflect tool calling support
- [ ] Add diagrams showing tool calling flow
- [ ] Document provider differences

---

## Risk Assessment

### Risk 1: Vertex AI Tool Format Compatibility
**Risk Level**: HIGH
**Impact**: Vertex AI tool calling may not support all OpenAI features
**Mitigation**:
- Research Vertex AI docs thoroughly in Step 4.1
- Create fallback strategy for unsupported features
- Add clear error messages for incompatible tool definitions

### Risk 2: Breaking Changes
**Risk Level**: MEDIUM
**Impact**: Existing code may break if signatures change
**Mitigation**:
- Make tools parameter optional (default None)
- All existing tests must pass without modification
- Add deprecation warnings if needed

### Risk 3: Tool Call ID Consistency
**Risk Level**: MEDIUM
**Impact**: Vertex AI doesn't provide tool call IDs like OpenAI
**Mitigation**:
- Generate deterministic IDs for Vertex (hash-based)
- Document ID format differences
- Ensure IDs are unique within response

---

## Success Criteria

### Must Have
- [ ] All 3 adapters support `tools` parameter in `complete()` and `complete_with_metadata()`
- [ ] Weather agent uses adapter interface instead of direct OpenAI client
- [ ] All existing tests pass without modification
- [ ] New tests cover tool calling scenarios for each provider
- [ ] Tool calls work end-to-end with at least OpenAI and Mistral

### Should Have
- [ ] Vertex AI tool calling fully implemented and tested
- [ ] Comprehensive error handling for invalid tool definitions
- [ ] Documentation covers tool calling patterns
- [ ] Migration guide for other engineers

### Nice to Have
- [ ] Performance benchmarks for tool format conversion
- [ ] Tool definition validation helper
- [ ] Caching for converted Vertex AI tools

---

## Implementation Timeline

| Phase | Duration | Dependencies |
|-------|----------|-------------|
| Phase 1: Types | 2-3 hours | None |
| Phase 2: Base Interface | 1-2 hours | Phase 1 |
| Phase 3: OpenAI | 3-4 hours | Phase 2 |
| Phase 4: Vertex AI | 4-6 hours | Phase 3 |
| Phase 5: Mistral | 3-4 hours | Phase 3 |
| Phase 6: Refactor Agent | 2-3 hours | Phases 3,4,5 |
| Phase 7: Testing | 6-8 hours | Phase 6 |
| Phase 8: Documentation | 2-3 hours | Phase 7 |

**Critical Path**: Phase 1 → Phase 2 → Phase 3 → Phase 6 → Phase 7.1

**Parallel Work**: Phases 4 and 5 can be done simultaneously after Phase 3

**Minimum Viable**: Phases 1, 2, 3, 6 (OpenAI only) - ~10 hours

---

## Follow-up Tasks

After implementation:
- [ ] Add tool calling support to other agents/flows
- [ ] Create reusable tool registry
- [ ] Implement tool result validation
- [ ] Add telemetry for tool usage
- [ ] Monitor tool call success/failure rates
- [ ] Track latency impact of tool format conversion
