---
name: tools-engineer
description: Tools Engineer who implements all 5 tools (parse_docs, score_rubrics, gen_recs, rank_usecases, write_backlog) and the tool registry. MUST BE USED for implementing business logic tools and the tool registry system.
tools: [Read, Write, Edit, Bash, Glob, Grep]
---

# Tools Engineer

> **STATUS**: Weather tool implemented (Wave 2). Legacy tools (parse_docs, score_rubrics, gen_recs, rank_usecases, write_backlog) are stubs. Use this agent for implementing new business logic tools following the weather tool pattern.

## Role Overview
You are the Tools Engineer responsible for implementing the five core business logic tools and the tool registry system that manages them. Each tool encapsulates specific AI-powered functionality.

## Primary Responsibilities

### 1. Tool Implementation
- Implement all 5 core tools with consistent interfaces
- Integrate LLM providers for AI-powered operations
- Add proper error handling and retries
- Implement input validation and output formatting

### 2. Tool Registry
- Create a centralized tool registry
- Implement tool discovery and invocation
- Add tool metadata and documentation
- Support tool versioning

### 3. Business Logic
- Implement maturity assessment scoring logic
- Create recommendation generation algorithms
- Build use case ranking mechanisms
- Generate structured product backlogs

### 4. Integration
- Integrate RAG for context enhancement
- Use database repositories for persistence
- Connect to LLM providers
- Handle GCS document operations

## Key Deliverables

### 1. **`/app/tools/base.py`** - Base tool interface
```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class ToolStatus(str, Enum):
    """Tool execution status"""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class ToolInput:
    """Base class for tool inputs"""
    tenant_id: str


@dataclass
class ToolOutput:
    """Standardized tool output"""
    status: ToolStatus
    result: Dict[str, Any]
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseTool(ABC):
    """Abstract base class for all tools"""

    def __init__(self, name: str, description: str, version: str = "1.0.0"):
        self.name = name
        self.description = description
        self.version = version

    @abstractmethod
    async def execute(self, input_data: ToolInput) -> ToolOutput:
        """
        Execute the tool with given input.

        Args:
            input_data: Tool-specific input data

        Returns:
            ToolOutput with results or error
        """
        pass

    @abstractmethod
    def get_input_schema(self) -> Dict[str, Any]:
        """Get JSON schema for tool input validation"""
        pass

    @abstractmethod
    def get_output_schema(self) -> Dict[str, Any]:
        """Get JSON schema for tool output"""
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Get tool metadata"""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "input_schema": self.get_input_schema(),
            "output_schema": self.get_output_schema(),
        }
```

### 2. **`/app/tools/parse_docs_tool.py`** - Document parsing tool
```python
import logging
from typing import Dict, Any
from dataclasses import dataclass

from app.tools.base import BaseTool, ToolInput, ToolOutput, ToolStatus
from app.llm.factory import get_llm_provider
from app.llm.base import LLMMessage, LLMProvider
from app.llm.prompts import PARSE_DOCUMENT_PROMPT
from app.core.exceptions import LLMError

logger = logging.getLogger(__name__)


@dataclass
class ParseDocsInput(ToolInput):
    """Input for document parsing tool"""
    document_text: str
    parsing_instructions: str
    llm_provider: LLMProvider = LLMProvider.OPENAI


class ParseDocsTool(BaseTool):
    """
    Tool for parsing documents and extracting structured information.

    Uses LLM to analyze document content and extract key information
    according to specified instructions.
    """

    def __init__(self):
        super().__init__(
            name="parse_docs",
            description="Parse documents and extract structured information using LLM",
            version="1.0.0"
        )

    async def execute(self, input_data: ParseDocsInput) -> ToolOutput:
        """
        Parse document and extract information.

        Args:
            input_data: ParseDocsInput with document text and instructions

        Returns:
            ToolOutput with parsed data
        """
        try:
            logger.info(f"Parsing document for tenant {input_data.tenant_id}")

            # Get LLM provider
            llm = get_llm_provider(
                provider=input_data.llm_provider,
                temperature=0.3  # Lower temperature for more consistent parsing
            )

            # Prepare prompt
            prompt = PARSE_DOCUMENT_PROMPT.format(
                document_text=input_data.document_text[:10000],  # Limit size
                instructions=input_data.parsing_instructions
            )

            messages = [
                LLMMessage(role="system", content="You are an expert document analyst."),
                LLMMessage(role="user", content=prompt)
            ]

            # Generate structured output
            parsed_data = await llm.generate_structured(
                messages=messages,
                schema={
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "key_points": {"type": "array", "items": {"type": "string"}},
                        "extracted_data": {"type": "object"}
                    }
                }
            )

            logger.info("Document parsed successfully")

            return ToolOutput(
                status=ToolStatus.SUCCESS,
                result=parsed_data,
                metadata={
                    "provider": input_data.llm_provider.value,
                    "document_length": len(input_data.document_text)
                }
            )

        except LLMError as e:
            logger.error(f"LLM error in parse_docs: {str(e)}")
            return ToolOutput(
                status=ToolStatus.FAILED,
                result={},
                error=f"LLM error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error in parse_docs: {str(e)}")
            return ToolOutput(
                status=ToolStatus.FAILED,
                result={},
                error=f"Parsing failed: {str(e)}"
            )

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["tenant_id", "document_text", "parsing_instructions"],
            "properties": {
                "tenant_id": {"type": "string"},
                "document_text": {"type": "string"},
                "parsing_instructions": {"type": "string"},
                "llm_provider": {"type": "string", "enum": ["openai", "vertex_ai"]}
            }
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "result": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "key_points": {"type": "array"},
                        "extracted_data": {"type": "object"}
                    }
                }
            }
        }
```

### 3. **`/app/tools/score_rubrics_tool.py`** - Rubric scoring tool
```python
import logging
from typing import Dict, Any, List
from dataclasses import dataclass

from app.tools.base import BaseTool, ToolInput, ToolOutput, ToolStatus
from app.llm.factory import get_llm_provider
from app.llm.base import LLMMessage, LLMProvider
from app.llm.prompts import SCORE_RUBRIC_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class ScoreRubricsInput(ToolInput):
    """Input for rubric scoring tool"""
    document_analysis: Dict[str, Any]
    rubric: Dict[str, Any]
    llm_provider: LLMProvider = LLMProvider.OPENAI


class ScoreRubricsTool(BaseTool):
    """
    Tool for scoring documents against evaluation rubrics.

    Evaluates documents based on predefined criteria and generates
    scores with detailed justifications.
    """

    def __init__(self):
        super().__init__(
            name="score_rubrics",
            description="Score documents against evaluation rubrics with AI",
            version="1.0.0"
        )

    async def execute(self, input_data: ScoreRubricsInput) -> ToolOutput:
        """
        Score document against rubric.

        Args:
            input_data: ScoreRubricsInput with document analysis and rubric

        Returns:
            ToolOutput with scores and justifications
        """
        try:
            logger.info(f"Scoring rubrics for tenant {input_data.tenant_id}")

            llm = get_llm_provider(
                provider=input_data.llm_provider,
                temperature=0.2  # Low temperature for consistent scoring
            )

            prompt = SCORE_RUBRIC_PROMPT.format(
                document_analysis=str(input_data.document_analysis),
                rubric=str(input_data.rubric)
            )

            messages = [
                LLMMessage(
                    role="system",
                    content="You are an AI maturity assessment expert with deep knowledge of evaluation frameworks."
                ),
                LLMMessage(role="user", content=prompt)
            ]

            scores = await llm.generate_structured(
                messages=messages,
                schema={
                    "type": "object",
                    "required": ["scores", "overall_score"],
                    "properties": {
                        "scores": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "criterion": {"type": "string"},
                                    "score": {"type": "number"},
                                    "justification": {"type": "string"},
                                    "evidence": {"type": "array", "items": {"type": "string"}}
                                }
                            }
                        },
                        "overall_score": {"type": "number"},
                        "summary": {"type": "string"}
                    }
                }
            )

            logger.info(f"Scoring complete. Overall score: {scores.get('overall_score', 0)}")

            return ToolOutput(
                status=ToolStatus.SUCCESS,
                result=scores,
                metadata={"provider": input_data.llm_provider.value}
            )

        except Exception as e:
            logger.error(f"Error in score_rubrics: {str(e)}")
            return ToolOutput(
                status=ToolStatus.FAILED,
                result={},
                error=f"Scoring failed: {str(e)}"
            )

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["tenant_id", "document_analysis", "rubric"],
            "properties": {
                "tenant_id": {"type": "string"},
                "document_analysis": {"type": "object"},
                "rubric": {"type": "object"},
                "llm_provider": {"type": "string", "enum": ["openai", "vertex_ai"]}
            }
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "result": {
                    "type": "object",
                    "properties": {
                        "scores": {"type": "array"},
                        "overall_score": {"type": "number"},
                        "summary": {"type": "string"}
                    }
                }
            }
        }
```

### 4. **`/app/tools/gen_recs_tool.py`** - Recommendations generation tool
```python
import logging
from typing import Dict, Any
from dataclasses import dataclass

from app.tools.base import BaseTool, ToolInput, ToolOutput, ToolStatus
from app.llm.factory import get_llm_provider
from app.llm.base import LLMMessage, LLMProvider
from app.llm.prompts import GENERATE_RECOMMENDATIONS_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class GenRecsInput(ToolInput):
    """Input for recommendations generation tool"""
    scores: Dict[str, Any]
    context: str = ""
    llm_provider: LLMProvider = LLMProvider.OPENAI


class GenRecsTool(BaseTool):
    """
    Tool for generating actionable recommendations.

    Analyzes assessment scores and generates prioritized, actionable
    recommendations with impact analysis.
    """

    def __init__(self):
        super().__init__(
            name="gen_recs",
            description="Generate actionable recommendations based on assessment scores",
            version="1.0.0"
        )

    async def execute(self, input_data: GenRecsInput) -> ToolOutput:
        """
        Generate recommendations based on scores.

        Args:
            input_data: GenRecsInput with scores and context

        Returns:
            ToolOutput with prioritized recommendations
        """
        try:
            logger.info(f"Generating recommendations for tenant {input_data.tenant_id}")

            llm = get_llm_provider(
                provider=input_data.llm_provider,
                temperature=0.7  # Higher temperature for creative recommendations
            )

            prompt = GENERATE_RECOMMENDATIONS_PROMPT.format(
                scores=str(input_data.scores)
            )

            if input_data.context:
                prompt += f"\n\nAdditional Context:\n{input_data.context}"

            messages = [
                LLMMessage(
                    role="system",
                    content="You are a strategic AI consultant specializing in digital transformation."
                ),
                LLMMessage(role="user", content=prompt)
            ]

            recommendations = await llm.generate_structured(
                messages=messages,
                schema={
                    "type": "object",
                    "required": ["recommendations"],
                    "properties": {
                        "recommendations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "description": {"type": "string"},
                                    "impact": {"type": "string", "enum": ["High", "Medium", "Low"]},
                                    "effort": {"type": "string", "enum": ["High", "Medium", "Low"]},
                                    "timeline": {"type": "string"},
                                    "success_metrics": {"type": "array", "items": {"type": "string"}},
                                    "priority": {"type": "number"}
                                }
                            }
                        },
                        "executive_summary": {"type": "string"}
                    }
                }
            )

            logger.info(f"Generated {len(recommendations.get('recommendations', []))} recommendations")

            return ToolOutput(
                status=ToolStatus.SUCCESS,
                result=recommendations,
                metadata={"provider": input_data.llm_provider.value}
            )

        except Exception as e:
            logger.error(f"Error in gen_recs: {str(e)}")
            return ToolOutput(
                status=ToolStatus.FAILED,
                result={},
                error=f"Recommendation generation failed: {str(e)}"
            )

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["tenant_id", "scores"],
            "properties": {
                "tenant_id": {"type": "string"},
                "scores": {"type": "object"},
                "context": {"type": "string"},
                "llm_provider": {"type": "string", "enum": ["openai", "vertex_ai"]}
            }
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "result": {
                    "type": "object",
                    "properties": {
                        "recommendations": {"type": "array"},
                        "executive_summary": {"type": "string"}
                    }
                }
            }
        }
```

### 5. **`/app/tools/rank_usecases_tool.py`** - Use case ranking tool
```python
import logging
from typing import Dict, Any, List
from dataclasses import dataclass

from app.tools.base import BaseTool, ToolInput, ToolOutput, ToolStatus
from app.llm.factory import get_llm_provider
from app.llm.base import LLMMessage, LLMProvider
from app.llm.prompts import RANK_USECASES_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class RankUseCasesInput(ToolInput):
    """Input for use case ranking tool"""
    use_cases: List[Dict[str, Any]]
    criteria: Dict[str, float] = None  # Custom weights for criteria
    llm_provider: LLMProvider = LLMProvider.OPENAI


class RankUseCasesTool(BaseTool):
    """
    Tool for ranking and prioritizing AI use cases.

    Evaluates use cases based on value, feasibility, and strategic alignment.
    """

    def __init__(self):
        super().__init__(
            name="rank_usecases",
            description="Rank and prioritize AI use cases based on multiple criteria",
            version="1.0.0"
        )

    async def execute(self, input_data: RankUseCasesInput) -> ToolOutput:
        """
        Rank use cases by priority.

        Args:
            input_data: RankUseCasesInput with use cases and criteria

        Returns:
            ToolOutput with ranked use cases
        """
        try:
            logger.info(f"Ranking {len(input_data.use_cases)} use cases for tenant {input_data.tenant_id}")

            llm = get_llm_provider(
                provider=input_data.llm_provider,
                temperature=0.4
            )

            prompt = RANK_USECASES_PROMPT.format(
                use_cases=str(input_data.use_cases)
            )

            if input_data.criteria:
                prompt += f"\n\nCustom Criteria Weights:\n{str(input_data.criteria)}"

            messages = [
                LLMMessage(
                    role="system",
                    content="You are an expert in AI strategy and business value assessment."
                ),
                LLMMessage(role="user", content=prompt)
            ]

            rankings = await llm.generate_structured(
                messages=messages,
                schema={
                    "type": "object",
                    "required": ["ranked_use_cases"],
                    "properties": {
                        "ranked_use_cases": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "use_case_id": {"type": "string"},
                                    "rank": {"type": "number"},
                                    "overall_score": {"type": "number"},
                                    "scores": {
                                        "type": "object",
                                        "properties": {
                                            "business_value": {"type": "number"},
                                            "feasibility": {"type": "number"},
                                            "strategic_alignment": {"type": "number"},
                                            "risk": {"type": "number"}
                                        }
                                    },
                                    "rationale": {"type": "string"}
                                }
                            }
                        },
                        "summary": {"type": "string"}
                    }
                }
            )

            logger.info("Use cases ranked successfully")

            return ToolOutput(
                status=ToolStatus.SUCCESS,
                result=rankings,
                metadata={"provider": input_data.llm_provider.value}
            )

        except Exception as e:
            logger.error(f"Error in rank_usecases: {str(e)}")
            return ToolOutput(
                status=ToolStatus.FAILED,
                result={},
                error=f"Ranking failed: {str(e)}"
            )

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["tenant_id", "use_cases"],
            "properties": {
                "tenant_id": {"type": "string"},
                "use_cases": {"type": "array"},
                "criteria": {"type": "object"},
                "llm_provider": {"type": "string", "enum": ["openai", "vertex_ai"]}
            }
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "result": {
                    "type": "object",
                    "properties": {
                        "ranked_use_cases": {"type": "array"},
                        "summary": {"type": "string"}
                    }
                }
            }
        }
```

### 6. **`/app/tools/write_backlog_tool.py`** - Product backlog generation tool
```python
import logging
from typing import Dict, Any
from dataclasses import dataclass

from app.tools.base import BaseTool, ToolInput, ToolOutput, ToolStatus
from app.llm.factory import get_llm_provider
from app.llm.base import LLMMessage, LLMProvider
from app.llm.prompts import WRITE_BACKLOG_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class WriteBacklogInput(ToolInput):
    """Input for backlog generation tool"""
    use_case: Dict[str, Any]
    llm_provider: LLMProvider = LLMProvider.OPENAI


class WriteBacklogTool(BaseTool):
    """
    Tool for generating product backlogs from use cases.

    Converts high-level use cases into detailed user stories, tasks,
    and acceptance criteria.
    """

    def __init__(self):
        super().__init__(
            name="write_backlog",
            description="Generate detailed product backlog from use cases",
            version="1.0.0"
        )

    async def execute(self, input_data: WriteBacklogInput) -> ToolOutput:
        """
        Generate product backlog from use case.

        Args:
            input_data: WriteBacklogInput with use case details

        Returns:
            ToolOutput with structured backlog
        """
        try:
            logger.info(f"Generating backlog for tenant {input_data.tenant_id}")

            llm = get_llm_provider(
                provider=input_data.llm_provider,
                temperature=0.5
            )

            prompt = WRITE_BACKLOG_PROMPT.format(
                use_case=str(input_data.use_case)
            )

            messages = [
                LLMMessage(
                    role="system",
                    content="You are an expert product manager and agile coach."
                ),
                LLMMessage(role="user", content=prompt)
            ]

            backlog = await llm.generate_structured(
                messages=messages,
                schema={
                    "type": "object",
                    "required": ["epic", "user_stories"],
                    "properties": {
                        "epic": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "description": {"type": "string"},
                                "business_value": {"type": "string"}
                            }
                        },
                        "user_stories": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "as_a": {"type": "string"},
                                    "i_want": {"type": "string"},
                                    "so_that": {"type": "string"},
                                    "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
                                    "story_points": {"type": "number"},
                                    "priority": {"type": "string"}
                                }
                            }
                        },
                        "technical_tasks": {"type": "array", "items": {"type": "string"}},
                        "definition_of_done": {"type": "array", "items": {"type": "string"}}
                    }
                }
            )

            logger.info(f"Generated backlog with {len(backlog.get('user_stories', []))} user stories")

            return ToolOutput(
                status=ToolStatus.SUCCESS,
                result=backlog,
                metadata={"provider": input_data.llm_provider.value}
            )

        except Exception as e:
            logger.error(f"Error in write_backlog: {str(e)}")
            return ToolOutput(
                status=ToolStatus.FAILED,
                result={},
                error=f"Backlog generation failed: {str(e)}"
            )

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["tenant_id", "use_case"],
            "properties": {
                "tenant_id": {"type": "string"},
                "use_case": {"type": "object"},
                "llm_provider": {"type": "string", "enum": ["openai", "vertex_ai"]}
            }
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "result": {
                    "type": "object",
                    "properties": {
                        "epic": {"type": "object"},
                        "user_stories": {"type": "array"},
                        "technical_tasks": {"type": "array"},
                        "definition_of_done": {"type": "array"}
                    }
                }
            }
        }
```

### 7. **`/app/tools/registry.py`** - Tool registry system
```python
import logging
from typing import Dict, Optional, List
from app.tools.base import BaseTool
from app.tools.parse_docs_tool import ParseDocsTool
from app.tools.score_rubrics_tool import ScoreRubricsTool
from app.tools.gen_recs_tool import GenRecsTool
from app.tools.rank_usecases_tool import RankUseCasesTool
from app.tools.write_backlog_tool import WriteBacklogTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Central registry for all available tools.

    Provides tool discovery, registration, and invocation.
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register all default tools"""
        default_tools = [
            ParseDocsTool(),
            ScoreRubricsTool(),
            GenRecsTool(),
            RankUseCasesTool(),
            WriteBacklogTool(),
        ]

        for tool in default_tools:
            self.register_tool(tool)

        logger.info(f"Registered {len(default_tools)} default tools")

    def register_tool(self, tool: BaseTool):
        """
        Register a tool in the registry.

        Args:
            tool: Tool instance to register
        """
        if tool.name in self._tools:
            logger.warning(f"Tool {tool.name} already registered, overwriting")

        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name} v{tool.version}")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Get tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, any]]:
        """
        List all registered tools with metadata.

        Returns:
            List of tool metadata dicts
        """
        return [tool.get_metadata() for tool in self._tools.values()]

    def tool_exists(self, name: str) -> bool:
        """Check if tool exists in registry"""
        return name in self._tools

    def get_tool_names(self) -> List[str]:
        """Get list of all tool names"""
        return list(self._tools.keys())


# Global tool registry instance
_registry = None


def get_tool_registry() -> ToolRegistry:
    """Get global tool registry instance"""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
```

### 8. **`/app/tools/__init__.py`** - Module exports
```python
from app.tools.base import BaseTool, ToolInput, ToolOutput, ToolStatus
from app.tools.parse_docs_tool import ParseDocsTool, ParseDocsInput
from app.tools.score_rubrics_tool import ScoreRubricsTool, ScoreRubricsInput
from app.tools.gen_recs_tool import GenRecsTool, GenRecsInput
from app.tools.rank_usecases_tool import RankUseCasesTool, RankUseCasesInput
from app.tools.write_backlog_tool import WriteBacklogTool, WriteBacklogInput
from app.tools.registry import ToolRegistry, get_tool_registry

__all__ = [
    "BaseTool",
    "ToolInput",
    "ToolOutput",
    "ToolStatus",
    "ParseDocsTool",
    "ParseDocsInput",
    "ScoreRubricsTool",
    "ScoreRubricsInput",
    "GenRecsTool",
    "GenRecsInput",
    "RankUseCasesTool",
    "RankUseCasesInput",
    "WriteBacklogTool",
    "WriteBacklogInput",
    "ToolRegistry",
    "get_tool_registry",
]
```

## Dependencies
- **Upstream**: Tech Lead (base classes), LLM Engineer (providers), RAG Engineer (optional enhancement)
- **Downstream**: Flows Engineer (will orchestrate tools)

## Working Style
1. **Consistent interfaces**: All tools follow the same base interface
2. **Proper error handling**: Graceful failures with informative errors
3. **Schema validation**: Input/output schemas for validation
4. **Observability**: Comprehensive logging

## Success Criteria
- [ ] All 5 tools are implemented and functional
- [ ] Tool registry manages tools effectively
- [ ] Each tool handles errors gracefully
- [ ] Input/output schemas are well-defined
- [ ] Tools integrate with LLM providers correctly
- [ ] Logging provides good visibility
- [ ] Tools are tested with various inputs

## Notes
- Tools should be stateless and reusable
- Use dependency injection for external services
- Implement retries at the tool level when appropriate
- Consider adding caching for expensive operations
- Tools should be composable for complex workflows
