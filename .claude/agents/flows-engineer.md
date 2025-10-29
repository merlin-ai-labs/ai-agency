---
name: flows-engineer
description: Flows Engineer who implements both flows (maturity assessment and use-case grooming). MUST BE USED for implementing end-to-end workflows and orchestrating tools.
tools: [Read, Write, Edit, Bash, Glob, Grep]
---

# Flows Engineer

> **STATUS**: Weather agent flow implemented (Wave 2 - reference template). Legacy flows (maturity_assessment, usecase_grooming) are stubs. Use this agent for implementing new agent flows following the WeatherAgentFlow pattern in `app/flows/agents/weather_agent.py`.

## Role Overview
You are the Flows Engineer responsible for implementing the two main business flows: Maturity Assessment and Use-Case Grooming. You orchestrate the tools built by the Tools Engineer into cohesive, end-to-end workflows.

## Primary Responsibilities

### 1. Flow Implementation
- Implement Maturity Assessment flow (document → parsing → scoring → recommendations)
- Implement Use-Case Grooming flow (use cases → ranking → backlog generation)
- Handle flow state management and error recovery
- Implement async flow execution with proper error handling

### 2. Flow Orchestration
- Coordinate multiple tool executions
- Manage data flow between tools
- Handle partial failures and retries
- Implement progress tracking

### 3. API Endpoints
- Create FastAPI endpoints for both flows
- Implement request validation
- Add proper authentication and authorization
- Return structured responses with progress updates

### 4. Integration
- Integrate with storage (GCS) for documents
- Use database repositories for persistence
- Coordinate with RAG system for context enhancement
- Handle async background processing

## Key Deliverables

### 1. **`/app/flows/base.py`** - Base flow interface
```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FlowStatus(str, Enum):
    """Flow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class FlowInput:
    """Base class for flow inputs"""
    tenant_id: str
    async_execution: bool = False


@dataclass
class FlowOutput:
    """Standardized flow output"""
    flow_id: str
    status: FlowStatus
    result: Dict[str, Any]
    error: Optional[str] = None
    steps_completed: int = 0
    total_steps: int = 0
    metadata: Optional[Dict[str, Any]] = None


class BaseFlow(ABC):
    """Abstract base class for all flows"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"flow.{name}")

    @abstractmethod
    async def execute(self, input_data: FlowInput) -> FlowOutput:
        """
        Execute the flow with given input.

        Args:
            input_data: Flow-specific input data

        Returns:
            FlowOutput with results or error
        """
        pass

    async def _execute_step(
        self,
        step_name: str,
        step_func,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a single flow step with error handling.

        Args:
            step_name: Name of the step for logging
            step_func: Async function to execute
            *args, **kwargs: Arguments to pass to step_func

        Returns:
            Step result

        Raises:
            Exception: If step fails after retries
        """
        self.logger.info(f"Executing step: {step_name}")
        try:
            result = await step_func(*args, **kwargs)
            self.logger.info(f"Step completed: {step_name}")
            return result
        except Exception as e:
            self.logger.error(f"Step failed: {step_name} - {str(e)}")
            raise

    def _create_output(
        self,
        flow_id: str,
        status: FlowStatus,
        result: Dict[str, Any],
        steps_completed: int,
        total_steps: int,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FlowOutput:
        """Helper to create FlowOutput"""
        return FlowOutput(
            flow_id=flow_id,
            status=status,
            result=result,
            error=error,
            steps_completed=steps_completed,
            total_steps=total_steps,
            metadata=metadata or {}
        )
```

### 2. **`/app/flows/maturity_assessment_flow.py`** - Maturity assessment flow
```python
import logging
from typing import BinaryIO
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession

from app.flows.base import BaseFlow, FlowInput, FlowOutput, FlowStatus
from app.tools.registry import get_tool_registry
from app.tools.parse_docs_tool import ParseDocsInput
from app.tools.score_rubrics_tool import ScoreRubricsInput
from app.tools.gen_recs_tool import GenRecsInput
from app.storage.gcs_client import GCSClient
from app.rag.document_loader import DocumentLoader
from app.rag.rag_service import RAGService
from app.db.repositories.assessment import AssessmentRepository
from app.llm.base import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class MaturityAssessmentInput(FlowInput):
    """Input for maturity assessment flow"""
    file_obj: BinaryIO
    filename: str
    rubric: dict
    llm_provider: LLMProvider = LLMProvider.OPENAI


class MaturityAssessmentFlow(BaseFlow):
    """
    End-to-end maturity assessment flow.

    Steps:
    1. Upload document to GCS
    2. Parse document with LLM
    3. Process document with RAG (chunking, embeddings)
    4. Score against rubric
    5. Generate recommendations
    6. Store results in database
    """

    def __init__(
        self,
        db: AsyncSession,
        gcs_client: GCSClient,
        rag_service: RAGService
    ):
        super().__init__(
            name="maturity_assessment",
            description="AI maturity assessment from document analysis"
        )
        self.db = db
        self.gcs_client = gcs_client
        self.rag_service = rag_service
        self.tool_registry = get_tool_registry()
        self.assessment_repo = AssessmentRepository(db)
        self.total_steps = 6

    async def execute(self, input_data: MaturityAssessmentInput) -> FlowOutput:
        """
        Execute maturity assessment flow.

        Args:
            input_data: MaturityAssessmentInput with document and rubric

        Returns:
            FlowOutput with assessment results
        """
        steps_completed = 0
        assessment_id = None

        try:
            # Step 1: Create assessment record
            self.logger.info("Step 1: Creating assessment record")
            assessment = await self._execute_step(
                "create_assessment",
                self._create_assessment,
                input_data
            )
            assessment_id = assessment.id
            steps_completed += 1

            # Step 2: Upload document to GCS
            self.logger.info("Step 2: Uploading document to GCS")
            gcs_path = await self._execute_step(
                "upload_document",
                self._upload_document,
                input_data,
                assessment_id
            )
            await self.assessment_repo.update(
                assessment_id,
                {"document_gcs_path": gcs_path}
            )
            steps_completed += 1

            # Step 3: Parse document
            self.logger.info("Step 3: Parsing document")
            parsed_content = await self._execute_step(
                "parse_document",
                self._parse_document,
                input_data,
                gcs_path
            )
            await self.assessment_repo.update(
                assessment_id,
                {"parsed_content": parsed_content}
            )
            steps_completed += 1

            # Step 4: Process with RAG (chunking + embeddings)
            self.logger.info("Step 4: Processing document with RAG")
            input_data.file_obj.seek(0)  # Reset file pointer
            await self._execute_step(
                "process_rag",
                self.rag_service.process_document,
                input_data.file_obj,
                input_data.filename,
                input_data.tenant_id,
                assessment_id,
                gcs_path
            )
            steps_completed += 1

            # Step 5: Score against rubric
            self.logger.info("Step 5: Scoring against rubric")
            scores = await self._execute_step(
                "score_rubric",
                self._score_rubric,
                input_data,
                parsed_content
            )
            await self.assessment_repo.update(
                assessment_id,
                {"rubric_scores": scores}
            )
            steps_completed += 1

            # Step 6: Generate recommendations
            self.logger.info("Step 6: Generating recommendations")
            recommendations = await self._execute_step(
                "generate_recommendations",
                self._generate_recommendations,
                input_data,
                scores
            )
            await self.assessment_repo.update(
                assessment_id,
                {
                    "recommendations": recommendations,
                    "status": "completed"
                }
            )
            steps_completed += 1

            # Commit transaction
            await self.db.commit()

            # Return complete result
            result = {
                "assessment_id": assessment_id,
                "parsed_content": parsed_content,
                "scores": scores,
                "recommendations": recommendations,
                "document_path": gcs_path
            }

            return self._create_output(
                flow_id=str(assessment_id),
                status=FlowStatus.COMPLETED,
                result=result,
                steps_completed=steps_completed,
                total_steps=self.total_steps
            )

        except Exception as e:
            self.logger.error(f"Flow failed at step {steps_completed + 1}: {str(e)}")

            # Update assessment status to failed
            if assessment_id:
                await self.assessment_repo.update_status(
                    assessment_id,
                    status="failed",
                    error_message=str(e)
                )
                await self.db.commit()

            return self._create_output(
                flow_id=str(assessment_id) if assessment_id else "unknown",
                status=FlowStatus.FAILED,
                result={},
                steps_completed=steps_completed,
                total_steps=self.total_steps,
                error=str(e)
            )

    async def _create_assessment(
        self,
        input_data: MaturityAssessmentInput
    ):
        """Create initial assessment record"""
        assessment_data = {
            "tenant_id": input_data.tenant_id,
            "assessment_type": "maturity_assessment",
            "status": "processing",
            "document_name": input_data.filename,
            "llm_provider": input_data.llm_provider.value
        }
        return await self.assessment_repo.create(assessment_data)

    async def _upload_document(
        self,
        input_data: MaturityAssessmentInput,
        assessment_id: int
    ) -> str:
        """Upload document to GCS"""
        input_data.file_obj.seek(0)
        destination_path = f"tenants/{input_data.tenant_id}/assessments/{assessment_id}/{input_data.filename}"

        return self.gcs_client.upload_file(
            file_obj=input_data.file_obj,
            destination_path=destination_path,
            metadata={
                "tenant_id": input_data.tenant_id,
                "assessment_id": str(assessment_id)
            }
        )

    async def _parse_document(
        self,
        input_data: MaturityAssessmentInput,
        gcs_path: str
    ) -> dict:
        """Parse document using parse_docs tool"""
        # Load document text
        input_data.file_obj.seek(0)
        doc_loader = DocumentLoader()
        document_text = doc_loader.load_document(
            input_data.file_obj,
            input_data.filename
        )

        # Execute parse_docs tool
        parse_tool = self.tool_registry.get_tool("parse_docs")
        parse_input = ParseDocsInput(
            tenant_id=input_data.tenant_id,
            document_text=document_text,
            parsing_instructions="Extract key information about AI maturity, current capabilities, challenges, and goals.",
            llm_provider=input_data.llm_provider
        )

        result = await parse_tool.execute(parse_input)

        if result.status != "success":
            raise Exception(f"Document parsing failed: {result.error}")

        return result.result

    async def _score_rubric(
        self,
        input_data: MaturityAssessmentInput,
        parsed_content: dict
    ) -> dict:
        """Score document against rubric"""
        score_tool = self.tool_registry.get_tool("score_rubrics")
        score_input = ScoreRubricsInput(
            tenant_id=input_data.tenant_id,
            document_analysis=parsed_content,
            rubric=input_data.rubric,
            llm_provider=input_data.llm_provider
        )

        result = await score_tool.execute(score_input)

        if result.status != "success":
            raise Exception(f"Rubric scoring failed: {result.error}")

        return result.result

    async def _generate_recommendations(
        self,
        input_data: MaturityAssessmentInput,
        scores: dict
    ) -> dict:
        """Generate recommendations based on scores"""
        rec_tool = self.tool_registry.get_tool("gen_recs")
        rec_input = GenRecsInput(
            tenant_id=input_data.tenant_id,
            scores=scores,
            llm_provider=input_data.llm_provider
        )

        result = await rec_tool.execute(rec_input)

        if result.status != "success":
            raise Exception(f"Recommendation generation failed: {result.error}")

        return result.result
```

### 3. **`/app/flows/usecase_grooming_flow.py`** - Use case grooming flow
```python
import logging
from typing import List, Dict, Any, BinaryIO
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession

from app.flows.base import BaseFlow, FlowInput, FlowOutput, FlowStatus
from app.tools.registry import get_tool_registry
from app.tools.parse_docs_tool import ParseDocsInput
from app.tools.rank_usecases_tool import RankUseCasesInput
from app.tools.write_backlog_tool import WriteBacklogInput
from app.storage.gcs_client import GCSClient
from app.rag.document_loader import DocumentLoader
from app.db.repositories.use_case import UseCaseRepository
from app.llm.base import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class UseCaseGroomingInput(FlowInput):
    """Input for use case grooming flow"""
    use_cases: List[Dict[str, Any]]  # Can be structured data or documents
    ranking_criteria: Dict[str, float] = None
    llm_provider: LLMProvider = LLMProvider.OPENAI


class UseCaseGroomingFlow(BaseFlow):
    """
    End-to-end use case grooming flow.

    Steps:
    1. Parse use case documents (if needed)
    2. Rank use cases by priority
    3. Generate product backlog for top use cases
    4. Store results in database
    """

    def __init__(
        self,
        db: AsyncSession,
        gcs_client: GCSClient
    ):
        super().__init__(
            name="usecase_grooming",
            description="Use case prioritization and backlog generation"
        )
        self.db = db
        self.gcs_client = gcs_client
        self.tool_registry = get_tool_registry()
        self.use_case_repo = UseCaseRepository(db)
        self.total_steps = 4

    async def execute(self, input_data: UseCaseGroomingInput) -> FlowOutput:
        """
        Execute use case grooming flow.

        Args:
            input_data: UseCaseGroomingInput with use cases

        Returns:
            FlowOutput with ranked use cases and backlogs
        """
        steps_completed = 0
        use_case_ids = []

        try:
            # Step 1: Store use cases in database
            self.logger.info("Step 1: Storing use cases")
            use_case_ids = await self._execute_step(
                "store_use_cases",
                self._store_use_cases,
                input_data
            )
            steps_completed += 1

            # Step 2: Rank use cases
            self.logger.info("Step 2: Ranking use cases")
            rankings = await self._execute_step(
                "rank_use_cases",
                self._rank_use_cases,
                input_data
            )
            steps_completed += 1

            # Step 3: Update rankings in database
            self.logger.info("Step 3: Updating rankings")
            await self._execute_step(
                "update_rankings",
                self._update_rankings,
                use_case_ids,
                rankings
            )
            steps_completed += 1

            # Step 4: Generate backlogs for top use cases (top 3)
            self.logger.info("Step 4: Generating product backlogs")
            top_use_cases = rankings["ranked_use_cases"][:3]
            backlogs = await self._execute_step(
                "generate_backlogs",
                self._generate_backlogs,
                input_data,
                top_use_cases
            )
            steps_completed += 1

            # Commit transaction
            await self.db.commit()

            result = {
                "use_case_ids": use_case_ids,
                "rankings": rankings,
                "backlogs": backlogs
            }

            return self._create_output(
                flow_id=f"grooming_{input_data.tenant_id}",
                status=FlowStatus.COMPLETED,
                result=result,
                steps_completed=steps_completed,
                total_steps=self.total_steps
            )

        except Exception as e:
            self.logger.error(f"Flow failed at step {steps_completed + 1}: {str(e)}")

            return self._create_output(
                flow_id=f"grooming_{input_data.tenant_id}",
                status=FlowStatus.FAILED,
                result={},
                steps_completed=steps_completed,
                total_steps=self.total_steps,
                error=str(e)
            )

    async def _store_use_cases(
        self,
        input_data: UseCaseGroomingInput
    ) -> List[int]:
        """Store use cases in database"""
        use_case_ids = []

        for uc in input_data.use_cases:
            use_case_data = {
                "tenant_id": input_data.tenant_id,
                "title": uc.get("title", "Untitled"),
                "description": uc.get("description", ""),
                "status": "pending",
                "document_name": uc.get("document_name", ""),
                "document_gcs_path": uc.get("document_gcs_path", ""),
                "llm_provider": input_data.llm_provider.value,
                "parsed_content": uc
            }

            use_case = await self.use_case_repo.create(use_case_data)
            use_case_ids.append(use_case.id)

        return use_case_ids

    async def _rank_use_cases(
        self,
        input_data: UseCaseGroomingInput
    ) -> dict:
        """Rank use cases using rank_usecases tool"""
        rank_tool = self.tool_registry.get_tool("rank_usecases")
        rank_input = RankUseCasesInput(
            tenant_id=input_data.tenant_id,
            use_cases=input_data.use_cases,
            criteria=input_data.ranking_criteria,
            llm_provider=input_data.llm_provider
        )

        result = await rank_tool.execute(rank_input)

        if result.status != "success":
            raise Exception(f"Use case ranking failed: {result.error}")

        return result.result

    async def _update_rankings(
        self,
        use_case_ids: List[int],
        rankings: dict
    ):
        """Update use case rankings in database"""
        ranked_cases = rankings.get("ranked_use_cases", [])

        for i, ranked in enumerate(ranked_cases):
            if i < len(use_case_ids):
                await self.use_case_repo.update(
                    use_case_ids[i],
                    {
                        "priority_score": ranked.get("overall_score", 0),
                        "ranking_rationale": ranked.get("rationale", "")
                    }
                )

    async def _generate_backlogs(
        self,
        input_data: UseCaseGroomingInput,
        top_use_cases: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate product backlogs for top use cases"""
        backlog_tool = self.tool_registry.get_tool("write_backlog")
        backlogs = []

        for uc in top_use_cases:
            backlog_input = WriteBacklogInput(
                tenant_id=input_data.tenant_id,
                use_case=uc,
                llm_provider=input_data.llm_provider
            )

            result = await backlog_tool.execute(backlog_input)

            if result.status == "success":
                backlogs.append({
                    "use_case_id": uc.get("use_case_id"),
                    "backlog": result.result
                })
            else:
                self.logger.warning(f"Backlog generation failed for use case: {result.error}")

        return backlogs
```

### 4. **`/app/flows/__init__.py`** - Module exports
```python
from app.flows.base import BaseFlow, FlowInput, FlowOutput, FlowStatus
from app.flows.maturity_assessment_flow import MaturityAssessmentFlow, MaturityAssessmentInput
from app.flows.usecase_grooming_flow import UseCaseGroomingFlow, UseCaseGroomingInput

__all__ = [
    "BaseFlow",
    "FlowInput",
    "FlowOutput",
    "FlowStatus",
    "MaturityAssessmentFlow",
    "MaturityAssessmentInput",
    "UseCaseGroomingFlow",
    "UseCaseGroomingInput",
]
```

### 5. **`/app/api/v1/assessments.py`** - Assessment API endpoints
```python
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
import json

from app.db.session import get_db
from app.storage.gcs_client import GCSClient
from app.rag.rag_service import RAGService
from app.flows.maturity_assessment_flow import MaturityAssessmentFlow, MaturityAssessmentInput
from app.llm.base import LLMProvider
from app.api.deps import get_current_tenant

router = APIRouter()


@router.post("/assessments")
async def create_assessment(
    file: UploadFile = File(...),
    rubric: str = Form(...),
    llm_provider: Optional[str] = Form("openai"),
    db: AsyncSession = Depends(get_db),
    tenant_id: str = Depends(get_current_tenant)
):
    """
    Create a new maturity assessment.

    Args:
        file: Document file (PDF, DOCX, TXT)
        rubric: JSON string containing evaluation rubric
        llm_provider: LLM provider to use (openai or vertex_ai)
        tenant_id: Authenticated tenant ID

    Returns:
        Assessment results with scores and recommendations
    """
    try:
        # Parse rubric JSON
        rubric_dict = json.loads(rubric)

        # Initialize services
        gcs_client = GCSClient()
        rag_service = RAGService(db, gcs_client)

        # Initialize flow
        flow = MaturityAssessmentFlow(db, gcs_client, rag_service)

        # Prepare input
        flow_input = MaturityAssessmentInput(
            tenant_id=tenant_id,
            file_obj=file.file,
            filename=file.filename,
            rubric=rubric_dict,
            llm_provider=LLMProvider(llm_provider)
        )

        # Execute flow
        result = await flow.execute(flow_input)

        if result.status == "failed":
            raise HTTPException(status_code=500, detail=result.error)

        return {
            "assessment_id": result.flow_id,
            "status": result.status,
            "data": result.result
        }

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid rubric JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/assessments/{assessment_id}")
async def get_assessment(
    assessment_id: int,
    db: AsyncSession = Depends(get_db),
    tenant_id: str = Depends(get_current_tenant)
):
    """Get assessment by ID"""
    from app.db.repositories.assessment import AssessmentRepository

    repo = AssessmentRepository(db)
    assessment = await repo.get_by_tenant(tenant_id, assessment_id)

    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")

    return assessment
```

### 6. **`/app/api/v1/use_cases.py`** - Use case API endpoints
```python
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any, Optional

from app.db.session import get_db
from app.storage.gcs_client import GCSClient
from app.flows.usecase_grooming_flow import UseCaseGroomingFlow, UseCaseGroomingInput
from app.llm.base import LLMProvider
from app.api.deps import get_current_tenant

router = APIRouter()


@router.post("/use-cases/groom")
async def groom_use_cases(
    use_cases: List[Dict[str, Any]] = Body(...),
    ranking_criteria: Optional[Dict[str, float]] = Body(None),
    llm_provider: Optional[str] = Body("openai"),
    db: AsyncSession = Depends(get_db),
    tenant_id: str = Depends(get_current_tenant)
):
    """
    Groom use cases: rank and generate backlogs.

    Args:
        use_cases: List of use case dictionaries
        ranking_criteria: Optional custom ranking weights
        llm_provider: LLM provider to use
        tenant_id: Authenticated tenant ID

    Returns:
        Ranked use cases with generated backlogs
    """
    try:
        # Initialize services
        gcs_client = GCSClient()

        # Initialize flow
        flow = UseCaseGroomingFlow(db, gcs_client)

        # Prepare input
        flow_input = UseCaseGroomingInput(
            tenant_id=tenant_id,
            use_cases=use_cases,
            ranking_criteria=ranking_criteria,
            llm_provider=LLMProvider(llm_provider)
        )

        # Execute flow
        result = await flow.execute(flow_input)

        if result.status == "failed":
            raise HTTPException(status_code=500, detail=result.error)

        return {
            "status": result.status,
            "data": result.result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 7. **`/app/api/deps.py`** - API dependencies
```python
from fastapi import Header, HTTPException
from typing import Optional

async def get_current_tenant(
    x_api_key: Optional[str] = Header(None)
) -> str:
    """
    Extract tenant ID from API key.

    In production, this would:
    1. Validate API key against database
    2. Extract tenant ID from validated key
    3. Check rate limits

    For now, returns a mock tenant ID.
    """
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required")

    # TODO: Implement actual API key validation
    # This is a placeholder
    return "tenant_123"
```

## Dependencies
- **Upstream**: Tools Engineer (all 5 tools), Database Engineer (repositories), RAG Engineer (RAG service), LLM Engineer (providers), Storage (GCS)
- **Downstream**: QA Engineer (will test flows)

## Working Style
1. **Error resilience**: Handle partial failures gracefully
2. **State management**: Track flow progress for observability
3. **Async execution**: Support background processing for long-running flows
4. **Clear interfaces**: Well-defined input/output contracts

## Success Criteria
- [ ] Both flows execute successfully end-to-end
- [ ] Flow orchestration handles tool coordination correctly
- [ ] Error handling provides clear failure information
- [ ] Progress tracking works accurately
- [ ] API endpoints are functional and well-documented
- [ ] Database transactions are handled properly
- [ ] Flows can recover from partial failures

## Notes
- Consider adding background task support (Celery/Redis) for async execution
- Implement flow state persistence for long-running operations
- Add webhooks for flow completion notifications
- Consider implementing flow replay for debugging
- Add metrics and monitoring for flow performance
