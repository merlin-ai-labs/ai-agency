"""FastAPI application entry point.

Provides:
- Health check endpoint
- POST /runs — Create and enqueue a new flow execution
- GET /runs/{run_id} — Retrieve run status and results
- POST /weather-chat — Weather agent chat endpoint
- Invoice Manager endpoints (imported from flows.invoice_manager.api)

TODO:
- Add authentication/API key middleware
- Add request validation with Pydantic models
"""

from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.core.langgraph_checkpointer import TenantAwarePostgresSaver
from app.flows.agents.weather_agent import WeatherAgentFlow

# Import invoice manager endpoints and register routes
from app.flows.invoice_manager.api import (
    CapabilitiesResponse,
    InvoiceManagerRunRequest,
    InvoiceManagerRunResponse,
    InvoiceSearchRequest,
    InvoiceSearchResponse,
    get_capabilities,
    run_invoice_manager,
    search_invoices_endpoint,
    upload_invoice_file,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup/shutdown events.

    Startup:
    - Connection pools are created lazily (on first use)
    - SQLAlchemy engine manages its own pool lifecycle
    - LangGraph checkpointer pool created on first checkpointer instance

    Shutdown:
    - Close LangGraph checkpointer pool if it was created
    - SQLAlchemy engine cleanup handled automatically
    """
    # Startup
    logger.info("application_startup", service="ai-agency")

    yield

    # Shutdown
    logger.info("application_shutdown", service="ai-agency")

    # Close LangGraph checkpointer pool if it exists
    if TenantAwarePostgresSaver._pool is not None:
        logger.info("closing_langgraph_checkpointer_pool")
        TenantAwarePostgresSaver._pool.close()
        logger.info("langgraph_checkpointer_pool_closed")


app = FastAPI(
    title="AI Agency",
    description="Lean AI agents platform for maturity assessment and use-case grooming",
    version="0.1.0",
    lifespan=lifespan,
)

# Register invoice manager routes
app.post("/api/v1/invoice-manager/run", response_model=InvoiceManagerRunResponse)(
    run_invoice_manager
)
app.get("/api/v1/invoice-manager/capabilities", response_model=CapabilitiesResponse)(
    get_capabilities
)
app.post("/api/v1/invoice-manager/search", response_model=InvoiceSearchResponse)(
    search_invoices_endpoint
)
app.post("/api/v1/invoice-manager/upload")(upload_invoice_file)


class RunRequest(BaseModel):
    """Request to create a new flow execution."""

    flow_name: str  # e.g., "maturity_assessment" or "usecase_grooming"
    tenant_id: str
    input_data: dict[str, Any]


class RunResponse(BaseModel):
    """Response with run ID and initial status."""

    run_id: str
    status: str  # "queued", "running", "completed", "failed"
    message: str | None = None


class WeatherChatRequest(BaseModel):
    """Request to chat with weather agent."""

    message: str
    tenant_id: str = "default"
    conversation_id: str | None = None


class WeatherChatResponse(BaseModel):
    """Response from weather agent."""

    response: str
    conversation_id: str
    tool_used: bool
    tool_results: dict[str, Any] | None = None


@app.get("/healthz")
async def healthz():
    """Health check endpoint for Cloud Run.

    Basic health check that doesn't test database connectivity.
    Used by Cloud Run for fast startup/liveness checks.
    """
    return {"status": "ok", "service": "ai-agency"}


@app.get("/health")
async def health():
    """Comprehensive health check with database status.

    Returns:
        - status: "ok" or "degraded"
        - service: "ai-agency"
        - database: "connected" or "error"
        - database_error: error message if database check fails
    """
    from sqlmodel import Session, select

    from app.db.base import get_session
    from app.db.models import Tenant

    health_status = {
        "status": "ok",
        "service": "ai-agency",
        "database": "unknown",
    }

    # Test database connectivity
    try:
        with Session(get_session()) as session:
            # Simple query to test connection
            result = session.exec(select(Tenant).limit(1)).first()
            health_status["database"] = "connected"
            logger.info("health_check_database_ok")
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["database"] = "error"
        health_status["database_error"] = str(e)
        logger.error("health_check_database_failed", error=str(e))

    return health_status


@app.post("/runs", response_model=RunResponse)
async def create_run(req: RunRequest):
    """
    Create a new flow run and enqueue it for execution.

    TODO:
    - Validate flow_name against registered flows
    - Insert run record into database (status=queued)
    - Trigger execution loop (background task or poller)
    - Return run_id
    """
    logger.info("create_run", flow=req.flow_name, tenant=req.tenant_id)

    # Stub: Generate fake run_id
    run_id = f"run_{req.tenant_id}_{req.flow_name}_stub"

    return RunResponse(
        run_id=run_id,
        status="queued",
        message="Run created (stub implementation)",
    )


@app.get("/runs/{run_id}", response_model=RunResponse)
async def get_run(run_id: str):
    """
    Retrieve the status and results of a run.

    TODO:
    - Query database for run record
    - Return status, output artifacts (GCS URLs), error messages
    - If completed, include link to assessment.json or backlog
    """
    logger.info("get_run", run_id=run_id)

    # Stub: Return fake status
    return RunResponse(
        run_id=run_id,
        status="completed",
        message="Run status (stub implementation)",
    )


@app.post("/weather-chat", response_model=WeatherChatResponse)
async def weather_chat(req: WeatherChatRequest):
    """
    Chat with weather agent.

    This endpoint provides a conversational interface to get weather information.
    The agent uses LLM tool calling to fetch real-time weather data when needed.

    Request:
        POST /weather-chat
        {
            "message": "What's the weather in London?",
            "tenant_id": "user123",
            "conversation_id": "optional-uuid"  # For continuing conversation
        }

    Response:
        {
            "response": "It's 15°C and cloudy in London",
            "conversation_id": "uuid",
            "tool_used": true,
            "tool_results": { ... }
        }

    Example:
        >>> curl -X POST http://localhost:8000/weather-chat \\
        ...   -H "Content-Type: application/json" \\
        ...   -d '{"message": "What is the weather in Paris?", "tenant_id": "test"}'
    """
    try:
        logger.info(
            "weather_chat",
            message=req.message,
            tenant_id=req.tenant_id,
            conversation_id=req.conversation_id,
        )

        # Initialize weather agent flow
        flow = WeatherAgentFlow()

        # Execute flow
        result = await flow.run(
            user_message=req.message,
            tenant_id=req.tenant_id,
            conversation_id=req.conversation_id,
        )

        logger.info(
            "weather_chat_success",
            conversation_id=result["conversation_id"],
            tool_used=result["tool_used"],
        )

        return WeatherChatResponse(
            response=result["response"],
            conversation_id=result["conversation_id"],
            tool_used=result["tool_used"],
            tool_results=result.get("tool_results"),
        )

    except Exception as e:
        logger.exception("weather_chat_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Weather chat failed: {str(e)}",
        ) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
