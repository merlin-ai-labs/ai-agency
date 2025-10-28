"""FastAPI application entry point.

Provides:
- Health check endpoint
- POST /runs — Create and enqueue a new flow execution
- GET /runs/{run_id} — Retrieve run status and results

TODO:
- Wire up database session management
- Add authentication/API key middleware
- Add request validation with Pydantic models
"""

from typing import Any

import structlog
from fastapi import FastAPI
from pydantic import BaseModel

logger = structlog.get_logger()

app = FastAPI(
    title="AI Agency",
    description="Lean AI agents platform for maturity assessment and use-case grooming",
    version="0.1.0",
)


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


@app.get("/healthz")
async def healthz():
    """Health check endpoint for Cloud Run."""
    return {"status": "ok", "service": "ai-agency"}


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
