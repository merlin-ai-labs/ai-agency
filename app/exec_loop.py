"""Simple execution loop for processing queued runs.

In this lean architecture, the execution loop runs in-process (no Pub/Sub).
It polls the database for runs with status='queued', executes them, and updates status.

TODO:
- Implement polling mechanism (background task or cron-like scheduler)
- Load flow graph by name
- Execute flow with input_data
- Handle errors and update run status
- Store output artifacts to GCS
"""


import structlog

logger = structlog.get_logger()


async def poll_and_execute():
    """
    Poll the database for queued runs and execute them.

    TODO:
    - Query: SELECT * FROM runs WHERE status='queued' ORDER BY created_at LIMIT 1
    - Update status to 'running'
    - Load flow graph from flows/{flow_name}/graph.py
    - Execute flow.run(input_data)
    - On success: Update status='completed', store artifacts
    - On failure: Update status='failed', store error message
    - Commit transaction
    """
    logger.info("poll_and_execute", message="Stub: no runs to process")


async def execute_run(run_id: str, flow_name: str, input_data: dict):
    """
    Execute a single run.

    Args:
        run_id: Unique run identifier
        flow_name: Name of the flow to execute (e.g., 'maturity_assessment')
        input_data: Input data for the flow

    TODO:
    - Dynamically import flow graph: importlib.import_module(f"app.flows.{flow_name}.graph")
    - Call flow.run(input_data)
    - Capture output and artifacts
    - Upload artifacts to GCS
    - Return result
    """
    logger.info("execute_run", run_id=run_id, flow=flow_name)

    # Stub implementation
    result = {
        "run_id": run_id,
        "status": "completed",
        "output": {},
        "artifacts": [],
    }

    return result


async def start_background_poller():
    """
    Start a background task that polls for queued runs.

    TODO:
    - Use asyncio.create_task() or BackgroundTasks from FastAPI
    - Run poll_and_execute() every N seconds
    - Add graceful shutdown handling
    """
    logger.info("start_background_poller", message="Stub: poller not started")
