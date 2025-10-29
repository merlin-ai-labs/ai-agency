"""Smoke test script — trigger a test run locally.

Creates a test run and polls for completion.

TODO:
- Add command-line arguments (flow name, tenant ID, etc.)
- Call POST /runs API
- Poll GET /runs/{id} until completed
- Display results
"""

import asyncio
import httpx
import time
import structlog

logger = structlog.get_logger()

API_BASE = "http://localhost:8080"


async def create_run(flow_name: str, tenant_id: str) -> str:
    """
    Create a test run.

    TODO:
    - Call POST /runs API
    - Return run_id
    """
    logger.info("smoke_run.create", flow=flow_name, tenant=tenant_id)

    payload = {
        "flow_name": flow_name,
        "tenant_id": tenant_id,
        "input_data": {"test": True},
    }

    # Stub implementation
    # async with httpx.AsyncClient() as client:
    #     response = await client.post(f"{API_BASE}/runs", json=payload)
    #     response.raise_for_status()
    #     data = response.json()
    #     return data["run_id"]

    return "run_stub_123"


async def poll_run(run_id: str, timeout: int = 60) -> dict:
    """
    Poll for run completion.

    TODO:
    - Poll GET /runs/{id} every N seconds
    - Return when status is completed or failed
    - Timeout after N seconds
    """
    logger.info("smoke_run.poll", run_id=run_id)

    start = time.time()

    while time.time() - start < timeout:
        # Stub implementation
        # async with httpx.AsyncClient() as client:
        #     response = await client.get(f"{API_BASE}/runs/{run_id}")
        #     response.raise_for_status()
        #     data = response.json()
        #     if data["status"] in ["completed", "failed"]:
        #         return data

        await asyncio.sleep(2)

    raise TimeoutError(f"Run {run_id} did not complete within {timeout}s")


async def main():
    """
    Main entry point.

    TODO:
    - Parse command-line arguments
    - Create run
    - Poll for completion
    - Display results
    """
    print("🧪 Running smoke test...")

    flow_name = "maturity_assessment"
    tenant_id = "test_tenant"

    print(f"Creating run: {flow_name} for tenant {tenant_id}")
    run_id = await create_run(flow_name, tenant_id)
    print(f"✓ Run created: {run_id}")

    print("Polling for completion...")
    result = await poll_run(run_id)
    print(f"✅ Run completed: {result}")


if __name__ == "__main__":
    asyncio.run(main())
