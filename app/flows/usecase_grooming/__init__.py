"""Use-Case Grooming Flow.

Orchestrates the following steps:
1. Load assessment.json from previous flow
2. Rank use-cases using RICE/WSJF scoring
3. Generate prioritized backlog
4. Emit backlog artifact to GCS
"""
