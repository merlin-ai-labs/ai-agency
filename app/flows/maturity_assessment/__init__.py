"""Maturity Assessment Flow.

Orchestrates the following steps:
1. Parse uploaded documents (PDFs, Word, etc.)
2. Score against maturity rubrics
3. Generate recommendations
4. Emit assessment.json + artifacts to GCS
"""
