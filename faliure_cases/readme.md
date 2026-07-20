# Failure Case Analysis

This directory documents known failure modes of the RAG + auditing system.

These cases are intentionally preserved to demonstrate:
- Where LLMs hallucinate confidently
- How partial evidence can mislead
- How conflicting sources break naive pipelines
- Why temporal validity matters

Each case includes:
- Raw RAG output
- Claim-level audit results
- Final safety verdict

The system is designed to surface these failures, not hide them.
