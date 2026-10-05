# SpeechLens Project Directives: GSD, Ponytail, CodeRabbit, and Ralph Loop

> **MANDATORY SYSTEM DIRECTIVE**: These rules are active on **every prompt, every task, and every session**.
> You do **not** need the user to prompt, trigger, or remind you to apply them. They are your default operating system.

---

## 1. Ponytail: Lazy Senior Dev & YAGNI Mode (Anti-Overengineering)

- **The Best Code is the Code Never Written**:
  - Stop at the first rung of the Ponytail Ladder:
    1. **Does this need to exist at all? (YAGNI)** If speculative, skip it.
    2. **Is it already in this codebase?** Reuse existing helpers, functions, or patterns. Never write duplicate utilities.
    3. **Does stdlib do it?** Use Python standard libraries (`pathlib`, `collections`, `itertools`, etc.).
    4. **Does an installed dependency already solve it?** Use existing packages in `requirements.txt`. Never add new dependencies without explicit request.
    5. **Can it be one line?** Make it one line.
    6. **Only then:** write the minimum code that works.
- **Root-Cause Bug Fixing**:
  - Always grep for callers of the function you touch.
  - Fix bugs at the root cause (the shared core logic) rather than patching surface symptoms in individual callers.
- **Minimal Diffs**:
  - No speculative scaffolding, no unrequested abstractions, no interfaces with a single implementation.
  - Deletion over addition. Boring and reliable over clever.

---

## 2. CodeRabbit: Autonomous Deep Code Review & Quality Assurance

- **Mandatory Self-Review on Every Code Edit**:
  - **Edge Cases**: Check against [`docs/edge_cases.md`](file:///c:/Personal/IIT%20MANDI/docs/edge_cases.md) (e.g. empty/short audio, index bounds, zero division, silence).
  - **Zero Silent Failures**: No unhandled NaNs or swallowed errors. Functions must fail fast with clear errors rather than returning corrupt state.
  - **Diff Cleanliness**: Ensure no stray print statements, broken imports, or unintended modifications.
  - **Verification Gate**: Never declare a task complete without running and verifying test suites.

---

## 3. Ralph Loop: Autonomous Iterative Execution Protocol

- **Never Stop Halfway on Broken Tests**:
  - When making code changes, immediately execute the test suite:
    ```powershell
    .\.venv\Scripts\Activate.ps1
    pytest -v
    ```
  - If a test fails, do **not** stop and ask the user to fix it or report a half-finished failure.
  - Read the error traceback, diagnose the root cause, apply the minimal fix (Ponytail style), and re-run the tests.
  - Continue the loop until all tests are green and all acceptance criteria are met.

---

## 4. Get Shit Done (GSD): Structured Planning & State Tracking

- **State Persistence**:
  - Track current milestone, completed work, and active phase in [`.planning/STATE.md`](file:///c:/Personal/IIT%20MANDI/.planning/STATE.md).
  - High-level architecture in [`.planning/PROJECT.md`](file:///c:/Personal/IIT%20MANDI/.planning/PROJECT.md) and phases in [`.planning/ROADMAP.md`](file:///c:/Personal/IIT%20MANDI/.planning/ROADMAP.md).
- **Atomic Milestones**:
  - Complete work in verifiable phases.
  - Each phase must have clear acceptance criteria verified by tests before proceeding to the next.

---

## 5. Environment & Execution Standards

- **Working Directory**: The active Python package is here in `speechlens`. Run scripts and tests with CWD inside `speechlens` using `.venv`.
- **Dependencies**: Use the dedicated Python 3.12 virtual environment at `.venv`.
- **eSpeak-NG**: Required for synthetic test fixtures and baseline generation. Available at `C:\Program Files\eSpeak NG\espeak-ng.exe`.
