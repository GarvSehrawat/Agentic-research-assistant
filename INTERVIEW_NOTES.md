# Interview Notes

Practical answers based on how *this* project is actually built, not generic
theory.

## Fundamentals

**1. What is Agentic AI?**
An LLM given the ability to perceive a task, plan steps, choose and call
tools, observe results, and decide when it's done — rather than producing one
static response. The key ingredients are: state, planning, tool use, and a
loop that can react to what tools return.

**2. How is an agent different from a chatbot?**
A chatbot maps a prompt to a completion. This agent maps a query to a
*workflow*: it can call a calculator, hit a web search API, retrieve document
chunks, check whether what it has is enough, and loop back if not — all
before it ever writes the user-facing answer. A chatbot has no notion of
"is my answer actually grounded in something real."

**3. What is an LLM?**
A large language model — a neural network trained to predict text — used
here as the reasoning/synthesis component: it classifies queries, decomposes
tasks, and writes the final answer, but it is deliberately *not* trusted to
do arithmetic or know today's news; those go to tools instead.

**4. What is LangChain?**
A library of building blocks (prompt templates, model wrappers, output
parsers) for LLM applications. This project uses `langchain-core` sparingly
and leans more on LangGraph for orchestration, keeping most logic in plain
Python (`app/agent/planner.py`, `app/tools/`) so the flow stays readable.

**5. What is LangGraph?**
A graph-based orchestration library for building stateful, cyclic LLM
workflows — nodes are functions that take/return a shared state dict, edges
(including conditional ones) decide what runs next. That's exactly the shape
of this project's `analyze -> plan -> execute -> check -> (loop) -> synthesize`
flow.

**6. Why use LangGraph (here)?**
Because the workflow isn't linear — the "evidence sufficient?" check can send
the agent back to re-plan and re-execute a bounded number of times. A plain
function pipeline can express that too (and this project *does* include a
plain-Python fallback runner, `_run_fallback`, used only if LangGraph isn't
importable), but LangGraph gives that loop a clear, inspectable graph
structure and off-the-shelf state merging between nodes.

## Architecture

**7. How does query decomposition work?**
`app/agent/planner.py:create_tasks` asks the LLM (via a structured-output
call) to break a query into at most N subtasks, each tagged with a suggested
tool. N is capped by research depth (`basic`/`standard`/`deep`) and by the
global `MAX_TASKS` safety limit. Simple queries skip this step entirely.

**8. How does the planner work?**
Two stages: `analyze_query` classifies the query (simple/moderate/complex,
which tool categories are plausibly needed) using structured LLM output with
a heuristic fallback if the LLM call fails; then, for non-simple queries,
`create_tasks` decomposes it into concrete subtasks.

**9. How does tool routing work?**
`app/agent/router.py:route_task` takes the tool the planner suggested and
validates it against the `ToolRegistry` — if it's not a real, registered
tool, it falls back to a small keyword heuristic (math symbols -> calculator,
"latest/current" -> web_search, "the document/report" -> document_search).
There is no path that lets an unregistered or invented tool name execute.

**10. How does the agent maintain state?**
A single `AgentState` `TypedDict` (`app/agent/state.py`) — query, analysis,
tasks, tool_results, sources, iteration count, errors, final answer — flows
through every graph node. LangGraph merges each node's partial return dict
into that shared state between steps.

**11. How do tools communicate with the agent?**
Every tool implements `BaseTool.execute(**kwargs) -> ToolResult` (`success`,
`output`, `error`, `sources`). Tools never raise past their own boundary —
failures come back as `ToolResult(success=False, error=...)` so the graph can
record the failure and continue instead of crashing the whole run.

## RAG

**12. What are embeddings?**
Numeric vector representations of text such that semantically similar text
ends up close together in vector space, enabling similarity search instead of
exact keyword matching.

**13. Why use FAISS?**
It's a fast, well-tested library for nearest-neighbor search over dense
vectors, runs locally with no external service, and is a natural fit for a
project-scale document index (`IndexFlatL2` here — exact search, which is
plenty fast at this scale and simpler than an approximate index).

**14. What is chunking?**
Splitting long documents into smaller overlapping pieces (`app/retrieval/
chunker.py`, ~800 chars with ~120 char overlap) before embedding, so each
retrieved unit is small enough to fit in an LLM prompt and specific enough to
be relevant, while overlap avoids losing context that straddles a chunk
boundary.

**15. How does document retrieval work?**
Upload -> extract text (PyMuPDF for PDF) -> chunk -> embed each chunk ->
store in FAISS with metadata (document id, filename, page, chunk id) ->
at query time, embed the query and run `vector_store.search()` for the
nearest chunks, which come back with their original metadata attached.

**16. How is RAG different from fine-tuning?**
Fine-tuning bakes new knowledge into the model's weights (expensive, static,
opaque). RAG keeps the model unchanged and instead retrieves relevant text at
query time and puts it in the prompt — cheaper to update (just re-index), and
the answer can cite exactly which retrieved passage it came from.

## Reliability

**17. How do you prevent infinite agent loops?**
Three independent caps: `MAX_AGENT_ITERATIONS` bounds the
plan/execute/check-evidence cycle, `MAX_TOOL_CALLS` bounds total tool
invocations across the whole run, `MAX_TASKS` bounds how many subtasks a plan
can contain. `_node_check_evidence` also forces `evidence_sufficient=True`
once the iteration cap is hit, guaranteeing the graph reaches
`synthesize_answer`.

**18. What happens when a tool fails?**
The task is marked `FAILED`, the error is recorded in `state["errors"]` and
in a `ToolExecution` DB row, and the agent keeps going with the tasks that did
succeed. The final answer synthesis only uses evidence that's actually
present — it doesn't pretend the failed task succeeded.

**19. How do you validate LLM output?**
`LLMProvider.generate_structured` asks for JSON matching a Pydantic schema,
parses and validates it, and on failure retries once with the validation
error appended to the prompt. If it still fails, callers (`analyze_query`,
`create_tasks`) fall back to a heuristic rather than raising into the request.

**20. How do you handle hallucinations?**
Structurally, by design: arithmetic goes to the calculator, current-events
questions go to search, document questions go to retrieval — the LLM is only
asked to *synthesize* from evidence actually collected, and the synthesis
prompt explicitly instructs it to say when evidence is insufficient rather
than invent facts or sources.

**21. How do you handle prompt injection?**
Every piece of retrieved content (web snippets, document chunks) is wrapped
with `fence_untrusted_content()` before it reaches the LLM, with an explicit
"untrusted data, not instructions" label, and the synthesis system prompt
tells the model never to follow instructions found inside that fenced
content — only to extract facts from it.

## Engineering

**22. Why Flask?**
Small, unopinionated, and well-understood — appropriate for a project this
size where the interesting engineering is in the agent, not the web
framework. An application-factory pattern (`create_app`) keeps it testable.

**23. Why SQLite?**
Zero setup, file-based, perfectly adequate for a single-process portfolio
project. `DATABASE_URL` is the only thing that would need to change to move
to Postgres.

**24. How would you scale this system?**
Move research execution to background workers (Celery/RQ + Redis) so
requests return immediately with a session id and poll/stream for results;
move from SQLite to Postgres for concurrent writes; move the FAISS index
behind a dedicated vector-store service if the document set grows past what
fits comfortably in one process's memory.

**25. How would you add authentication?**
An API-key or JWT middleware in front of the Flask blueprints, checked before
`/api/research` and `/api/documents/*`, with per-key usage/rate tracking
recorded alongside the existing `ToolExecution`/`ResearchSession` tables.

**26. How would you make research asynchronous?**
`POST /api/research` would enqueue a job and return `{session_id,
status: "pending"}` immediately; a worker process would run the same
`ResearchAgent.run()` and write results to the same `ResearchSession` row;
`GET /api/research/<id>` (which already exists) becomes the polling endpoint,
or a websocket/SSE stream could push updates.

**27. How would you reduce LLM costs?**
Skip planning for simple queries (already done); cache repeated
query-analysis/planning results; use a cheaper model for
classification/routing and reserve a stronger model for final synthesis;
cap `MAX_TASKS`/`MAX_TOOL_CALLS` more aggressively for `basic` depth
(already parameterized per depth).

**28. How would you improve latency?**
Run independent subtasks concurrently instead of sequentially (currently
`_node_execute_tools` loops tasks one at a time); stream the synthesis
response token-by-token to the client instead of waiting for the full
completion.

## Design

**29. Why not simply use one LLM call?**
Because one call can't run a calculator, can't know today's news, and can't
see a document you just uploaded. It also can't tell you *why* it's right —
there's no evidence trail. The agent's value is in that trail (sources,
tool results) and in offloading tasks the LLM is bad at.

**30. When should you NOT use an agentic architecture?**
For simple, single-shot Q&A where a direct model call already answers well
(this project explicitly special-cases that — see the `simple` query-type
short-circuit); when latency matters more than grounding; when the added
complexity (more LLM calls, more failure modes, more cost) isn't justified by
the task.

**31. What are the limitations of your system?**
No concurrent tool execution within a task list (sequential only); the
evidence-sufficiency check is a single LLM judgment call, not a rigorous
verifier; FAISS `IndexFlatL2` won't scale to very large document collections
without switching index types; web search quality depends entirely on the
configured provider; no authentication or rate limiting yet.

**32. What would you improve in version 2?**
Async task execution (both for subtasks within a run and for the API itself),
a real evaluation set with human-graded answer quality (not just tool
selection accuracy), a more rigorous verifier step (e.g. checking each claim
in the draft answer against its cited source), and swapping SQLite/local
FAISS for Postgres + a managed vector store for multi-user scale.
