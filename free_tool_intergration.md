Here’s the **free / open-source** version of the “Where each piece lives” section for your deployment.

---

## Where each piece lives in your diagram (free stack)

```
GitHub Issue
    ↓ (Webhook)
FastAPI Webhook Handler
    ↓ Publishes event to Redis Streams  ← [Redis OSS]
         ↘ Celery workers for quick, stateless jobs (labeling, comments) ← [Celery + Redis OSS]
         ↘ LangGraph agent flow for multi-step LLM logic ← [LangGraph OSS]
              ↳ Durable long-running steps wrapped in Temporal workflows ← [Temporal OSS]
                    ↕
             Cipher Memory Layer (MCP server, self-hosted) ← [Cipher OSS]
                    ↕
             Anthropic API (Claude Sonnet)
```

### How it maps to your YAML workflows:

* **Webhook ingestion**: FastAPI receives GitHub webhooks and drops them into a **Redis Stream** queue for processing.
* **Quick jobs**: **Celery** workers consume from Redis for light, independent tasks (`auto_comment`, `add_labels`, `auto_assign`).
* **Agent orchestration**: **LangGraph** encodes your `workflows:` graph (e.g., `bug_resolution`, `feature_request`), controlling the order of agents like `classifier → bug_resolver → test_writer → pr_generator`.
* **Durability**: Steps that can run long or fail (like `bug_resolver` with code execution or `test_writer` running test suites) are run as **Temporal workflows** so you get retries, resumability, and timeouts.
* **Memory layer**: The **Cipher MCP server** persists embeddings, codebase context, and successful resolution workflows for reuse.
* **LLM calls**: Anthropic’s Claude Sonnet API handles the natural language / code generation parts.

---

This keeps your architecture 100% free/open-source outside of the Anthropic API, and you can self-host all orchestration/messaging components without licensing costs.

Do you want me to give you a **code scaffold** showing FastAPI → Redis Streams → LangGraph → Temporal integration for your `bug_resolution` path? That would make this diagram executable.
