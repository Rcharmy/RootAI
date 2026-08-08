# Design decisions

This document captures the architectural decisions in RootAI and the reasoning behind them. It exists so that a reader can reconstruct the "why" from just reading the code, and so that an interviewer can dive into any specific tradeoff without needing me to explain it.

Structured as a sequence of questions. Each one names a decision and explains the alternatives considered.

---

## Data and schema

### Why order_item grain instead of order grain?

An order can contain multiple line items. The Olist dataset carries item-level price, freight, and seller information that gets lost if the base grain is order. Order-item grain preserves that granularity and lets the agent slice by seller, category, and item-level attributes without joining back to raw tables at query time.

The tradeoff: a query for "revenue by state" needs to be aware that one order row would be counted once, while at line grain it is counted multiple times if the order had multiple items. The `line_value` column in the denorm view is defined as `price + freight_value` per line item, and summing it yields correct revenue at any dimension slice.

### Why a single denormalized view instead of a star schema?

The agent's SQL Explorer generates queries via LLM. Constraining the LLM to one view with 31 columns is significantly easier than having it choose the right join across 9 tables every time. The denorm view materialization runs once at data setup and is fast on DuckDB.

For a production system serving many concurrent users, this tradeoff would reverse: joins on demand are cheaper than storing the denormalized view. For a portfolio project where the read pattern is one-user, one-query, materialization wins on both simplicity and query time.

### Why like-for-like windows only?

The Olist dataset has clean data from January 2017 through August 2018. September-October 2018 is truncated (order-status transitions never completed for many orders). Comparing full-year 2017 to partial-2018 would introduce a window-length confounding effect that swamps whatever real signal the agent is trying to find.

The Planner's prompt explicitly instructs like-for-like comparisons (H1 vs H1, Q2 vs Q2) and forbids windows ending in September or October 2018. All 20 labeled eval cases use like-for-like windows.

---

## Agent architecture

### Why LangGraph instead of a custom state machine or LangChain agents?

Three requirements drove this:

1. **State reducers.** The Hypothesis Former needs to emit both new hypotheses AND updates to existing ones in a single structured output. LangGraph's custom reducer pattern lets us define `upsert_by_id` once and have it work for hypotheses and evidence.
2. **Conditional edges.** The Router decides continue/conclude/abort as a real routing choice, not just a node that returns state. LangGraph's conditional edge API makes this explicit.
3. **Streaming updates.** The Streamlit UI's live per-node progress relies on LangGraph's `.stream()` yielding node-completion events. Custom state machines could do this but would require more plumbing.

LangChain agents were considered but rejected: they are optimized for tool-use loops, not for the kind of state-heavy investigation with cross-node evidence linking that this project needed.

### Why the Router as a node instead of a conditional edge?

A conditional edge is a pure function of state. The Router makes an LLM call before deciding, which is not a pure function. Modeling it as a node lets us log the Router's reasoning and cost, keep it inside the trace, and treat routing decisions as first-class events auditable from the JSON trace.

The tradeoff: every hop now includes one additional LLM call. On Llama 3.3 70B this is ~500 tokens in, ~100 tokens out per hop. Roughly $0.0005 per routing decision. Worth it for auditability.

### Why 6 nodes instead of fewer?

Each node does one job and can be tested in isolation. Merging Planner and SQL Explorer would save one LLM call but couple two distinct responsibilities. The eval harness benefits from having per-node trace records so failure mode analysis can pinpoint which node lost the plot.

---

## LLM and structured output

### Why Llama 3.3 70B via Groq instead of GPT-4 or Claude?

Free tier is the main reason. Groq gives 100K tokens per day on their generous free tier for developers, which is enough for 5-10 full investigations per day. GPT-4 and Claude at similar quality cost real money. For a portfolio project that gets shared with recruiters, I want to be sure there is no cost barrier to reviewers running it themselves.

Structured output quality on Llama 3.3 70B is meaningfully better than on 3.1 8B (the smaller free-tier option). During development I considered switching to 8B for cheaper iteration but the JSON validation errors were frequent enough that testing on 8B would have introduced false debugging signals. See the "Model iteration during development" section below.

### Why with_structured_output instead of prompted JSON?

`with_structured_output` uses Groq's constrained decoding (also known as JSON mode), which forces the model to emit output matching the target Pydantic schema exactly. Prompted JSON works about 90% of the time; constrained decoding hits nearly 100%. The 10% difference matters when the agent is running against 20 eval cases automated.

The tradeoff: some models handle constrained decoding better than others. Groq's Llama 3.3 70B implementation is reliable in my testing.

### Why not use LangChain's `.with_structured_output()`? It IS LangChain's implementation.

I use it because it works, and Groq's implementation via LangChain has been solid throughout Phase 3-8.

---

## Guardrails

### Why four separate guardrails instead of one big check?

The four guardrails (SQL safety, ambiguity, confidence cap, cost/token budget) operate at different points in the graph:

- SQL safety runs before every SQL execution
- Ambiguity check runs once, right after the Planner
- Confidence cap runs after the Hypothesis Former, before storing hypotheses
- Cost budget runs at the Router, gating whether to loop or conclude

Combining them would mean putting all guardrail logic in one place, which either bloats the Router or forces every node to know about every guardrail. Distributing them means each guardrail has one job and can be tested independently. Confidence cap has 8 unit tests. SQL safety has 8 unit tests. They don't share code.

### Why a confidence cap by evidence count?

The Hypothesis Former can produce hypotheses at any confidence level. Left unchecked, it often over-commits: a single supporting evidence at 0.95 confidence is not defensible calibration.

The cap curve is `1 evidence -> max 0.60, 2 -> max 0.75, 3 -> max 0.85, 4+ -> max 0.95`. This is a manual calibration curve, not a Bayesian posterior. It reflects the practitioner's intuition that a single finding rarely warrants high confidence and that most real-world confidence should top out below 1.0 to leave room for unknown unknowns.

The cap only reduces confidence; it never raises it. So if the LLM was already cautious (0.4 on a single evidence), the cap does nothing.

Interviewers ask: "Why not a Bayesian posterior?" Because computing one requires a prior distribution over hypotheses and a likelihood function tying evidence to hypotheses. Neither is available at the level of specificity the Hypothesis Former operates at. Manual calibration is a rough proxy but is transparent, auditable, and cheap.

### Why does the Router default to CONCLUDE on LLM failure, not CONTINUE?

Discovered during development. Initial implementation defaulted to CONTINUE when the LLM call failed. When Groq rate-limited us mid-investigation, the Router failed → CONTINUE → SQL Explorer failed → CONTINUE → Router failed → infinite loop against a rate-limited API. Burned quota trying to fail.

Fixed by defaulting to CONCLUDE. When the routing brain is offline, the safer action is to stop and let the Writer synthesize whatever state has been gathered. The Writer has a fallback path that produces a "no clear cause" brief if there is nothing usable in state.

Interview soundbite: "In an agentic system, the failure mode of the failure handler matters more than most things."

---

## Analytical tools

### Why a whitelist of Python functions instead of LLM-generated code?

An LLM given `exec()` will do dangerous things. Even if you sandbox it, the LLM can execute arbitrary computation that is not auditable at review time. In an interview, "I gave the LLM `exec()`" is a red flag; "I gave it a whitelist" is a green flag.

Three whitelisted tools cover the analytical patterns needed for this project:

- `contribution_analysis`: decomposes a delta into per-dimension contributions
- `top_k_by_dimension`: ranks a single-window aggregate
- `pct_change_summary`: finds outliers by percent change

The LLM picks the tool name and arguments (dimension_col, metric_col, etc). It does not write Python. Adding a new analytical pattern is a code change, not a prompt change.

### Why does `contribution_analysis` reset the DataFrame index?

Learned during Phase 3.5 debugging. Pandas has two indexing modes (positional via `.iloc` and label-based via `.loc`) and if a DataFrame's index is not aligned with the row iteration, you get errors like `cannot do positional indexing on RangeIndex with these indexers [5] of type str`. Resetting the index guarantees positional and label-based indexing are aligned.

Also, defensive programming: `iloc[idx]` is unambiguously positional. Never mix `.iterrows()` and `head(N)` on a DataFrame with a non-integer index.

---

## Memory

### Why ChromaDB with local embeddings instead of a hosted vector DB?

Two reasons:

1. **Zero external dependency.** ChromaDB persists to a local directory. No API keys, no cost, no network. A reviewer can clone the repo and run it end-to-end.
2. **Embedding cost.** `all-MiniLM-L6-v2` is the default ChromaDB embedding model. It runs locally, computing embeddings for 5 seed investigations and every completed investigation. Zero token cost to the LLM budget.

The tradeoff: `all-MiniLM-L6-v2` is a small (23M parameter) model. Its semantic clustering is loose. Queries about "delivery time got worse" cluster near queries about "AOV was 2% lower" because both are numeric-decline framings. For a production system I would swap in OpenAI `text-embedding-3-small` or a domain-fine-tuned model.

### Why seed 5 canonical investigations at startup?

Empty ChromaDB means the first user query has no retrieval hits. Retrieval is one of the more interesting agent behaviors to demo. Seeding 5 investigations covering the top three case archetypes (single-cause growth, single-cause decline, null-case) ensures the retrieval mechanism has content to retrieve on the very first user query.

Honest note in the code: "Production would populate this from real historical investigations. For the demo we seed 5 canonical investigations to make the retrieval mechanism visible."

---

## Evaluation

### Why deterministic scoring instead of LLM-as-judge?

Reproducibility. If I run `python evals/run_evals.py --start 0 --end 20` twice, I want the same scores. LLM-as-judge introduces stochasticity: the judge's confidence and reasoning shift between runs, so scores drift.

Deterministic scoring is also cheaper: no additional LLM calls per case. The score is computed in pure Python from the agent's output and the ground truth.

The tradeoff: deterministic scoring is coarser. A cause that says "customer_state expansion in Sao Paulo" and one that says "SP state growth" would both hit the keyword-match test, but a human reviewer would recognize them as equivalent. Keyword-match with a smart stop-word filter is a rough proxy.

### Why include null cases in the eval set?

Null cases (4 of 20) test that the agent will decline to attribute when the data is broadly distributed. An agent that fabricates causes for random 2% drifts is worse than one that admits uncertainty. The scoring on null cases penalizes forbidden attributions (agent claimed a `must_not_claim` cause with confidence >= 0.5) while giving full credit for empty `ranked_causes` or all-low-confidence outputs.

### Why not test the eval harness against the labeled cases?

I did. `scripts/test_metrics.py` has 10 unit tests covering full-credit, partial-credit, and zero paths for all three scorers. All pass.

---

## What I would do differently

For the next iteration of a project like this:

1. **Test end-to-end with fewer, smaller cases during Phase 3-4.** I built all six nodes and then tested. In hindsight, testing the full graph after each node came online would have caught the Python Analyst's positional-indexing bug earlier than Phase 8.

2. **Design the token budget upfront.** I hit Groq rate limits multiple times mid-development. Estimating "an investigation is 15-25K tokens" and multiplying by expected iterations would have made the daily cap a first-class constraint from Day 1 instead of a surprise on Day 4.

3. **Pick a deployment target that matches the runtime.** Streamlit Community Cloud's Python 3.14 default breaks the chromadb -> tokenizers -> PyO3 chain. If I had known that upfront I would have chosen Hugging Face Spaces (Docker-based, real Python version control) from the start.

4. **Use a domain-fine-tuned embedding model.** `all-MiniLM-L6-v2` was chosen for offline availability. But the retrieval quality on "revenue decline" vs "AOV decline" is loose enough that a small fine-tune on 100 real investigations would meaningfully improve semantic clustering.

5. **Cache the DuckDB DataFrame** between the SQL Explorer's execution and the Python Analyst's re-execution. Right now the analyst re-executes the same SQL to get the DataFrame. Fine for a demo. In production, pass the DataFrame directly.

---

## Observed weaknesses in agent behavior

These are known, documented for future prompt-engineering iteration:

- **Hypothesis Former describes rather than causally frames.** Statements tend to read like "growth was driven by X category increasing" (descriptive) rather than "X category increased because Y mechanism" (causal). Tightening the prompt to require a mechanism after the dimension identification would help.
- **SQL Explorer occasionally repeats slices.** Prompt tells it not to; it sometimes ignores.
- **Router sometimes concludes at confidence 0.7 when it could keep going.** The threshold is set aggressive to preserve token budget. A production system might raise this to 0.85 for higher-stakes decisions.
- **Streamlit Cloud deployment blocked.** See `docs/design_decisions.md` under "What I would do differently."

---

## Model iteration during development

Considered switching to Llama 3.1 8B during development to conserve tokens on Groq's rate-limited free tier. Would have quadrupled the tokens-per-day available. But 8B's structured-output reliability is meaningfully weaker: frequent JSON validation errors that would introduce false debugging signals.

Instead, deferred end-to-end testing during Phase 4-7 in favor of import-only syntax checks and pure-Python unit tests. Integration testing happens at Phase 8 evaluation, where token consumption is inherent to the phase's purpose. This is a tradeoff between "cheap iteration" and "accurate testing signal." I chose the latter.

The GROQ_MODEL env var makes model switching a one-line config change, so a future maintainer can revisit this.

---

## Attribution and acknowledgments

Built by Charmy Raj. See top-level README.md for links.

Olist dataset: [Kaggle - Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), CC-BY-NC-SA-4.0.