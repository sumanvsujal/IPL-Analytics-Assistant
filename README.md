# IPL Analytics Assistant

Explore historical IPL player records, leaderboards, matchups, venues, and seasons with a FastAPI API and Streamlit interface.

The default demo runs on a CPU with **no model downloads, GPU, API keys, or paid inference**. It routes questions to bundled Parquet tables, computes metrics, and presents factual summaries with source labels. Metric questions include built-in explanations. Insights use local keyword search. These are deterministic analytics answers, not generated Mistral responses.

## Run the demo

Use Python 3.11 or 3.12. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/ui/streamlit_app.py
```

The UI runs the analytics in the same process. No separate API server is necessary. Try:

- Who are the top IPL run scorers?
- Compare Bumrah and Chahal.
- V Kohli vs JJ Bumrah head-to-head
- How did IPL 2024 compare to 2023?
- What does strike rate mean?

Answers cover the bundled historical data, not live scores. Unknown questions receive a limitation message. Comparisons report recorded statistics without promising future outcomes.

## Share on a free hosting tier

For Streamlit Community Cloud, select this GitHub repository, choose `main` (after your changes are pushed), and set the entrypoint to `app/ui/streamlit_app.py`. Use Python 3.12. The root `requirements.txt` installs only the CPU demo dependencies. Leave `API_URL`, `HF_API_KEY`, and model settings unset; no secrets are required.

This supports a free demo deployment subject to the hosting service's current availability, resource limits, and sleep policy. Deployment has not been performed by these code changes.

## Optional separate API

```bash
python -m uvicorn app.api.main:app --port 8000
```

Set `API_URL` in the UI environment only when connecting it to a separately running backend. Without it, the UI works independently.

- `GET /health`: data counts and `answer_mode` (`analytics` by default).
- `POST /chat`: `{"query":"Compare Bumrah and Chahal."}` returns an answer and retrieval counts.
- `POST /debug/retrieve`: same request, returning structured facts and context.

## Validate

```bash
python -m unittest discover -s tests -v
python -m app.scripts.test_answer_quality
```

The tests assert offline operation, source statistics, metric explanations, unknown-question behavior, semantic-search fallback, and in-process UI calls. The second command runs the 14 representative questions and prints their answers; it is a demonstration runner, not an assertion suite.

## Optional model experiments

The original Mistral/LoRA generator remains available for separate experiments:

```bash
pip install -r app/model_requirements.txt
ANSWER_MODE=llm LOAD_MODEL=true python -m uvicorn app.api.main:app --port 8000
```

Local quantized generation needs compatible hardware and model access. Adapter tokenizer/config files are present, but adapter weights are absent from this checkout. The existing Hugging Face generation integration uses a legacy endpoint and is not verified; the free demo does not depend on it.

Semantic insight search is also optional. Install the model dependencies, build the index with `python -m app.services.insight_retriever --build`, and set `INSIGHT_SEARCH=semantic`. This downloads an embedding model and requires access to Hugging Face. Missing model/index files fall back to keyword search. Default keyword mode never attempts a download.
