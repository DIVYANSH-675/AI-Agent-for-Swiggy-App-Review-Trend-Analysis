# Swiggy Review Trends

## Project Overview

This project builds an AI system that analyzes Google Play Store reviews for **Swiggy** (a popular food delivery app in India) and creates daily trend reports. The system automatically finds issues, requests, and feedback from thousands of reviews and tracks how they change over time.

> ⚠️ **Important Disclaimer**
>
> **API KEY NOTICE:** The API keys used in original example code are **REVOKED** and will not work. You **must** use your own OpenAI or compatible API key to run this project.

---

## Quick summary

* Input: Google Play reviews for Swiggy (June 2024 → present)
* Output: Daily topic labels and 30-day trend reports (CSV + interactive HTML)
* Approach: Use an LLM-based topic router + embeddings for canonicalization + efficient daily batch processing

---

## Quick Start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Set API credentials (choose one)

**OpenAI**

```bash
export OPENAI_API_KEY="sk-your-openai-key-here"
```

**Megallm-compatible API**

```bash
export MEGALLM_API_KEY="sk-mega-your-key-here"
export MEGALLM_BASE_URL="https://ai.megallm.io/v1"
```

### 3) Run the pipeline (notebook)

```bash
jupyter notebook notebooks/00_complete_pipeline.ipynb
```

The notebooks will:

* load and clean reviews
* run the LLM topic router
* canonicalize topics using embeddings
* produce CSV + HTML trend reports in `output/`

---

## Project directory (renamed from original to avoid "assignment" label)

```
swiggy_review_trends/
├── notebooks/                          # Main notebooks (for cloud API usage)
│   ├── 00_complete_pipeline.ipynb      # Runs entire pipeline
│   ├── 01_setup_and_clean.ipynb        # Data cleaning
│   ├── 02_topic_router.ipynb           # Topic detection
│   ├── 04_topic_canonicalization.ipynb # Merge similar topics
│   ├── 05_trend_analysis.ipynb         # Create reports
│   ├── cache.db                        # API response cache (sqlite)
│   └── utils/
│       └── llm_client.py               # LLM client used in notebooks
│
├── Notebook2/                          # Local LLM notebooks (Ollama)
│   ├── 00_complete_pipeline.ipynb
│   └── utils/
│       └── llm_client.py
│
├── data/                               # Processed data and daily batches
│   ├── reviews_clean.parquet
│   ├── labels_initial.parquet
│   ├── novel_topic_summary.parquet
│   ├── daily_batches/
│   │   └── reviews_YYYY-MM-DD.parquet
│   └── daily_labels/
│       └── labels_YYYY-MM-DD.parquet
│
├── output/                             # Final reports
│   ├── topics_trend_2025-10-28.csv
│   ├── topics_trend_2025-10-28.html
│   └── topics_trend_2025-10-28_debug.csv
│
├── registry/
│   └── topic_registry.json             # Topic definitions
│
├── utils/
│   └── llm_client.py                   # Shared utilities for API calls
│
├── swiggy_scraped.csv                  # Raw review data (if present)
├── requirements.txt                    # Python deps
├── README.md                           # This high-level file
└── venv/
```

---

## What this project does (unchanged content style)

1. Review Processing Pipeline — cleans and processes hundreds of thousands of reviews
2. AI Topic Detection — uses an LLM to identify topic labels for each review
3. Novel Topic Discovery — flags reviews that map to no known topic and summarizes discovered novel topics
4. Topic Deduplication — merges semantically similar topics (embedding similarity thresholding)
5. 30-Day Trend Reports — counts daily occurrences and produces CSV + HTML visualizations

---

## Topics covered by the system

The system tracks 32 topics across categories (exact labels preserved):

**Logistics** (6 topics):

* Order Incomplete
* ETA Jump After Payment
* Late Delivery
* Can't Find Address
* No Cancel Option
* No Delivery Yet

**Food** (4 topics):

* Stale Food
* Wrong Item
* Packaging Leak
* Poor Quality

**Pricing** (3 topics):

* High Fees
* Surge Pricing
* Overpriced Items

**Support** (3 topics):

* Bot Only No Human Support
* Refund Friction
* No Response to Complaint

**App/Payments** (6 topics):

* Payment Failure
* Login Bug
* Cart Bug
* OTP Issue
* COD Not Available
* Money Already Eaten

**Partner** (2 topics):

* Rude Delivery Person
* Unprofessional Behavior

**Merch** (2 topics):

* Out of Stock
* Limited Options

**Sentiment** (6 topics):

* Positive Experience
* Negative Generic
* Fast Delivery
* Good Quality Food
* Very Good Service
* Great App

---

## Output format (sample)

CSV columns: `topic_id,YYYY-MM-DD,...,YYYY-MM-DD,7d_change_pct`

Sample CSV snippet:

```csv
topic_id,2025-09-28,2025-09-29,...,2025-10-25,7d_change_pct
POSITIVE_EXPERIENCE,412,219,...,228,-5.41%
LATE_DELIVERY,60,55,...,43,-10.17%
MONEY_ALREADY_EATEN,17,15,...,16,9.76%
```

---

## Technical stack (same as original)

* polars for data processing
* duckdb for analytics
* openai / compatible API for LLM calls
* sentence-transformers for embeddings
* jupyter notebooks for orchestration

Recommended versions are included in `requirements.txt` below.

---

## Example files (copy-paste ready)

### `requirements.txt`

```text
polars>=0.20.0
duckdb>=0.10.0
openai>=1.10.0
sentence-transformers>=2.5.0
jupyter>=1.1.0
tqdm>=4.66.0
python-dotenv>=1.0.0
requests>=2.31.0
pandas>=2.0.0
plotly>=5.0.0
```

---

### `utils/llm_client.py` (example; adapt to your API)

```python
# utils/llm_client.py
import os
import time
import json
from typing import Any, Dict, Optional

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
MEGALLM_KEY = os.getenv("MEGALLM_API_KEY")
MEGALLM_BASE = os.getenv("MEGALLM_BASE_URL")

# Minimal client wrapper that chooses provider by env variables
class LLMClient:
    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or ("megallm" if MEGALLM_KEY and MEGALLM_BASE else "openai")

    def call(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.provider == "openai":
            return self._call_openai(payload)
        else:
            return self._call_megallm(payload)

    def _call_openai(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import openai
        except ImportError:
            raise RuntimeError("openai package not installed")
        if not OPENAI_KEY:
            raise RuntimeError("OPENAI_API_KEY not set")
        openai.api_key = OPENAI_KEY
        # example payload handling - adapt to your prompt / model choices
        response = openai.ChatCompletion.create(
            model=payload.get("model", "gpt-4o-mini"),
            messages=payload.get("messages", []),
            temperature=payload.get("temperature", 0.0),
            max_tokens=payload.get("max_tokens", 512),
        )
        return response

    def _call_megallm(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        import requests
        if not MEGALLM_KEY or not MEGALLM_BASE:
            raise RuntimeError("MEGALLM_API_KEY or MEGALLM_BASE_URL not set")
        url = MEGALLM_BASE.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {MEGALLM_KEY}", "Content-Type": "application/json"}
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()

# Basic usage:
# client = LLMClient()
# out = client.call({"messages": [{"role":"user","content":"Hello"}]})
```

---

### `scripts/pipeline_main.py` (skeleton)

```python
# scripts/pipeline_main.py
import os
from utils.llm_client import LLMClient
import polars as pl

def run_daily_batch(date_str: str):
    """Load daily parquet, run topic router, save labels"""
    # 1) load reviews
    in_path = f"data/daily_batches/reviews_{date_str}.parquet"
    df = pl.read_parquet(in_path)

    # 2) prepare payloads and send to LLM client (batching is recommended)
    client = LLMClient()
    # Example per-row labeling - use batching in real runs
    labels = []
    for row in df.iter_rows(named=True):
        payload = {
            "messages": [{"role":"user", "content": f"Label this review: {row['review_text']}"}],
            "model": "gpt-4o-mini"
        }
        resp = client.call(payload)
        labels.append({"review_id": row['review_id'], "labels": resp})

    # 3) save labels to parquet
    out_path = f"data/daily_labels/labels_{date_str}.parquet"
    # convert and write - example only
    # pl.from_dicts(labels).write_parquet(out_path)

if __name__ == '__main__':
    import sys
    date_str = sys.argv[1]
    run_daily_batch(date_str)
```

---

## Troubleshooting (unchanged)

**API Key Issues:**

```bash
echo $MEGALLM_API_KEY
echo $OPENAI_API_KEY
```

If you receive `401` or `Invalid API Key` errors, ensure you set environment variables correctly and that your key is valid.

**Cache Problems:**

* Remove `notebooks/cache.db` if responses are stale

**Import Errors in notebooks:**

```python
import sys
sys.path.append('../')
```

---

## Development timeline

* Development window used earlier: 24 hours (kept as a note)

---

## Files generated by runs (example)

* `data/reviews_clean.parquet` — final cleaned reviews
* `data/labels_initial.parquet` — labelled reviews
* `output/topics_trend_YYYY-MM-DD.csv` — trend CSV
* `output/topics_trend_YYYY-MM-DD.html` — interactive report

---

## Contact / Attribution

If you have questions or want to collaborate, contact: **[divyanshgupta0704@gmail.com](mailto:divyanshgupta0704@gmail.com)**

---


<!-- You can extend this document by adding full notebook JSONs or more scripts. This file is intended as a single copy-paste-ready markdown containing the main README plus key files for quick public release. -->
