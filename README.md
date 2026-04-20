# IPL Analytics Assistant

## Hybrid LLM + Structured Data Intelligence System

---

## Overview

This project implements a domain-specific AI system for IPL cricket analytics using a hybrid architecture.

Instead of relying purely on an LLM, the system combines:

- Structured data retrieval  
- Deterministic metric computation  
- A fine-tuned language model (Mistral-7B + LoRA)  

The goal is to eliminate **numerical hallucination** and ensure reliable, data-grounded answers.

---

## Core Idea

Traditional LLMs are weak at numbers.

This system fixes that by separating responsibilities:

- Structured data → retrieval  
- Deterministic logic → computation  
- LLM → explanation only  

This ensures every answer is **factually correct and interpretable**.

---

## Key Features

- Hybrid query pipeline (route → retrieve → compute → generate)  
- Intent classification and entity extraction  
- Structured IPL dataset querying (batting, bowling, matchups, venues)  
- Deterministic metric computation layer  
- Fine-tuned LLM for explanations  
- FastAPI backend with debugging endpoints  
- Streamlit UI for interaction  
- Guardrails to prevent incorrect comparisons  

---

## System Architecture

```
User Query
→ Query Router (intent + entities)
→ Structured Retriever / Insight Retriever
→ Deterministic Metrics Layer
→ Context Builder
→ Fine-tuned Generator (Mistral + LoRA)
→ Answer
```

---

## Deterministic Metrics

All metrics are computed before reaching the model:

- Strike rate  
- Dot-ball percentage  
- Scoring-ball percentage  
- Boundary percentage  
- Boundary frequency  
- Runs per dismissal  
- Balls per dismissal  

The LLM never calculates numbers.

---

## Project Structure

```
ipl_project/
├── app/
│   ├── api/
│   ├── services/
│   ├── scripts/
│   ├── ui/
│   └── requirements.txt
├── data/
│   └── analytics/
├── finetuning/
│   ├── configs/
│   ├── scripts/
│   └── outputs/
├── README.md
└── .gitignore
```

---

## Setup

### Install dependencies

```
pip install -r app/requirements.txt
```

### Create virtual environment (optional)

```
python -m venv .venv
.venv\Scripts\activate
```

---

## Run the Project

### Start Backend (FastAPI)

```
python -m uvicorn app.api.main:app --reload --port 8000
```

---

### Start UI (Streamlit)

```
streamlit run app/ui/streamlit_app.py
```

---

## API Endpoints

### Health Check

```
GET /health
```

---

### Chat Endpoint

Runs full pipeline:

```
POST /chat
```

Example:

```
{
  "query": "Compare JJ Bumrah and YS Chahal"
}
```

---

### Debug Endpoint (No LLM)

```
POST /debug/retrieve
```

Returns:
- route  
- intent  
- entities  
- retrieved data  
- constructed context  

---

## Dataset

Structured IPL analytics stored in Parquet format:

- Batter career and season stats  
- Bowler career and season stats  
- Batter vs bowler matchups  
- Venue statistics  
- Phase-wise scoring data  
- Insight summaries  

---

## Engineering Highlights

- Hybrid AI system combining LLM + structured data  
- Deterministic computation eliminates hallucinations  
- LoRA fine-tuning for domain-specific responses  
- Modular architecture with clear separation of concerns  
- Fully debuggable pipeline  

---

## Summary

This project demonstrates how to build reliable AI systems by combining structured data processing with language models instead of relying on generation alone.
