# RAGForge — Production Retrieval-Augmented AI System

A Python project scaffold for building a GPU-accelerated RAG system with FastAPI, FAISS, OpenAI, and sentence-transformers.

## Current Structure

- `app/` — application code (services, utils, config, entrypoint)
- `data/` — local data artifacts (gitignored)
- `index/` — local vector index artifacts (gitignored)
- `requirements-gpu.txt` — pinned GPU-ready dependencies
- `gpu_check.py` — verifies PyTorch + FAISS GPU visibility

## Prerequisites

- Linux with NVIDIA GPU drivers installed
- Python 3.11 available (`python3.11`)
- Optional (recommended on hybrid graphics laptops): `prime-run`

## Setup (GPU)

1. Create virtual environment:

   ```bash
   python3.11 -m venv .venv
   ```

2. Install dependencies:

   ```bash
   .venv/bin/python -m pip install --upgrade pip
   .venv/bin/python -m pip install -r requirements-gpu.txt
   ```

3. Verify GPU access:

   ```bash
   prime-run .venv/bin/python gpu_check.py
   ```

   If `prime-run` is unavailable, run:

   ```bash
   .venv/bin/python gpu_check.py
   ```

## Installed Core Stack

- FastAPI + Uvicorn
- OpenAI SDK
- tiktoken
- pydantic
- python-dotenv
- sentence-transformers (uses CUDA-enabled PyTorch)
- FAISS GPU (`faiss-gpu-cu12`)

## Notes

- The source files under `app/` are currently scaffolded and ready for implementation.
- `data/` and `index/` are intentionally ignored from git for local/large artifacts.
# RAGForge-Production-Retrieval-Augmented-AI-System
