# MS² Spectral Search REST API

Fast MS/MS (MS²) spectral search over local **MSP** libraries with Flask, MatchMS, and **Flash Entropy Search**.

## Features
- MSP library loading with caching
- Flash Entropy and Cosine scoring
- Batch queries
- Lightweight usage logging

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install Flask matchms ms_entropy requests
mkdir -p libraries logs
python server.py
```

Visit: http://localhost:5000
