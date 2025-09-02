# Usage Guide

## API Endpoints

### `GET /health`
Check server health.

### `GET /libraries`
List available libraries.

### `POST /search`
Search a single spectrum.

Example:
```json
{
  "library": "GNPS.msp",
  "search_method": "flash_entropy",
  "precursor_mz": 315.1234,
  "peaks": [[50.0, 100.0], [77.039, 250.0]]
}
```

### `POST /batch_search`
Submit multiple spectra.

Example:
```json
[
  {"library": "GNPS.msp", "precursor_mz": 315.1234, "peaks": [[50.0, 100.0]]},
  {"library": "nist.msp", "precursor_mz": 250.4567, "peaks": [[45.0, 80.0]]}
]
```

## Logging
Logs saved in `logs/` for request usage tracking.

## Testing
Run example client:
```bash
python testing.py
```
