# RAG Evaluation Reports

This directory contains benchmark evaluation results.

## Files

- `latest.json` - Most recent benchmark results in JSON format
- `latest.md` - Most recent human-readable report in Markdown
- `comparison.json` - Comparison against baseline metrics
- `summary.json` - Compact JSON summary for programmatic use

## Workflow

1. Run benchmark: `python scripts/run_benchmark.py`
2. Compare results: `python scripts/compare_metrics.py`
3. Generate report: `python scripts/generate_report.py`

## CI/CD

Results are automatically generated on:
- Push to main/develop branches
- Pull requests
- Manual trigger via GitHub Actions

## Baseline

Baseline metrics are stored in `../baselines/baseline_metrics.json` and version-controlled.

To establish a new baseline after significant changes:
```bash
python scripts/run_benchmark.py --save-baseline
git add evaluations/baselines/baseline_metrics.json
git commit -m "docs: update benchmark baseline"
```
