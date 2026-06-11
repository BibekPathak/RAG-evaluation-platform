#!/usr/bin/env python3
"""
Benchmark Metrics Comparator

Compares current benchmark results against baseline metrics.

Usage:
    python scripts/compare_metrics.py --current evaluations/reports/latest.json --baseline evaluations/baselines/baseline_metrics.json
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple


def load_json(path: str) -> Dict:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values."""
    if old_value == 0:
        return float('inf') if new_value != 0 else 0.0
    return ((new_value - old_value) / old_value) * 100


def compare_metric(
    metric_name: str,
    baseline_value: float,
    current_value: float,
    threshold: Dict = None
) -> Dict[str, Any]:
    """Compare a single metric against baseline and thresholds."""
    change_pct = calculate_percentage_change(baseline_value, current_value)

    result = {
        "metric": metric_name,
        "baseline": baseline_value,
        "current": current_value,
        "change_pct": change_pct,
        "passed": True,
        "status": "improved",
        "message": ""
    }

    if threshold:
        if "min" in threshold and current_value < threshold["min"]:
            result["passed"] = False
            result["status"] = "below_threshold"
            result["message"] = f"Below minimum threshold of {threshold['min']}"
        elif "max" in threshold and current_value > threshold["max"]:
            result["passed"] = False
            result["status"] = "above_threshold"
            result["message"] = f"Above maximum threshold of {threshold['max']}"

    if abs(change_pct) > 3 and change_pct < 0:
        result["passed"] = False
        result["status"] = "degraded"
        result["message"] = f"Degraded by {abs(change_pct):.1f}% (exceeds 3% threshold)"

    return result


def compare_metrics(
    current_results: Dict,
    baseline: Dict,
    thresholds: Dict = None,
    max_degradation_pct: float = 3.0
) -> Dict[str, Any]:
    """Compare current results against baseline."""
    comparison = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "baseline_version": baseline.get("version", "unknown"),
        "baseline_created": baseline.get("created_at", "unknown"),
        "passed": True,
        "total_metrics": 0,
        "passed_metrics": 0,
        "failed_metrics": 0,
        "improved_metrics": 0,
        "degraded_metrics": 0,
        "results": [],
        "summary": {
            "overall_status": "PASS"
        }
    }

    baseline_metrics = baseline.get("metrics", baseline)
    current_metrics = current_results.get("metrics", current_results)

    metric_order = [
        "context_recall",
        "context_precision",
        "mrr",
        "ndcg",
        "hit_rate",
        "faithfulness",
        "answer_relevancy",
        "context_utilization",
        "hallucination_rate"
    ]

    for metric in metric_order:
        if metric not in current_metrics:
            continue

        baseline_value = baseline_metrics.get(metric, 0.0)
        current_value = current_metrics.get(metric, 0.0)
        threshold = thresholds.get(metric) if thresholds else None

        result = compare_metric(metric, baseline_value, current_value, threshold)
        comparison["results"].append(result)
        comparison["total_metrics"] += 1

        if result["passed"]:
            comparison["passed_metrics"] += 1
        else:
            comparison["failed_metrics"] += 1
            comparison["passed"] = False

        if result["change_pct"] > 0:
            comparison["improved_metrics"] += 1
        elif result["change_pct"] < -max_degradation_pct:
            comparison["degraded_metrics"] += 1

    if comparison["failed_metrics"] > 0:
        comparison["summary"]["overall_status"] = "FAIL"
    elif comparison["improved_metrics"] > comparison["degraded_metrics"]:
        comparison["summary"]["overall_status"] = "PASS"
    else:
        comparison["summary"]["overall_status"] = "MARGINAL"

    return comparison


def format_comparison_table(comparison: Dict) -> str:
    """Format comparison results as markdown table."""
    lines = []
    lines.append("## RAG Evaluation Results")
    lines.append("")
    lines.append("| Metric | Baseline | Current | Change | Status |")
    lines.append("|--------|---------|--------|-------|--------|")

    status_icons = {
        "improved": "✅",
        "degraded": "❌",
        "below_threshold": "🔴",
        "above_threshold": "🔴",
        "maintained": "⚪"
    }

    for result in comparison["results"]:
        baseline_str = f"{result['baseline']:.1%}" if isinstance(result['baseline'], (int, float)) else "N/A"
        current_str = f"{result['current']:.1%}" if isinstance(result['current'], (int, float)) else "N/A"

        change_str = f"{result['change_pct']:+.1f}%"
        if abs(result['change_pct']) == float('inf'):
            change_str = "N/A"

        status = status_icons.get(result["status"], "⚪")

        lines.append(f"| {result['metric'].replace('_', ' ').title()} | {baseline_str} | {current_str} | {change_str} | {status} |")

    lines.append("")
    lines.append(f"**Result: {comparison['summary']['overall_status']}**")
    lines.append(f"({comparison['passed_metrics']}/{comparison['total_metrics']} metrics passed)")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark metrics against baseline")
    parser.add_argument(
        "--current",
        type=str,
        required=True,
        help="Path to current benchmark results JSON"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to baseline metrics JSON"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="benchmark_config.json",
        help="Path to benchmark configuration (for thresholds)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save comparison results JSON"
    )

    args = parser.parse_args()

    try:
        current_results = load_json(args.current)
    except FileNotFoundError:
        print(f"Error: Current results file not found: {args.current}")
        return 1

    try:
        baseline = load_json(args.baseline)
    except FileNotFoundError:
        print(f"Error: Baseline file not found: {args.baseline}")
        return 1

    thresholds = None
    if Path(args.config).exists():
        config = load_json(args.config)
        thresholds = config.get("thresholds", {})

    comparison = compare_metrics(current_results, baseline, thresholds)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPARISON")
    print("=" * 60)
    print(f"\nBaseline: {comparison['baseline_created']}")
    print(f"Compared: {comparison['timestamp']}")
    print(f"\nOverall Result: {comparison['summary']['overall_status']}")
    print(f"Metrics: {comparison['passed_metrics']}/{comparison['total_metrics']} passed")
    print(f"Improved: {comparison['improved_metrics']}")
    print(f"Degraded: {comparison['degraded_metrics']}")

    print("\n" + format_comparison_table(comparison))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to: {output_path}")

    return 0 if comparison["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
