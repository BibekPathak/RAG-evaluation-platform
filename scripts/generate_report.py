#!/usr/bin/env python3
"""
Benchmark Report Generator

Generates human-readable markdown reports from benchmark results.

Usage:
    python scripts/generate_report.py --results evaluations/reports/latest.json --output evaluations/reports/latest.md
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


def load_json(path: str) -> Dict:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def format_metric_value(value: float, metric_type: str = "percentage") -> str:
    """Format metric value for display."""
    if metric_type == "percentage":
        return f"{value:.1f}%"
    elif metric_type == "latency":
        return f"{value:.0f}ms"
    elif metric_type == "cost":
        return f"${value:.6f}"
    return f"{value:.4f}"


def generate_markdown_report(results: Dict, comparison: Dict = None) -> str:
    """Generate markdown report from benchmark results."""
    lines = []

    lines.append("# RAG Evaluation Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    lines.append("")

    if comparison:
        lines.append(f"**Status:** {'✅ PASS' if comparison['passed'] else '❌ FAIL'}")
        lines.append(f"**Comparison Baseline:** {comparison.get('baseline_created', 'Unknown')[:10]}")
        lines.append("")
        lines.append("## Comparison Summary")
        lines.append("")
        lines.append(f"- Total Metrics: {comparison['total_metrics']}")
        lines.append(f"- Passed: {comparison['passed_metrics']}")
        lines.append(f"- Failed: {comparison['failed_metrics']}")
        lines.append(f"- Improved: {comparison['improved_metrics']}")
        lines.append(f"- Degraded: {comparison['degraded_metrics']}")
        lines.append("")

    lines.append("## Configuration")
    lines.append("")
    config = results.get("config", {})
    lines.append(f"- Embedding Model: `{config.get('embedding_model', 'N/A')}`")
    lines.append(f"- LLM: `{config.get('llm', 'N/A')}`")
    lines.append(f"- Retriever: `{config.get('retriever', 'N/A')}`")
    lines.append("")

    datasets = results.get("datasets", {})
    lines.append("## Evaluation Coverage")
    lines.append("")
    lines.append(f"- Total Questions: {datasets.get('total_questions', 'N/A')}")
    lines.append(f"- Retrieval Evaluated: {datasets.get('evaluated_retrieval', 'N/A')}")
    lines.append(f"- Generation Evaluated: {datasets.get('evaluated_generation', 'N/A')}")
    lines.append("")

    metrics = results.get("metrics", {})

    lines.append("## Retrieval Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")

    retrieval_metrics = [
        ("context_recall", "Context Recall"),
        ("context_precision", "Context Precision"),
        ("mrr", "MRR"),
        ("ndcg", "NDCG"),
        ("hit_rate", "Hit Rate")
    ]

    for metric_id, metric_name in retrieval_metrics:
        if metric_id in metrics:
            value = format_metric_value(metrics[metric_id], "percentage")
            lines.append(f"| {metric_name} | {value} |")

    lines.append("")

    lines.append("## Generation Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")

    generation_metrics = [
        ("faithfulness", "Faithfulness"),
        ("answer_relevancy", "Answer Relevancy"),
        ("context_utilization", "Context Utilization"),
        ("hallucination_rate", "Hallucination Rate")
    ]

    for metric_id, metric_name in generation_metrics:
        if metric_id in metrics:
            value = format_metric_value(metrics[metric_id], "percentage")
            lines.append(f"| {metric_name} | {value} |")

    lines.append("")

    perf = results.get("performance", {})
    lines.append("## Performance")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")

    if "total_latency_ms" in perf:
        lines.append(f"| Total Latency | {format_metric_value(perf['total_latency_ms'], 'latency')} |")
    if "avg_latency_ms" in perf:
        lines.append(f"| Avg Latency | {format_metric_value(perf['avg_latency_ms'], 'latency')} |")
    if "estimated_cost_usd" in perf:
        lines.append(f"| Estimated Cost | {format_metric_value(perf['estimated_cost_usd'], 'cost')} |")

    lines.append("")

    if comparison and comparison.get("results"):
        lines.append("## Detailed Comparison")
        lines.append("")
        lines.append("| Metric | Baseline | Current | Change | Status |")
        lines.append("|--------|----------|---------|--------|--------|")

        status_icons = {
            "improved": "✅",
            "degraded": "❌",
            "below_threshold": "🔴",
            "above_threshold": "🔴",
            "maintained": "⚪"
        }

        for result in comparison["results"]:
            baseline_str = f"{result['baseline']:.1%}"
            current_str = f"{result['current']:.1%}"

            change_str = f"{result['change_pct']:+.1f}%"
            if abs(result['change_pct']) == float('inf'):
                change_str = "N/A"

            status = status_icons.get(result["status"], "⚪")

            lines.append(f"| {result['metric'].replace('_', ' ').title()} | {baseline_str} | {current_str} | {change_str} | {status} |")

        lines.append("")

    lines.append("---")
    lines.append("*Generated by RAG Evaluation Platform*")

    return "\n".join(lines)


def generate_json_summary(results: Dict, comparison: Dict = None) -> Dict[str, Any]:
    """Generate compact JSON summary for programmatic use."""
    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": results.get("version", "1.0.0"),
        "metrics": results.get("metrics", {}),
        "performance": results.get("performance", {}),
        "passed": comparison.get("passed", True) if comparison else True
    }

    if comparison:
        summary["comparison"] = {
            "baseline_created": comparison.get("baseline_created"),
            "total_metrics": comparison["total_metrics"],
            "passed_metrics": comparison["passed_metrics"],
            "failed_metrics": comparison["failed_metrics"],
            "status": comparison["summary"]["overall_status"]
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to benchmark results JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluations/reports/latest.md",
        help="Path to output markdown file"
    )
    parser.add_argument(
        "--comparison",
        type=str,
        help="Optional path to comparison results JSON"
    )
    parser.add_argument(
        "--json-summary",
        type=str,
        help="Optional path to save JSON summary"
    )

    args = parser.parse_args()

    try:
        results = load_json(args.results)
    except FileNotFoundError:
        print(f"Error: Results file not found: {args.results}")
        return 1

    comparison = None
    if args.comparison:
        try:
            comparison = load_json(args.comparison)
        except FileNotFoundError:
            print(f"Warning: Comparison file not found: {args.comparison}")

    markdown_report = generate_markdown_report(results, comparison)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(markdown_report)

    print(f"Markdown report saved to: {output_path}")

    if args.json_summary:
        summary = generate_json_summary(results, comparison)
        json_path = Path(args.json_summary)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"JSON summary saved to: {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
