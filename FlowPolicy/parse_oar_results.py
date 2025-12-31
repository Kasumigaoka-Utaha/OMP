#!/usr/bin/env python3
"""Parse OAR stdout logs and export task top-5 mean scores to an Excel-friendly file."""

from __future__ import annotations

import argparse
import html
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple


START_PATTERN = re.compile(r"开始运行环境:\s*(.+)")
COMMAND_PATTERN = re.compile(r"执行命令:\s*bash\s+\S*train_policy\.sh\s+(\S+)\s+(\S+)")
SCORE_PATTERN = re.compile(r"test_mean_score:\s*([+-]?\d+(?:\.\d+)?)")
RESULT_PATTERN = re.compile(r"环境\s+(.+?)\s+运行(成功|失败)")


def summarize_task(
    task: str,
    policy: Optional[str],
    scores: List[float],
    status: str,
) -> Optional[Dict[str, object]]:
    """Create a summary row for successful tasks."""
    if status != "成功" or not scores:
        return None

    if len(scores) >= 5:
        top_five = sorted(scores, reverse=True)[:5]
        value = statistics.fmean(top_five)
    else:
        value = scores[-1]

    return {"task": task, "top5_mean_score": value}


def parse_log(log_path: Path) -> Tuple[List[Dict[str, object]], Optional[str]]:
    """Parse the log file and collect summaries for each successful task."""
    results: List[Dict[str, object]] = []

    current_task: Optional[str] = None
    current_policy: Optional[str] = None
    scores: List[float] = []
    seen_policies: List[str] = []

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        stripped = line.strip()

        start_match = START_PATTERN.search(stripped)
        if start_match:
            current_task = start_match.group(1)
            current_policy = None
            scores = []
            continue

        if current_task:
            cmd_match = COMMAND_PATTERN.search(stripped)
            if cmd_match:
                current_policy = cmd_match.group(1)
                if current_policy not in seen_policies:
                    seen_policies.append(current_policy)

            score_match = SCORE_PATTERN.search(stripped)
            if score_match:
                scores.append(float(score_match.group(1)))

            result_match = RESULT_PATTERN.search(stripped)
            if result_match:
                task_name, status = result_match.groups()
                if task_name == current_task:
                    summary = summarize_task(task_name, current_policy, scores, status)
                    if summary:
                        results.append(summary)
                current_task = None
                current_policy = None
                scores = []

    policy_name = seen_policies[0] if seen_policies else None
    return results, policy_name


def write_excel(rows: List[Dict[str, object]], output_path: Path) -> None:
    """Write the results to an Excel-friendly HTML table (opens in Excel)."""
    header = ["task", "top5_mean_score"]
    lines = ["<table>", "<tr>"]
    for key in header:
        lines.append(f"<th>{html.escape(key)}</th>")
    lines.append("</tr>")

    for row in rows:
        lines.append("<tr>")
        for key in header:
            value = row.get(key, "")
            lines.append(f"<td>{html.escape(str(value))}</td>")
        lines.append("</tr>")
    lines.append("</table>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_id",
        help="ID of the OAR stdout log (e.g., 145556 for OAR.145556.stdout)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("."),
        help="Directory containing the OAR stdout log",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to write the output .xls file",
    )
    args = parser.parse_args()

    log_path = args.log_dir / f"OAR.{args.log_id}.stdout"
    rows, policy = parse_log(log_path)
    if not rows:
        raise SystemExit("No successful tasks with test_mean_score entries found.")

    policy_label = policy or "unknown_policy"
    output_path = args.output_dir / f"{policy_label}-top5_mean_score.xls"

    write_excel(rows, output_path)
    print(f"Wrote {len(rows)} task summaries to {output_path}")


if __name__ == "__main__":
    main()
