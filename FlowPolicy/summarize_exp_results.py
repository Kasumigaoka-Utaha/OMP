#!/usr/bin/env python3
"""Summarize per-task and per-category top-5 mean scores from exp_results/*.xls."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

# Category definitions
CATEGORY_TASKS: Dict[str, List[str]] = {
    "adroit-hammer": ["adroit_hammer"],
    "adroit-pen": ["adroit_pen"],
    "adroit-door": ["adroit_door"],
    "metaworld-easy": [
        "metaworld_reach-wall",
        "metaworld_handle-pull",
        "metaworld_lever-pull",
        "metaworld_reach",
        "metaworld_faucet-open",
        "metaworld_dial-turn",
        "metaworld_door-open",
        "metaworld_peg-unplug-side",
        "metaworld_window-close",
        "metaworld_door-unlock",
        "metaworld_handle-press",
        "metaworld_faucet-close",
        "metaworld_button-press",
        "metaworld_plate-slide-side",
        "metaworld_drawer-close",
        "metaworld_door-close",
        "metaworld_plate-slide-back",
        "metaworld_plate-slide",
        "metaworld_window-open",
        "metaworld_button-press-wall",
    ],
    "metaworld-medium": [
        "metaworld_soccer",
        "metaworld_hammer",
        "metaworld_coffee-pull",
        "metaworld_peg-insert-side",
        "metaworld_basketball",
    ],
    "metaworld-hard": [
        "metaworld_assembly",
        "metaworld_hand-insert",
        "metaworld_pick-place",
        "metaworld_push",
    ],
    "metaworld-very-hard": [
        "metaworld_shelf-place",
        "metaworld_disassemble",
        "metaworld_pick-place-wall",
        "metaworld_stick-pull",
        "metaworld_stick-push",
    ],
}

CATEGORY_ORDER = [
    "adroit-hammer",
    "adroit-pen",
    "adroit-door",
    "metaworld-easy",
    "metaworld-medium",
    "metaworld-hard",
    "metaworld-very-hard",
]

ROW_PATTERN = re.compile(
    r"<tr>\s*<td>(?P<task>[^<]+)</td>\s*<td>(?P<score>[+-]?[0-9]*\.?[0-9]+)</td>\s*</tr>",
    re.IGNORECASE | re.DOTALL,
)


def parse_policy_file(path: Path) -> Tuple[str, Dict[str, float]]:
    """Return policy name and per-task scores parsed from an .xls (HTML table)."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    scores: Dict[str, float] = {}
    for match in ROW_PATTERN.finditer(text):
        task = match.group("task").strip()
        scores[task] = float(match.group("score"))

    # Policy name is the prefix before "-top5_mean_score"
    stem = path.stem
    policy = stem.split("-top5_mean_score")[0]
    return policy, scores


def build_task_rows(policy: str, scores: Dict[str, float]) -> List[Dict[str, object]]:
    """Flatten per-task rows following the desired category order."""
    rows: List[Dict[str, object]] = []
    for category in CATEGORY_ORDER:
        for task in CATEGORY_TASKS[category]:
            if task not in scores:
                continue
            rows.append(
                {
                    "policy": policy,
                    "category": category,
                    "task": task,
                    "top5_mean_score": scores[task],
                }
            )
    return rows


def build_category_rows(policy: str, scores: Dict[str, float]) -> List[Dict[str, object]]:
    """Compute per-category averages for the policy."""
    rows: List[Dict[str, object]] = []
    for category in CATEGORY_ORDER:
        vals = [scores[t] for t in CATEGORY_TASKS[category] if t in scores]
        if not vals:
            continue
        avg_score = sum(vals) / len(vals)
        rows.append(
            {
                "policy": policy,
                "category": category,
                "avg_top5_mean_score": avg_score,
            }
        )
    return rows


def main() -> None:
    exp_dir = Path("exp_results")
    xls_files = sorted(exp_dir.glob("*.xls"))
    if not xls_files:
        raise SystemExit("No .xls files found under exp_results")

    task_rows: List[Dict[str, object]] = []
    category_rows: List[Dict[str, object]] = []

    for path in xls_files:
        policy, scores = parse_policy_file(path)
        task_rows.extend(build_task_rows(policy, scores))
        category_rows.extend(build_category_rows(policy, scores))

    # Policies present
    policies = sorted({row["policy"] for row in task_rows})

    # Build per-policy category task lists (for sorted sheet)
    tasks_by_policy: Dict[str, Dict[str, List[Tuple[str, float]]]] = {
        p: {c: [] for c in CATEGORY_ORDER} for p in policies
    }
    for row in task_rows:
        tasks_by_policy[row["policy"]][row["category"]].append(
            (row["task"], float(row["top5_mean_score"]))
        )
    for p in policies:
        for c in CATEGORY_ORDER:
            tasks_by_policy[p][c].sort(key=lambda kv: kv[0])  # sort by task name

    # Per-policy averages per category
    avg_by_policy: Dict[str, Dict[str, float]] = {p: {} for p in policies}
    for row in category_rows:
        avg_by_policy[row["policy"]][row["category"]] = float(
            row["avg_top5_mean_score"]
        )

    # Per-policy overall average across all tasks
    overall_avg: Dict[str, float] = {}
    for policy in policies:
        scores = [
            score
            for cat in CATEGORY_ORDER
            for _, score in tasks_by_policy[policy][cat]
        ]
        overall_avg[policy] = sum(scores) / len(scores) if scores else 0.0

    # SpreadsheetML with two worksheets and grid borders
    def xml_escape(val: str) -> str:
        return (
            val.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    lines: List[str] = []
    lines.append('<?xml version="1.0"?>')
    lines.append('<?mso-application progid="Excel.Sheet"?>')
    lines.append(
        '<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet" '
        'xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:x="urn:schemas-microsoft-com:office:excel" '
        'xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">'
    )
    lines.append("<Styles>")
    lines.append(
        '<Style ss:ID="bordered">'
        "<Borders>"
        '<Border ss:Position="Bottom" ss:LineStyle="Continuous" ss:Weight="1"/>'
        '<Border ss:Position="Left" ss:LineStyle="Continuous" ss:Weight="1"/>'
        '<Border ss:Position="Right" ss:LineStyle="Continuous" ss:Weight="1"/>'
        '<Border ss:Position="Top" ss:LineStyle="Continuous" ss:Weight="1"/>'
        "</Borders>"
        "</Style>"
    )
    lines.append("</Styles>")

    def start_sheet(name: str) -> None:
        lines.append(f'<Worksheet ss:Name="{xml_escape(name)}">')
        lines.append("<Table>")

    def end_sheet() -> None:
        lines.append("</Table>")
        lines.append("</Worksheet>")

    def add_row(values: List[Tuple[str, str]]) -> None:
        lines.append("<Row>")
        for v, data_type in values:
            lines.append(
                f'<Cell ss:StyleID="bordered"><Data ss:Type="{data_type}">{xml_escape(v)}</Data></Cell>'
            )
        lines.append("</Row>")

    # Sheet 1: Sorted Results formatted like format.xlsx (tasks down rows, policies across columns)
    start_sheet("Sorted Results")
    add_row([("environment", "String")] + [(p, "String") for p in policies])
    for category in CATEGORY_ORDER:
        for task in CATEGORY_TASKS[category]:
            row_vals: List[Tuple[str, str]] = [(task, "String")]
            for policy in policies:
                score = next(
                    (s for t, s in tasks_by_policy[policy][category] if t == task),
                    None,
                )
                if score is None:
                    row_vals.append(("", "String"))
                else:
                    row_vals.append((str(score), "Number"))
            add_row(row_vals)
        # Category average row
        avg_row: List[Tuple[str, str]] = [("average", "String")]
        for policy in policies:
            vals = [s for _, s in tasks_by_policy[policy][category]]
            if vals:
                avg_row.append((str(sum(vals) / len(vals)), "Number"))
            else:
                avg_row.append(("", "String"))
        add_row(avg_row)
        # Blank separator
        add_row([("", "String")] * (1 + len(policies)))
    end_sheet()

    # Sheet 2: Category Averages (policy rows, category columns + overall)
    start_sheet("Category Averages")
    add_row(
        [("policy", "String")]
        + [(c, "String") for c in CATEGORY_ORDER]
        + [("average_all_tasks", "String")]
    )
    for policy in policies:
        row_vals: List[Tuple[str, str]] = [(policy, "String")]
        for cat in CATEGORY_ORDER:
            val = avg_by_policy.get(policy, {}).get(cat)
            if val is None:
                row_vals.append(("", "String"))
            else:
                row_vals.append((str(val), "Number"))
        row_vals.append((str(overall_avg.get(policy, "")), "Number"))
        add_row(row_vals)
    end_sheet()

    lines.append("</Workbook>")

    output_path = exp_dir / "../summary_top5_mean_scores.xls"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote task and category summaries to {output_path}")


if __name__ == "__main__":
    main()
