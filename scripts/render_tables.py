"""Generate the documentation's results tables from results/*.json.

WHY THIS EXISTS

Four times in this project's history, prose in README.md or docs/learning.md
contradicted the results files while the results themselves were correct. A grid
re-run changes fourteen numbers at once; the tables quoting them were maintained by
hand; nobody re-reads a document they did not just edit. The worst instance
reported a configuration as significant at p = 0.0444 when the authoritative value
was 0.059, and another described an arm as *not* significant after it had become
significantly worse.

Every one of those was caught by an audit script rather than by reading. That is
the finding this file acts on: at this density of numbers, review does not work and
mechanisation does.

HOW IT WORKS

Each generated region in a document is fenced by HTML comments:

    <!-- generated:headline -->
    ...table...
    <!-- /generated:headline -->

Running with no arguments replaces the content between those markers. `--check`
regenerates and compares without writing, exiting non-zero on any difference, so
the test suite fails when a document disagrees with results/ instead of the
disagreement shipping.

`--check` also fails on a document that contains no markers at all. An unmarked
document is unverified rather than clean, and the first version of this script
reported it as up to date -- deleting the markers would have returned the tables to
hand maintenance with the gate still green.

The markers are deliberately visible in the source. A reader editing the markdown
should be able to see that a region is derived and will be overwritten.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DOCS = [ROOT / "README.md", ROOT / "docs" / "learning.md"]

#: Unicode minus, not hyphen. The documents use it throughout and a hand-written
#: table that mixed the two was one of the ways drift became hard to spot by eye.
#: The linter's "ambiguous character" warning is precisely the property wanted: it
#: must not be a hyphen, so the rule is suppressed here rather than obeyed.
MINUS = "−"  # noqa: RUF001


def label_for(doc: Path) -> str:
    """A readable name for a document, without assuming it lives under the repo.

    `relative_to` raises on any path outside ROOT, which would turn a cosmetic
    label into a crash -- including for a caller passing an absolute path from
    somewhere else.
    """
    try:
        return doc.relative_to(ROOT).as_posix()
    except ValueError:
        return doc.name


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def measured(payload: dict) -> dict[str, dict]:
    return {r["name"]: r for r in payload["results"] if r["measured"]}


def signed(value: float, *, bold: bool = False) -> str:
    """Format a delta with a real minus sign, never a stray '+-'."""
    text = f"{abs(value):.4f}"
    sign = "+" if value >= 0 else MINUS
    return f"**{sign}{text}**" if bold else f"{sign}{text}"


def pvalue(p: float | None, significant: bool) -> str:
    if p is None:
        return "—"
    return f"**{p:.4f}** ✓" if significant else f"{p:.3f}"


def headline() -> str:
    """Original versus paraphrased on the shared subset, ordered by effect."""
    orig, para = load("ablation.json"), load("ablation-paraphrased.json")
    on_original, on_paraphrased = measured(orig), measured(para)
    sig = para["significance_vs_baseline"]

    rows = []
    for name in on_paraphrased:
        if name not in on_original:
            continue
        a = on_original[name]["metrics_common_subset"]["ndcg@10"]
        b = on_paraphrased[name]["metrics_common_subset"]["ndcg@10"]
        s = sig.get(name, {})
        rows.append((name, a, b, s.get("delta"), s.get("p_holm"), bool(s.get("significant_at_05"))))

    # Baseline last: it anchors the table and has no delta of its own.
    rows.sort(key=lambda r: (r[3] is None, -(r[3] or 0)))

    lines = [
        "| configuration | original | paraphrased | change | Δ vs base | p (Holm) |",
        "|---|---|---|---|---|---|",
    ]
    for name, a, b, delta, p, is_sig in rows:
        change = f"{MINUS}{abs((b - a) / a * 100):.0f}%" if b < a else f"+{(b - a) / a * 100:.0f}%"
        # Bold marks "significant", not "good" -- a significant decline is bolded
        # too, and labelled, because a bare bold delta on a negative number read as
        # an improvement once already.
        cell = f"**{b:.4f}**" if is_sig and (delta or 0) > 0 else f"{b:.4f}"
        note = " *(worse)*" if is_sig and (delta or 0) < 0 else ""
        lines.append(
            f"| `{name}` | {a:.4f} | {cell} | {change} | "
            f"{'—' if delta is None else signed(delta, bold=is_sig)} | "
            f"{pvalue(p, is_sig)}{note} |"
        )
    return "\n".join(lines)


def full(which: str) -> str:
    """Every measured configuration for one wording, with CI and Recall@50."""
    payload = load("ablation.json" if which == "original" else "ablation-paraphrased.json")
    sig = payload["significance_vs_baseline"]
    rows = sorted(measured(payload).values(), key=lambda r: -r["metrics_common_subset"]["ndcg@10"])
    lines = [
        "| configuration | nDCG@10 | 95% CI | Recall@50 | MRR | Δ vs base | p (Holm) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        m, ci = r["metrics_common_subset"], r["ndcg10_ci95"]
        s = sig.get(r["name"], {})
        is_sig = bool(s.get("significant_at_05"))
        lines.append(
            f"| `{r['name']}` | {m['ndcg@10']:.4f} | [{ci[0]:.3f}, {ci[1]:.3f}] | "
            f"{m['recall@50']:.4f} | {m['mrr']:.4f} | "
            f"{'—' if s.get('delta') is None else signed(s['delta'])} | "
            f"{pvalue(s.get('p_holm'), is_sig)} |"
        )
    return "\n".join(lines)


def overlap() -> str:
    """The low/high lexical-overlap split that exposes the confound."""
    payload = load("ablation.json")
    by_name = measured(payload)
    base = by_name["baseline-bm25-fixed512"]
    lo_b = base["ndcg10_low_overlap_queries"]
    hi_b = base["ndcg10_high_overlap_queries"]

    wanted = [
        "baseline-bm25-fixed512",
        "rerank-bm25-100",
        "rerank-candidates-50",
        "rerank-candidates-25",
        "rerank-candidates-200",
        "hybrid-plus-rerank",
        "retrieval-hybrid-rrf",
        "retrieval-dense-bge",
    ]
    lines = [
        "| configuration | low-overlap nDCG | vs base | high-overlap nDCG | vs base |",
        "|---|---|---|---|---|",
    ]
    for name in wanted:
        r = by_name.get(name)
        if r is None:
            continue
        lo, hi = r["ndcg10_low_overlap_queries"], r["ndcg10_high_overlap_queries"]
        if name == "baseline-bm25-fixed512":
            lines.append(f"| `{name}` | {lo:.4f} | — | {hi:.4f} | — |")
            continue
        dl, dh = (lo - lo_b) / lo_b * 100, (hi - hi_b) / hi_b * 100
        fmt = lambda v: f"+{v:.1f}%" if v >= 0 else f"{MINUS}{abs(v):.1f}%"  # noqa: E731
        lines.append(f"| `{name}` | {lo:.4f} | {fmt(dl)} | {hi:.4f} | {fmt(dh)} |")
    return "\n".join(lines)


def depth() -> str:
    """Candidate depth against the recall ceiling it buys."""
    payload = load("ablation.json")
    by_name = measured(payload)
    order = [
        (25, "rerank-candidates-25"),
        (50, "rerank-candidates-50"),
        (100, "rerank-bm25-100"),
        (200, "rerank-candidates-200"),
    ]
    best = max(
        (by_name[n]["metrics_common_subset"]["ndcg@10"] for _, n in order if n in by_name),
        default=0,
    )
    lines = ["| depth | nDCG@10 | recall ceiling |", "|---|---|---|"]
    for d, name in order:
        r = by_name.get(name)
        if r is None:
            continue
        v = r["metrics_common_subset"]["ndcg@10"]
        cell = f"**{v:.4f}**" if v == best else f"{v:.4f}"
        lines.append(f"| {d} | {cell} | {r['first_stage_recall_ceiling']:.1%} |")
    return "\n".join(lines)


def accepted() -> str:
    """All labels against the audit-accepted subset."""
    a, b = measured(load("ablation.json")), measured(load("ablation-accepted.json"))
    shared = [n for n in a if n in b]
    shared.sort(key=lambda n: -b[n]["metrics_common_subset"]["ndcg@10"])
    lines = ["| configuration | all labels | accepted | Δ |", "|---|---|---|---|"]
    for name in shared[:8]:
        x = a[name]["metrics_common_subset"]["ndcg@10"]
        y = b[name]["metrics_common_subset"]["ndcg@10"]
        lines.append(f"| `{name}` | {x:.4f} | {y:.4f} | {signed(y - x)} |")
    return "\n".join(lines)


BLOCKS = {
    "headline": headline,
    "full-original": lambda: full("original"),
    "full-paraphrased": lambda: full("paraphrased"),
    "overlap-split": overlap,
    "candidate-depth": depth,
    "accepted-subset": accepted,
}


def apply(text: str) -> tuple[str, list[str]]:
    """Replace every generated region, returning the text and the names seen."""
    seen: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        if name not in BLOCKS:
            raise SystemExit(f"unknown generated block {name!r}; known: {sorted(BLOCKS)}")
        seen.append(name)
        return f"{match.group('open')}\n{BLOCKS[name]()}\n{match.group('close')}"

    pattern = re.compile(
        r"(?P<open><!-- generated:(?P<name>[a-z-]+) -->)"
        r".*?"
        r"(?P<close><!-- /generated:(?P=name) -->)",
        re.S,
    )
    return pattern.sub(replace, text), seen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="exit non-zero if any document disagrees with results/"
    )
    args = parser.parse_args()

    stale: list[str] = []
    unmarked: list[str] = []
    for doc in DOCS:
        before = doc.read_text(encoding="utf-8")
        after, seen = apply(before)
        label = label_for(doc)
        if not seen:
            # A document with no markers is not "clean", it is unchecked. Reporting
            # it as clean is how a check stops being able to fail: delete the
            # markers and the tables go back to being hand-maintained while the
            # gate stays green. Every silent bug in this project had this shape.
            print(f"  {label}: NO GENERATED BLOCKS -- its tables are not checked")
            unmarked.append(label)
            continue
        if before == after:
            print(f"  {label}: {len(seen)} block(s) up to date")
            continue
        if args.check:
            stale.append(label)
            print(f"  {label}: STALE -- {len(seen)} block(s) disagree with results/")
        else:
            doc.write_text(after, encoding="utf-8")
            print(f"  {label}: rewrote {len(seen)} block(s)")

    if stale:
        print(
            "\nDocumentation disagrees with results/. Run:\n"
            "    python scripts/render_tables.py\n"
            "and re-read the surrounding prose -- the numbers moved, so the sentences\n"
            "describing them may be wrong too. A regenerated table does not fix a\n"
            "paragraph that draws the wrong conclusion from it."
        )
    if unmarked and args.check:
        print(
            f"\n{unmarked} have no generated blocks. Either the markers were removed --\n"
            "in which case those tables are now unverified and the numbers in them can\n"
            "drift freely -- or this list of documents is wrong. Both need a human."
        )
    if stale or (unmarked and args.check):
        sys.exit(1)


if __name__ == "__main__":
    main()
