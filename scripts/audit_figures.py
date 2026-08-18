"""Check every figure quoted in prose against the results files.

WHY THIS EXISTS

`render_tables.py` generates the *tables*. It cannot touch the sentences around
them, and those sentences are full of numbers: "0.6641 against 0.5667", "loses 73%",
"significant at p = 0.0444". Every drift this project has produced was in prose of
exactly that kind, and each one was found by an ad-hoc script written in the moment
and thrown away -- which is why the same class of defect kept coming back.

Run in one sitting after the generation arm was re-measured, an earlier version of
this found three live errors: a limitations list claiming six of fifteen
configurations were unmeasured when all fifteen were, a paragraph claiming query
paraphrasing "is not done" when it is the study's headline, and a sentence claiming
dense retrieval beat the baseline on low-overlap queries while a generated table
forty lines above it showed dense losing by 17.8%.

HOW IT WORKS

Every number written to three or four decimal places in prose -- outside generated
blocks and code fences, which are checked elsewhere -- must appear somewhere in
`results/**/*.json`, at three or four decimal places, in absolute value. That is a
weak check by design: it does not know which number belongs in which sentence, only
that the figure exists somewhere in the measurements. It still catches a figure left
behind by a re-run, which is the failure that actually happens.

Numbers that legitimately do not appear in the results need an entry in `ALLOWED`
with a reason. Two kinds qualify: a historical value quoted deliberately, and a
difference derived from two results files. Requiring the reason in writing is the
point -- an allowlist without one is a place for failures to be filed away.
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

#: Figures that are correct without appearing in any results file, and why.
#: Each entry is a claim someone has to stand behind, not a silencer.
ALLOWED = {
    "0.2003": (
        "the withdrawn hybrid-plus-rerank score, quoted in section 14b precisely "
        "because it was wrong -- it was measured against the wrong shortlist"
    ),
    "0.0444": (
        "the incorrect p-value that shipped for two commits, quoted in section 14c "
        "as the example of documentation drift"
    ),
    "0.0015": (
        "derived: the largest negative difference between ablation.json and "
        "ablation-accepted.json (embed-e5-base). Verified against both files"
    ),
    "0.0101": (
        "derived: the largest positive difference between the same two files "
        "(rerank-candidates-50). Verified against both files"
    ),
}

#: Three or four decimal places. Two is too noisy -- versions, section numbers and
#: ordinary prose are full of them -- and this project reports its metrics at four.
FIGURE = re.compile(r"\b\d+\.\d{3,4}\b")


def prose(path: Path) -> str:
    """Document text with generated regions and code fences removed.

    Generated tables are verified by regenerating them, and a code fence is a
    quotation rather than a claim. Including either would report the same figure
    twice and train a reader to skim the output.
    """
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"<!-- generated:([a-z-]+) -->.*?<!-- /generated:\1 -->", "", text, flags=re.S)
    return re.sub(r"```.*?```", "", text, flags=re.S)


def figures_in_results() -> set[str]:
    """Every number in every results file, at the precisions prose uses.

    Absolute values are included because prose writes a delta as "-0.0391" using a
    typographic minus sign, which the figure pattern does not capture.
    """
    found: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            for value in (node, abs(node)):
                found.add(f"{value:.3f}")
                found.add(f"{value:.4f}")

    for path in sorted(RESULTS.rglob("*.json")):
        try:
            walk(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            # A results file that will not parse is a separate problem, and the
            # ablation runner reports it. Skipping keeps this check single-purpose.
            continue
    return found


def unverifiable() -> list[tuple[str, str, str]]:
    """Figures in prose that no results file contains: (document, figure, context)."""
    known = figures_in_results()
    out: list[tuple[str, str, str]] = []
    for doc in DOCS:
        if not doc.exists():
            continue
        text = prose(doc)
        for match in FIGURE.finditer(text):
            figure = match.group(0)
            if figure in known or figure in ALLOWED:
                continue
            context = text[max(0, match.start() - 70) : match.start() + 40]
            out.append((doc.name, figure, " ".join(context.split())))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Check prose figures against results/")
    parser.add_argument(
        "--list-allowed", action="store_true", help="print the allowlist and its reasons"
    )
    args = parser.parse_args()

    if args.list_allowed:
        for figure, reason in sorted(ALLOWED.items()):
            print(f"  {figure}: {reason}")
        return

    missing = unverifiable()
    if not missing:
        print(f"  every figure in prose appears in results/ ({len(ALLOWED)} allowed exceptions)")
        return

    print(f"  {len(missing)} figure(s) in prose appear in no results file:\n")
    for doc, figure, context in missing:
        print(f"  {doc}: {figure}")
        print(f"      ...{context}...")
    print(
        "\nEither the figure is stale -- a re-run moved it and the sentence was not "
        "updated -- or\nit is legitimately derived, in which case add it to ALLOWED "
        "with the reason."
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
