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

WHAT THIS CANNOT CATCH

`results/archive/` holds the earlier 216-query runs, and those are results files
too, so a decimal left behind from the smaller benchmark is still "found somewhere"
and passes. That is deliberate -- the archive is quoted on purpose in several
places -- but it means a stale decimal from a superseded run can survive this check.
Percentages are what caught the one real instance, because a *relative* change
between two values rarely coincides across two different benchmarks.

The check also cannot tell whether a figure is in the right sentence. It only knows
the number exists in some measurement. Both limits are the price of a check simple
enough that its output is believed.

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
    "50.6%": (
        "derived: dense retrieval's high-overlap score against the baseline's. "
        "The generated overlap table computes the same figure; the prose repeats it"
    ),
    "31.7%": (
        "a parsing diagnostic -- hidden inline-XBRL as a share of one filing's "
        "characters. Measured during the corpus build, not part of any ablation"
    ),
    "111.7%": (
        "the stale reranking figure from the 216-query overlap table, quoted in "
        "section 14e as the example this check was extended to catch"
    ),
    "104.4%": (
        "its corrected counterpart, quoted in the same sentence so the comparison "
        "is legible. The live value is in the generated overlap-split table"
    ),
    "76%": (
        "the wrong BM25 loss figure that stood in two places, quoted in 14e beside the correct one"
    ),
    "73.2%": (
        "derived: the baseline's relative nDCG loss under paraphrasing, computed "
        "from ablation.json and ablation-paraphrased.json. Rounded to 73% elsewhere"
    ),
    "71.8%": (
        "from a one-off retrieval-depth diagnostic that was never written to "
        "results/, so it cannot be re-checked here. Weaker evidence than anything "
        "else in this document, and flagged as such rather than silently trusted"
    ),
}

#: Three or four decimal places. Two is too noisy -- versions, section numbers and
#: ordinary prose are full of them -- and this project reports its metrics at four.
FIGURE = re.compile(r"\b\d+\.\d{3,4}\b")

#: Percentages are the most drift-prone figures here, because nearly every one is
#: derived from two results values, so a re-run moves it while the sentence stays
#: put. Adding this pattern found a whole stale table in docs/learning.md still
#: carrying the 216-query numbers, and a claim that dense retrieval scored higher
#: on low-overlap queries than high-overlap ones -- true of the smaller benchmark,
#: false on this one, and never re-checked.
PERCENT = re.compile(r"\b(\d+(?:\.\d)?)%")


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


def percentages_in_results() -> set[str]:
    """Every results value expressed as a percentage, at the precisions prose uses.

    A percentage in the documents is almost always a value scaled by 100 or a
    relative change between two of them. Only the first is reconstructed here; a
    relative change that no single value explains has to be justified in ALLOWED,
    which is the cost of keeping the check simple enough to trust.
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
            for value in (float(node) * 100, abs(float(node)) * 100, float(node)):
                found.add(f"{value:.0f}")
                found.add(f"{value:.1f}")

    for path in sorted(RESULTS.rglob("*.json")):
        try:
            walk(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return found


def unverifiable() -> list[tuple[str, str, str]]:
    """Figures in prose that no results file contains: (document, figure, context).

    Two patterns, because they fail differently. A decimal is usually copied from a
    results file and goes stale when that file changes. A percentage is usually
    computed from two of them and goes stale the same way, but is far easier to miss
    by eye -- which is how a whole table of 216-query percentages survived in
    docs/learning.md next to a generated table contradicting it.
    """
    decimals = figures_in_results()
    percents = percentages_in_results()
    out: list[tuple[str, str, str]] = []
    for doc in DOCS:
        if not doc.exists():
            continue
        text = prose(doc)
        for pattern, known, suffix in (
            (FIGURE, decimals, ""),
            (PERCENT, percents, "%"),
        ):
            for match in pattern.finditer(text):
                figure = match.group(1) if suffix else match.group(0)
                if figure in known or figure + suffix in ALLOWED:
                    continue
                context = text[max(0, match.start() - 70) : match.start() + 40]
                out.append((doc.name, figure + suffix, " ".join(context.split())))
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
