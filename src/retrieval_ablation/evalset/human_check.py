"""Read a completed verification sample back into a reportable result.

WHY THIS EXISTS

`build.py` writes `verification_sample.md` for a person to mark, and its docstring
has always said "a reader marks each entry, the marks are fed back, and the verified
subset becomes reportable on its own terms". Nothing fed them back. The file could
have been filled in completely and the project would still report its labels as
unverified, because no code could read a tick.

That is the same defect as a docstring describing an exclusion its function did not
perform: the mechanism was documented, believed, and absent. Anyone who spent an
hour on the sample would have got nothing for it.

WHAT IT REFUSES TO DO

It never reports a rate from an unmarked or partly marked file as though the sample
were finished. An empty sample gives `rejection_rate = None` and the words "not
measured"; a partial one reports the count it actually has and says so. The sample
exists to replace a guess with a measurement, and a confident number derived from
four ticks would defeat that.

It also does not relabel anything on its own. `--apply` writes the verdicts into the
eval set; without it this only reports, because reading a file and changing a
benchmark are different acts.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .build import QUERIES_PATH, SAMPLE_PATH
from .schema import Verification, read_eval_set, write_eval_set

#: A sample entry begins with a numbered heading naming the query id.
ENTRY = re.compile(r"^##\s+\d+\.\s+`([^`]+)`", re.M)

#: A ticked box. Accepts x or X, and tolerates whatever a person writes after a
#: rejection, because someone giving a reason should not have to match a format.
TICKED_OK = re.compile(r"^-\s*\[[xX]\]\s*ok\b", re.M)
TICKED_REJECT = re.compile(r"^-\s*\[[xX]\]\s*reject\b(.*)$", re.M)


@dataclass(frozen=True)
class Verdict:
    query_id: str
    accepted: bool
    reason: str


@dataclass(frozen=True)
class Summary:
    n_entries: int
    n_marked: int
    n_accepted: int
    n_rejected: int
    n_marked_both: int
    verdicts: tuple[Verdict, ...]

    @property
    def rejection_rate(self) -> float | None:
        """None until something is marked: a rate over zero ticks is not a rate."""
        return None if not self.n_marked else self.n_rejected / self.n_marked

    @property
    def complete(self) -> bool:
        return self.n_entries > 0 and self.n_marked == self.n_entries


def parse(path: Path | None = None) -> Summary:
    """Read the sample and return only what is actually marked."""
    target = SAMPLE_PATH if path is None else path
    text = target.read_text(encoding="utf-8") if target.exists() else ""

    starts = [(m.group(1), m.start()) for m in ENTRY.finditer(text)]
    verdicts: list[Verdict] = []
    both = 0
    for index, (query_id, start) in enumerate(starts):
        end = starts[index + 1][1] if index + 1 < len(starts) else len(text)
        block = text[start:end]
        ok = TICKED_OK.search(block)
        reject = TICKED_REJECT.search(block)
        if ok and reject:
            # Contradictory marks are dropped rather than resolved. Choosing which
            # tick was meant would invent a verdict the person did not give.
            both += 1
            continue
        if ok:
            verdicts.append(Verdict(query_id, True, ""))
        elif reject:
            tail = reject.group(1)
            tail = tail.split("reason:", 1)[-1] if "reason:" in tail else tail
            verdicts.append(Verdict(query_id, False, tail.replace("&mdash;", "").strip()))

    return Summary(
        n_entries=len(starts),
        n_marked=len(verdicts),
        n_accepted=sum(1 for v in verdicts if v.accepted),
        n_rejected=sum(1 for v in verdicts if not v.accepted),
        n_marked_both=both,
        verdicts=tuple(verdicts),
    )


def apply_verdicts(summary: Summary, queries_path: Path | None = None) -> int:
    """Write the verdicts into the eval set, returning how many queries changed.

    Only queries a person actually marked are touched. Everything else keeps the
    status it had, so running this against a half-finished sample cannot relabel the
    rest of the benchmark by omission.
    """
    target = QUERIES_PATH if queries_path is None else queries_path
    queries = read_eval_set(target)
    by_id = {v.query_id: v for v in summary.verdicts}

    changed = 0
    out = []
    for query in queries:
        verdict = by_id.get(query.query_id)
        if verdict is None or query.verification is (
            Verification.HUMAN_VERIFIED if verdict.accepted else Verification.REJECTED
        ):
            out.append(query)
            continue
        metadata = dict(query.metadata)
        if verdict.reason:
            metadata["human_reject_reason"] = verdict.reason
        out.append(
            dataclasses.replace(
                query,
                verification=(
                    Verification.HUMAN_VERIFIED if verdict.accepted else Verification.REJECTED
                ),
                metadata=metadata,
            )
        )
        changed += 1

    if changed:
        write_eval_set(out, target)
    return changed


def report(summary: Summary) -> str:
    """A short, honest description of the sample's state."""
    if summary.n_entries == 0:
        return "  no verification sample found, or it contains no entries"

    lines = [
        f"  {summary.n_marked} of {summary.n_entries} entries marked"
        + ("" if summary.complete else "  (incomplete)")
    ]
    if summary.n_marked_both:
        lines.append(
            f"  {summary.n_marked_both} entries had both boxes ticked and were "
            f"skipped rather than guessed"
        )
    if summary.rejection_rate is None:
        lines.append("  human rejection rate: not measured -- nothing is marked yet")
        lines.append("  The labels stay GENERATED, which is what the project reports.")
    else:
        lines.append(
            f"  accepted {summary.n_accepted}, rejected {summary.n_rejected} "
            f"-- rejection rate {summary.rejection_rate:.1%} of what is marked"
        )
        if not summary.complete:
            lines.append(
                "  A rate over the marked subset only: not over the sample, and not "
                "over the eval set."
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read the human verification sample")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the verdicts into queries.jsonl as HUMAN_VERIFIED or REJECTED. "
        "Off by default: reading a file and changing the benchmark are different acts.",
    )
    parser.add_argument("--json", action="store_true", help="emit the summary as JSON")
    args = parser.parse_args()

    summary = parse()
    if args.json:
        print(
            json.dumps(
                {
                    "n_entries": summary.n_entries,
                    "n_marked": summary.n_marked,
                    "n_accepted": summary.n_accepted,
                    "n_rejected": summary.n_rejected,
                    "n_marked_both": summary.n_marked_both,
                    "rejection_rate": summary.rejection_rate,
                    "complete": summary.complete,
                },
                indent=2,
            )
        )
        return

    print(report(summary))
    if args.apply:
        changed = apply_verdicts(summary)
        print(f"  wrote {changed} verification change(s) into {QUERIES_PATH.name}")


if __name__ == "__main__":
    main()
