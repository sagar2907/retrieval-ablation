"""The documentation's results tables must agree with results/*.json.

This is the test that closes the failure this project produced four times: prose
quoting numbers that a later re-run had changed, in documents nobody re-read.
Every instance was found by an ad-hoc audit script and none by review, so the
check belongs in the suite rather than in someone's memory.

It reads the committed results and the committed documents. No network, no models,
and nothing regenerated -- if this fails, either the tables are stale or the
generator changed, and both are things a reader would want to know before trusting
a figure.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# Imported directly: a missing generator is a broken repository, not a reason to
# skip the check that the documentation agrees with results/.
import audit_figures  # noqa: E402
import render_tables  # noqa: E402


def documents() -> list[Path]:
    return [p for p in render_tables.DOCS if p.exists()]


class TestGeneratedTables:
    def test_every_generated_block_matches_the_results_files(self):
        """Regenerating in memory must be a no-op.

        A failure here means a document disagrees with results/. That happened
        four times before this test existed, and the worst instance reported a
        configuration as significant at p = 0.0444 when the value was 0.059 --
        published, in the PDF, for two commits.
        """
        stale = []
        for doc in documents():
            before = doc.read_text(encoding="utf-8")
            after, seen = render_tables.apply(before)
            if seen and before != after:
                stale.append(doc.name)
        assert not stale, (
            f"{stale} disagree with results/. Run `python scripts/render_tables.py`, "
            f"then re-read the surrounding prose -- a regenerated table does not fix a "
            f"sentence drawing the wrong conclusion from it."
        )

    def test_the_documents_actually_contain_generated_blocks(self):
        """Guards against the check passing because it found nothing to check.

        `apply()` is a no-op on a document with no markers, so a silent removal of
        every marker would leave this suite green while the tables went back to
        being hand-maintained -- the exact failure mode, wearing the costume of a
        passing test.
        """
        total = 0
        for doc in documents():
            total += len(re.findall(r"<!-- generated:[a-z-]+ -->", doc.read_text(encoding="utf-8")))
        assert total >= 6, f"expected at least 6 generated blocks across the docs, found {total}"

    def test_every_marker_names_a_block_the_generator_knows(self):
        """An unknown name would otherwise be skipped rather than reported."""
        for doc in documents():
            names = re.findall(r"<!-- generated:([a-z-]+) -->", doc.read_text(encoding="utf-8"))
            unknown = sorted(set(names) - set(render_tables.BLOCKS))
            assert not unknown, f"{doc.name} references unknown blocks {unknown}"

    def test_open_and_close_markers_are_balanced(self):
        """An unclosed marker makes the regex swallow the rest of the document."""
        for doc in documents():
            text = doc.read_text(encoding="utf-8")
            opens = re.findall(r"<!-- generated:([a-z-]+) -->", text)
            closes = re.findall(r"<!-- /generated:([a-z-]+) -->", text)
            assert sorted(opens) == sorted(closes), f"{doc.name}: {opens} vs {closes}"


class TestStatedTestCount:
    """The README states a test count, which is a number in prose like any other.

    It was 468 while the suite collected 500 -- stale by the same mechanism as every
    other drifted figure here, and understating the work rather than overstating it,
    which is why nobody would have noticed.
    """

    def test_the_readme_states_the_current_number_of_tests(self):
        """Counted by collecting, because parametrize makes static counting wrong.

        Eight test files use `parametrize`, so counting `def test_` gives 469 where
        the suite collects 500. Collection is the only authority. `--collect-only`
        executes nothing, so this cannot recurse.
        """
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        stated = re.search(r"\*\*([\d,]+) tests pass", readme)
        assert stated, "README no longer states a test count in the expected form"
        claimed = int(stated.group(1).replace(",", ""))

        # Fixed argv, no shell, so nothing here is caller-controlled.
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-p", "no:cacheprovider"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        found = re.search(r"(\d+) tests? collected", proc.stdout)
        assert found, f"could not read a collection count from:\n{proc.stdout[-1500:]}"

        assert claimed == int(found.group(1)), (
            f"README says {claimed} tests, collection finds {found.group(1)}"
        )


class TestCheckModeCanActuallyFail:
    """The CLI gate is verified by running it, not by reading it.

    Both of these were confirmed by injecting the failure into the real README and
    watching `--check` exit 1 and the suite go red. They are here so that stays
    true, because the one thing this project proved repeatedly is that a guard
    nobody has watched fail is not known to work.
    """

    def run_check(self, monkeypatch, docs: list[Path]) -> int:
        monkeypatch.setattr(render_tables, "DOCS", docs)
        monkeypatch.setattr(sys, "argv", ["render_tables.py", "--check"])
        try:
            render_tables.main()
        except SystemExit as exc:
            return int(exc.code or 0)
        return 0

    def test_check_fails_on_a_stale_table(self, monkeypatch, tmp_path):
        """A wrong number inside a generated region must exit non-zero."""
        doc = tmp_path / "stale.md"
        doc.write_text(
            "<!-- generated:candidate-depth -->\n| depth | nDCG@10 |\n| 25 | 0.9999 |\n"
            "<!-- /generated:candidate-depth -->\n",
            encoding="utf-8",
        )

        assert self.run_check(monkeypatch, [doc]) == 1
        # --check must not have "fixed" it silently; the gate reports, it does not edit.
        assert "0.9999" in doc.read_text(encoding="utf-8")

    def test_check_fails_on_a_document_with_no_markers(self, monkeypatch, tmp_path):
        """Regression: an unmarked document was reported as up to date.

        `--check` printed "no generated blocks" and exited 0, so deleting every
        marker turned the tables back into hand-maintained prose with the gate
        still green -- a check that cannot fail, which is the exact defect it was
        written to prevent. An unmarked document is unverified, not clean.
        """
        doc = tmp_path / "unmarked.md"
        doc.write_text("| depth | nDCG@10 |\n| 25 | 0.9999 |\n", encoding="utf-8")

        assert self.run_check(monkeypatch, [doc]) == 1

    def test_check_passes_on_a_correctly_generated_document(self, monkeypatch, tmp_path):
        """Otherwise the two tests above would pass on a gate that always fails."""
        doc = tmp_path / "fresh.md"
        doc.write_text(
            f"<!-- generated:candidate-depth -->\n{render_tables.depth()}\n"
            "<!-- /generated:candidate-depth -->\n",
            encoding="utf-8",
        )

        assert self.run_check(monkeypatch, [doc]) == 0


class TestFormatting:
    def test_a_negative_delta_never_renders_with_a_plus(self):
        """Regression: a formatter produced "+-0.0391" and bolded it.

        The row was `embed-e5-base`, the one configuration significantly *worse* on
        both wordings, and the bold plus made it read as the study's strongest
        improvement. Sign handling gets its own test because the value was correct
        in results/ and wrong only in the rendering.
        """
        # The ambiguous-character rule is suppressed because it is the assertion:
        # the output must be U+2212, so writing a hyphen would defeat the test.
        assert render_tables.signed(-0.0391) == "−0.0391"  # noqa: RUF001
        assert render_tables.signed(-0.0391, bold=True) == "**−0.0391**"  # noqa: RUF001
        assert "+-" not in render_tables.signed(-0.5)
        assert "+-" not in render_tables.signed(-0.5, bold=True)

    def test_a_positive_delta_keeps_its_plus(self):
        assert render_tables.signed(0.068) == "+0.0680"
        assert render_tables.signed(0.068, bold=True) == "**+0.0680**"

    def test_no_generated_table_contains_a_plus_minus_pair(self):
        """The defect above, asserted against the rendered documents."""
        for doc in documents():
            assert "+-" not in doc.read_text(encoding="utf-8"), doc.name

    def test_a_missing_pvalue_is_not_reported_as_zero(self):
        """The baseline has no p-value, and 0.000 would read as overwhelming."""
        assert render_tables.pvalue(None, False) == "—"


class TestProseFigures:
    """The tables are generated; the sentences around them are not.

    Every drift this project produced was in prose of that kind, and each was found
    by a script written in the moment and thrown away. This is that script, kept.
    """

    def test_every_figure_in_prose_appears_in_the_results(self):
        """Run against the real documents, because those are what is at risk.

        Three live errors were found the first time this ran: a limitations list
        claiming six of fifteen configurations were unmeasured when all fifteen
        were, a paragraph calling query paraphrasing "not done" when it is the
        headline finding, and a sentence claiming dense retrieval beat the baseline
        on low-overlap queries while a generated table on the same page showed it
        losing by 17.8%.
        """
        missing = audit_figures.unverifiable()

        assert not missing, (
            "figures in prose that no results file contains: "
            + "; ".join(f"{doc} {fig}" for doc, fig, _ in missing)
            + ". Either a re-run moved the number and the sentence was not updated, "
            "or it is derived and needs an entry in audit_figures.ALLOWED with a reason."
        )

    def test_the_check_can_actually_fail(self, monkeypatch, tmp_path):
        """A figure no run produced must be reported.

        Asserted rather than trusted: this repository has produced several guards
        that could not fail, and each looked exactly like this one.
        """
        doc = tmp_path / "made-up.md"
        doc.write_text("A sentence quoting 0.7431, which no run produced.", encoding="utf-8")
        monkeypatch.setattr(audit_figures, "DOCS", [doc])

        found = audit_figures.unverifiable()

        assert [figure for _, figure, _ in found] == ["0.7431"]

    def test_generated_blocks_and_code_fences_are_not_scanned(self, monkeypatch, tmp_path):
        """Those are verified by regenerating them; a fence is a quotation."""
        doc = tmp_path / "fenced.md"
        doc.write_text(
            "<!-- generated:headline -->\n0.7431\n<!-- /generated:headline -->\n```\n0.7431\n```\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(audit_figures, "DOCS", [doc])

        assert audit_figures.unverifiable() == []

    def test_every_allowlist_entry_carries_a_reason(self):
        """An allowlist without reasons is where failures get quietly filed away."""
        for figure, reason in audit_figures.ALLOWED.items():
            assert reason and len(reason) > 30, f"{figure} needs a real justification"

    def test_a_stale_percentage_is_reported(self, monkeypatch, tmp_path):
        """Regression: a whole table of 216-query percentages survived a re-run.

        docs/learning.md carried a second, hand-maintained copy of the overlap-split
        table showing the baseline at 0.1091 and reranking at +111.7%, forty lines
        from a generated table showing 0.0823 and +104.4%. The decimals in it are
        still findable in results/archive/, so only the percentage exposed it --
        a relative change between two values rarely coincides across benchmarks.
        """
        doc = tmp_path / "stale.md"
        doc.write_text("| rerank-bm25-100 | 0.2309 | +111.7% |", encoding="utf-8")
        monkeypatch.setattr(audit_figures, "DOCS", [doc])
        # Emptied deliberately: 14e now quotes this very figure while explaining the
        # defect, so it sits in the live allowlist. The mechanism is what is under
        # test, not the current contents of ALLOWED.
        monkeypatch.setattr(audit_figures, "ALLOWED", {})

        assert "111.7%" in [figure for _, figure, _ in audit_figures.unverifiable()]

    def test_a_percentage_that_matches_a_measurement_passes(self, monkeypatch, tmp_path):
        """Otherwise the check would flag every correct figure too."""
        doc = tmp_path / "fine.md"
        pct = audit_figures.percentages_in_results()
        assert "20.4" in pct, "expected the label-audit rejection rate to be reconstructible"
        doc.write_text("The audit rejected 20.4% of the labels.", encoding="utf-8")
        monkeypatch.setattr(audit_figures, "DOCS", [doc])

        assert audit_figures.unverifiable() == []
