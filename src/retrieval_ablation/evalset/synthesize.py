"""Build the labelled retrieval benchmark from document structure, deterministically.

Design choice, and the reason it is not the obvious one: gold labels are derived
from the corpus's own table structure rather than by asking a language model to
invent questions. A generated question carries no guarantee that its answer is in
the document at all, so an LLM-first eval set needs a verification pass before any
number computed from it means anything -- and this project must not ship metrics
resting on unverified labels.

Deriving from tables inverts that. A row label, a column header and a cell value
form a fact that provably exists at a known character span, so the *label* is
correct by construction and only the *phrasing* is synthetic. Reproducible with no
API key, no cost, and no clock.

THE CONFOUND THIS MODULE MUST NOT HIDE
--------------------------------------
A templated query built from a row label reuses the document's exact wording. Ask
"What was Apple's research and development expense in fiscal 2025?" and the string
"Research and development" appears verbatim in the gold passage. BM25 will find it
immediately. A retrieval ablation run purely on such queries would conclude that
lexical search is excellent and reranking adds little -- and that conclusion would
be an artifact of how the queries were written, not a finding about retrieval.

So `lexical_overlap` is computed for every query and stored. The ablation reports
low-overlap and high-overlap subsets separately, which converts the confound into
a measured variable. Paraphrasing (see `paraphrase.py`) reduces overlap further,
and is the one part of eval-set construction that needs a model.

Adversarial by construction: the corpus holds four consecutive annual reports per
company, so "research and development expense in fiscal 2025" has three
near-identical distractor passages differing only in the figures. Retrieval has to
discriminate on the year, which is exactly where naive similarity fails.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass

from ..config import GLOBAL_SEED
from ..corpus.models import Block, Document, GoldPassage, Span
from .schema import EvalQuery, QueryKind, Verification

#: A cell holding a reportable quantity: digits, optionally with thousands
#: separators, decimals, currency, percent or parenthesised negatives.
_QUANTITY_RE = re.compile(r"^[$(\-\s]*\d[\d,]*(?:\.\d+)?[)%\s]*$")

#: A usable row label: mostly letters, several characters, not a bare year.
_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 ,'\-/&().]{9,90}$")

#: Column headers that identify a period. Anything else (a segment name, a
#: "Change" column) is not a fiscal period and would make the query ambiguous.
_PERIOD_RE = re.compile(r"(19|20)\d{2}")

_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "did",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "much",
        "of",
        "on",
        "or",
        "record",
        "recorded",
        "report",
        "reported",
        "that",
        "the",
        "to",
        "was",
        "were",
        "what",
        "which",
        "with",
        "year",
    ]
)

#: Trailing footnote marker on a row label, e.g. "Long-term debt, net2". Filings
#: attach these as plain digits, and left in place they leak into the query text
#: as "...net of current maturities2", which reads like a typo and adds a token
#: that appears in the gold passage for no semantic reason.
_FOOTNOTE_SUFFIX_RE = re.compile(r"(?<=[a-z\)])\s?\d{1,2}$")

#: Labels that are section headings inside a statement rather than line items.
_NON_ITEM_LABELS = frozenset(
    {
        "total",
        "assets",
        "liabilities",
        "total assets",
        "total liabilities",
        "current assets",
        "non-current assets",
        "current liabilities",
        "shareholders equity",
    }
)


@dataclass(frozen=True, slots=True)
class TableFact:
    """A value located by row label and period, with the span that reports it."""

    doc_id: str
    row_label: str
    period: str
    value: str
    #: Span of the specific row line, not the whole table. A tight span makes
    #: reachability meaningful: a chunker that splits a table mid-way genuinely
    #: fails to cover the row, and the metric should say so.
    span: Span
    section_path: tuple[str, ...]
    block_id: str


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS and len(w) > 2}


def lexical_overlap(query: str, passage: str) -> float:
    """Fraction of the query's content words that also occur in the passage.

    The confound detector. 1.0 means every meaningful word in the question is
    present verbatim in the answer passage, which makes the query a string-match
    exercise rather than a retrieval one.
    """
    query_words = _content_words(query)
    if not query_words:
        return 0.0
    return len(query_words & _content_words(passage)) / len(query_words)


def _row_span(doc: Document, block: Block, row_index: int) -> Span | None:
    """Locate a rendered table row's line within the document text.

    The table is rendered into the canonical text line by line, so the row's span
    is found by walking the block's own lines. Derived from the text rather than
    recomputed from the grid, because the text is what chunks and gold labels
    share -- recomputing would risk the two drifting apart.
    """
    rendered = doc.slice(block.span)
    cursor = block.span.start
    for index, line in enumerate(rendered.split("\n")):
        if index == row_index:
            stripped = line.strip()
            if not stripped:
                return None
            lead = len(line) - len(line.lstrip())
            return Span(cursor + lead, cursor + lead + len(stripped))
        cursor += len(line) + 1
    return None


def _is_entity_name(label: str, company: str) -> bool:
    """Whether a row label is an entity name rather than a reportable line item.

    Segment and subsidiary tables list company names down the stub column, which
    produces queries like "How much did American Express Co report for american
    express company in 2022?" -- grammatically intact and completely meaningless.
    Compared on content words so "American Express Company" is caught against
    "AMERICAN EXPRESS CO" despite the differing suffix.
    """
    label_words = _content_words(label)
    company_words = _content_words(company)
    if not label_words or not company_words:
        return False
    return label_words <= company_words or company_words <= label_words


def extract_table_facts(doc: Document) -> list[TableFact]:
    """Every (row label, period, value) triple this document reports in a table."""
    facts: list[TableFact] = []
    company = doc.metadata.get("company", "")

    for block in doc.blocks:
        table = block.table
        if table is None or table.n_rows <= table.n_header_rows:
            continue

        header = table.rows[table.n_header_rows - 1] if table.n_header_rows else ()
        # Which columns name a fiscal period. Without one the value cannot be
        # pinned to a year, and "what was revenue" across four filings would have
        # no single correct answer.
        periods = {
            i: match.group()
            for i, cell in enumerate(header)
            if i > 0 and (match := _PERIOD_RE.search(cell))
        }
        if not periods:
            continue

        for row_offset, row in enumerate(table.rows[table.n_header_rows :]):
            if not row:
                continue
            label = _FOOTNOTE_SUFFIX_RE.sub("", row[0].strip().rstrip(":")).strip()
            if not _LABEL_RE.match(label):
                continue
            if label.lower() in _NON_ITEM_LABELS or _is_entity_name(label, company):
                continue

            # The rendered line index: caption occupies line 0 when present, and
            # the markdown delimiter row adds one more line after the headers.
            line_index = (1 if table.caption else 0) + table.n_header_rows + 1 + row_offset
            span = _row_span(doc, block, line_index)
            if span is None or span.length < 8:
                continue

            for column, period in periods.items():
                if column >= len(row):
                    continue
                value = row[column].strip()
                if not value or not _QUANTITY_RE.match(value):
                    continue
                facts.append(
                    TableFact(
                        doc_id=doc.doc_id,
                        row_label=label,
                        period=period,
                        value=value,
                        span=span,
                        section_path=block.section_path,
                        block_id=block.block_id,
                    )
                )
                # One fact per row is enough. Emitting every period of every row
                # would flood the candidate pool with near-duplicates and let a
                # single table dominate the sample.
                break

    return facts


#: Phrasings for a table-lookup query. Several forms rather than one so the eval
#: set is not measuring a single sentence template, but all of them necessarily
#: reuse the row label -- which is what `lexical_overlap` records.
_TEMPLATES = (
    "What was {company}'s {label} for {period}?",
    "How much did {company} report for {label} in {period}?",
    "In {period}, what amount did {company} record as {label}?",
    "Report {company}'s {label} figure for the {period} fiscal year.",
    "{company} {label} {period}",
)


def _query_id(fact: TableFact) -> str:
    """Content-addressed, so the same fact always yields the same id.

    Position-based ids would renumber the whole eval set whenever the corpus or
    the sampling changed, breaking any verification work already done against it.

    The character offset is deliberately NOT part of the hash, and that omission
    is load-bearing. How a table is rendered into text -- pipe table versus
    header-repeating row sentences -- is one of the ablation's axes, and changing
    it changes the canonical text and therefore every offset in the document. If
    the id depended on the offset, the same underlying fact would get two
    different ids under two renderings, and the two configurations could not be
    compared on a shared query set at all. Keying on (document, row label, period)
    identifies the *fact*, which is what is actually being asked about, and lets
    each rendering carry its own correct gold span for the same question.
    """
    digest = hashlib.sha256(f"{fact.doc_id}|{fact.row_label}|{fact.period}".encode()).hexdigest()
    return f"q-{digest[:12]}"


def build_queries(
    docs: Sequence[Document],
    n_queries: int = 220,
    seed: int = GLOBAL_SEED,
) -> list[EvalQuery]:
    """Sample a stratified, deterministic eval set from the corpus.

    Stratified by sector so no industry's filing conventions dominate, and capped
    per document so one enormous filing cannot supply a quarter of the queries.
    Both would otherwise turn the benchmark into a measurement of a single
    company's house style.
    """
    rng = random.Random(seed)

    by_sector: dict[str, list[tuple[Document, TableFact]]] = {}
    for doc in docs:
        facts = extract_table_facts(doc)
        if not facts:
            continue
        sector = doc.metadata.get("sector", "unknown")
        # Cap per document before stratifying, so the cap is not defeated by a
        # sector that happens to contain one very large filing.
        rng.shuffle(facts)
        for fact in facts[:_MAX_FACTS_PER_DOC]:
            by_sector.setdefault(sector, []).append((doc, fact))

    if not by_sector:
        return []

    # Round-robin across sectors so the quota is filled evenly even when sectors
    # have very different candidate counts.
    for pool in by_sector.values():
        rng.shuffle(pool)
    sectors = sorted(by_sector)
    selected: list[tuple[Document, TableFact]] = []
    cursor = 0
    while len(selected) < n_queries and any(by_sector[s] for s in sectors):
        sector = sectors[cursor % len(sectors)]
        if by_sector[sector]:
            selected.append(by_sector[sector].pop())
        cursor += 1

    queries: list[EvalQuery] = []
    for index, (doc, fact) in enumerate(selected):
        company = doc.metadata.get("company", fact.doc_id).title()
        template = _TEMPLATES[index % len(_TEMPLATES)]
        text = template.format(company=company, label=fact.row_label.lower(), period=fact.period)
        passage_text = doc.slice(fact.span)

        queries.append(
            EvalQuery(
                query_id=_query_id(fact),
                text=text,
                gold=(
                    GoldPassage(
                        passage_id=f"{fact.doc_id}:{fact.span.start}-{fact.span.end}",
                        doc_id=fact.doc_id,
                        span=fact.span,
                        gain=2,
                    ),
                ),
                kind=QueryKind.TABLE_LOOKUP,
                verification=Verification.GENERATED,
                lexical_overlap=lexical_overlap(text, passage_text),
                metadata={
                    "ticker": doc.metadata.get("ticker", ""),
                    "sector": doc.metadata.get("sector", ""),
                    "report_date": doc.metadata.get("report_date", ""),
                    "period": fact.period,
                    "section": " > ".join(fact.section_path),
                    "expected_value": fact.value,
                },
            )
        )

    # Deduplicate by id: two documents can report the same label and period, and a
    # repeated query would be scored twice and weight that fact double.
    seen: set[str] = set()
    unique = []
    for query in queries:
        if query.query_id not in seen:
            seen.add(query.query_id)
            unique.append(query)
    return sorted(unique, key=lambda q: q.query_id)


#: Ceiling on candidates drawn from one filing. A 1.3-million-character bank
#: filing yields thousands of table rows; without a cap it would supply most of
#: the benchmark and the result would describe that one document.
_MAX_FACTS_PER_DOC = 40
