"""The frozen corpus definition.

Reproducibility requires that the corpus be defined by a committed list, not by a
query. "The 30 largest US companies" resolves differently every quarter, so an
ablation table published against it could never be reproduced. Tickers are
hardcoded; CIKs are resolved from EDGAR's official mapping at ingest time, because
a mistyped CIK silently fetches a different company's filings and no downstream
test would catch it.

Selection criteria, stated so the choice can be argued with:

- **Sector spread.** Filing conventions differ sharply by industry. Banks report
  credit-loss allowances in deeply nested tables; pharmaceutical companies carry
  pipeline and patent-expiry tables; utilities carry rate-case tables; retailers
  carry store-count and same-store-sales tables. A single-sector corpus would
  measure a chunker against one house style.

- **Filing-agent diversity.** Sector spread also buys variety in the software
  that generated the HTML, which is what actually determines how hostile the
  markup is.

- **Size.** Large filers produce long, table-heavy filings, which is the
  difficulty this project is designed to measure.
"""

from __future__ import annotations

#: Ticker -> sector label. The sector is carried into document metadata so
#: retrieval can be filtered by it, which is one of the ablation axes.
CORPUS_TICKERS: dict[str, str] = {
    # Technology and semiconductors
    "AAPL": "technology",
    "MSFT": "technology",
    "NVDA": "semiconductors",
    "INTC": "semiconductors",
    "CSCO": "technology",
    # Pharmaceutical and biotechnology
    "PFE": "pharmaceuticals",
    "MRK": "pharmaceuticals",
    "JNJ": "pharmaceuticals",
    "ABBV": "pharmaceuticals",
    "AMGN": "biotechnology",
    # Banking and financial services
    "JPM": "banking",
    "BAC": "banking",
    "GS": "banking",
    "AXP": "financial_services",
    "BLK": "asset_management",
    # Energy and heavy industry
    "XOM": "energy",
    "CVX": "energy",
    "CAT": "industrials",
    "BA": "aerospace",
    "GE": "industrials",
    # Consumer and retail
    "WMT": "retail",
    "KO": "beverages",
    "PG": "consumer_goods",
    "COST": "retail",
    "NKE": "apparel",
    # Telecommunications, utilities, healthcare services
    "T": "telecommunications",
    "VZ": "telecommunications",
    "DUK": "utilities",
    "SO": "utilities",
    "UNH": "health_insurance",
}

#: Annual reports rather than quarterly. A 10-K carries the full financial
#: statements, all the notes, and the risk factors -- the material that makes
#: retrieval hard. A 10-Q is a thinner update and would dilute the corpus.
CORPUS_FORM = "10-K"

#: Filings per company. 30 companies x 4 years is roughly 120 filings, which at
#: the ~226,000 characters measured for a representative large-filer 10-K comes
#: to about 27 million characters, or the 12,000-page corpus this study targets.
#:
#: Multiple years per company is deliberate and load-bearing for the eval set:
#: consecutive annual reports repeat their section structure almost verbatim
#: while the figures change. That makes the corpus adversarial in the way real
#: retrieval is hard -- "what was research and development expense" has four
#: near-identical candidate passages per company, and only one is right for a
#: given fiscal year. A single-year corpus would make retrieval far too easy and
#: would flatter every configuration equally.
FILINGS_PER_COMPANY = 4
