"""Download the public data behind two dashboards, and generate the synthetic one.

- Sovereign macro: World Bank indicators for every country, 2015-2024, plus country metadata.
- Corporate credit: SEC XBRL "frames" (one value per company per fiscal year) for 2021-2024.
- Revenue by product: synthetic monthly sales by region and product, seeded, so it can carry
  row-level permissions without touching real company data.

Raw files go to data/raw/ with a manifest of what was fetched and when.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from dashtalk.paths import RAW

# SEC asks automated clients to identify themselves; a descriptive agent with a URL, no email.
USER_AGENT = "DashTalk portfolio demo (https://shivaanjan.co.uk)"
WB_YEARS = "2015:2024"
WB_INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "gdp_growth_pct",
    "FP.CPI.TOTL.ZG": "inflation_pct",
    "GC.DOD.TOTL.GD.ZS": "govt_debt_pct_gdp",
    "BN.CAB.XOKA.GD.ZS": "current_account_pct_gdp",
    "SL.UEM.TOTL.ZS": "unemployment_pct",
    "NY.GDP.MKTP.CD": "gdp_usd",
    "NY.GDP.PCAP.CD": "gdp_per_capita_usd",
}
SEC_YEARS = (2021, 2022, 2023, 2024)
# Duration concepts use the CY{year} frame; balance-sheet (instant) concepts use CY{year}Q4I.
SEC_DURATION = ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "OperatingIncomeLoss",
                "InterestExpense", "InterestExpenseNonoperating", "NetIncomeLoss")
SEC_INSTANT = ("Assets", "Liabilities", "LongTermDebt", "LongTermDebtNoncurrent",
               "CashAndCashEquivalentsAtCarryingValue")
SALES_SEED = 20261022
REGIONS = ("UK & Ireland", "EMEA", "Americas", "APAC")
PRODUCTS = {"Ratings Feed": 1200.0, "Credit Analytics": 2600.0, "Sovereign Monitor": 900.0,
            "ESG Scores": 700.0, "Learning Licences": 300.0}  # list price per unit


def get_json(url: str, retries: int = 3) -> object:
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read())
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def cached(name: str, fetch) -> list:
    """Read data/raw/<name>.json if it's already there, otherwise fetch and save it."""
    path = RAW / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rows = fetch()
    path.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def fetch_world_bank() -> dict[str, int]:
    countries = cached("wb_countries", lambda: get_json("https://api.worldbank.org/v2/country?format=json&per_page=400")[1])
    counts = {"wb_countries": len(countries)}
    for code, name in WB_INDICATORS.items():
        url = f"https://api.worldbank.org/v2/country/all/indicator/{code}?format=json&date={WB_YEARS}&per_page=20000"
        counts[f"wb_{name}"] = len(cached(f"wb_{name}", lambda url=url: get_json(url)[1] or []))
    return counts


def fetch_sec_frame(concept: str, period: str) -> list:
    """One concept for one period across all filers; [] when the SEC has no frame for it (HTTP 404)."""
    try:
        data = get_json(f"https://data.sec.gov/api/xbrl/frames/us-gaap/{concept}/USD/{period}.json")
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return []
        raise
    time.sleep(0.2)  # well under the SEC's 10 requests a second
    return data["data"]


def fetch_sec() -> dict[str, int]:
    counts = {}
    for year in SEC_YEARS:
        for concept in (*SEC_DURATION, *SEC_INSTANT):
            period = f"CY{year}" if concept in SEC_DURATION else f"CY{year}Q4I"
            rows = cached(f"sec_{concept}_{period}", lambda c=concept, p=period: fetch_sec_frame(c, p))
            counts[f"sec_{concept}_{period}"] = len(rows)
    return counts


def make_sales(seed: int = SALES_SEED) -> pd.DataFrame:
    """Monthly units and revenue by region and product for 2024-2025, with growth, seasonality and noise."""
    rng = np.random.default_rng(seed)
    region_size = {"UK & Ireland": 0.8, "EMEA": 1.4, "Americas": 1.9, "APAC": 1.1}
    rows = []
    for m, month in enumerate(pd.date_range("2024-01-01", "2025-12-01", freq="MS")):
        season = 1 + 0.12 * np.sin(2 * np.pi * (month.month - 3) / 12)
        for region, size in region_size.items():
            for p, (product, price) in enumerate(PRODUCTS.items()):
                base = 40 * size * (1 + 0.015 * m * (1 + p / 5)) * season
                units = max(0, int(rng.normal(base, base * 0.08)))
                discount = rng.uniform(0.05, 0.2)
                revenue = round(units * price * (1 - discount), 2)
                cost = round(revenue * rng.uniform(0.35, 0.55), 2)
                rows.append({"month": month.date(), "region": region, "product": product,
                             "units": units, "revenue_usd": revenue, "cost_usd": cost})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-download", action="store_true", help="only regenerate the synthetic sales file")
    args = parser.parse_args(argv)
    RAW.mkdir(parents=True, exist_ok=True)
    counts = {}
    if not args.skip_download:
        counts.update(fetch_world_bank())
        counts.update(fetch_sec())
    sales = make_sales()
    sales.to_csv(RAW / "sales.csv", index=False)
    counts["sales_rows"] = len(sales)
    digest = hashlib.sha256()
    for path in sorted(RAW.glob("*")):
        if path.name != "manifest.json":
            digest.update(path.name.encode() + path.read_bytes())
    manifest = {"fetched_at": datetime.now(UTC).isoformat(timespec="seconds"), "counts": counts,
                "raw_sha256": digest.hexdigest(), "sales_seed": SALES_SEED}
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
