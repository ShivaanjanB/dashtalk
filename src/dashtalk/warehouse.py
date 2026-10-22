"""Build the DuckDB warehouse from data/raw: one table per dashboard.

- sovereign_macro: one row per country and year (World Bank; regional aggregates dropped)
- corporate_credit: one row per company and year (SEC XBRL frames; companies with revenue of at least
  USD 1bn that year and the core fields present)
- revenue_by_product: one row per month, region and product (synthetic)
- corporate_exclusions: SEC income figures blanked as filing scale errors
"""

from __future__ import annotations

import argparse
import json

import duckdb
import pandas as pd

from dashtalk.fetch import SEC_YEARS, WB_INDICATORS
from dashtalk.paths import RAW, WAREHOUSE

MIN_REVENUE_USD = 1e9
MAX_INCOME_TO_REVENUE = 10


def _raw(name: str) -> list:
    return json.loads((RAW / f"{name}.json").read_text(encoding="utf-8"))


def sovereign_macro() -> pd.DataFrame:
    countries = pd.DataFrame([
        {"iso3": c["id"], "country": c["name"], "region": c["region"]["value"].strip(),
         "income_group": c["incomeLevel"]["value"].strip()}
        for c in _raw("wb_countries") if c["region"]["value"].strip() != "Aggregates"
    ])
    wide = None
    for name in WB_INDICATORS.values():
        part = pd.DataFrame([{"iso3": r["countryiso3code"], "year": int(r["date"]), name: r["value"]}
                             for r in _raw(f"wb_{name}") if r["countryiso3code"]])
        wide = part if wide is None else wide.merge(part, on=["iso3", "year"], how="outer")
    frame = countries.merge(wide, on="iso3", how="inner")
    return frame.sort_values(["country", "year"]).reset_index(drop=True)


def _frame(concept: str, period: str) -> pd.DataFrame:
    rows = _raw(f"sec_{concept}_{period}") if (RAW / f"sec_{concept}_{period}.json").exists() else []
    return pd.DataFrame([{"cik": r["cik"], "company": r["entityName"].strip(), "hq_location": r.get("loc"),
                          concept: float(r["val"]), f"{concept}_end": r["end"]} for r in rows],
                        columns=["cik", "company", "hq_location", concept, f"{concept}_end"])


def corporate_credit() -> pd.DataFrame:
    years = []
    for year in SEC_YEARS:
        dur, inst = f"CY{year}", f"CY{year}Q4I"
        frame = None
        for concept, period in [("Revenues", dur), ("RevenueFromContractWithCustomerExcludingAssessedTax", dur),
                                ("OperatingIncomeLoss", dur), ("InterestExpense", dur),
                                ("InterestExpenseNonoperating", dur), ("NetIncomeLoss", dur), ("Assets", inst),
                                ("Liabilities", inst), ("LongTermDebt", inst), ("LongTermDebtNoncurrent", inst),
                                ("CashAndCashEquivalentsAtCarryingValue", inst)]:
            part = _frame(concept, period)
            if frame is None:
                frame = part
            else:
                frame = frame.merge(part.drop(columns=["company", "hq_location"]), on="cik", how="outer")
                names = part[["cik", "company", "hq_location"]]
                frame = frame.merge(names, on="cik", how="left", suffixes=("", "_new"))
                frame["company"] = frame["company"].fillna(frame.pop("company_new"))
                frame["hq_location"] = frame["hq_location"].fillna(frame.pop("hq_location_new"))
        dates = [c for c in frame.columns if c.endswith("_end")]
        values = frame.columns.difference(["cik", "company", "hq_location", *dates])
        frame[values] = frame[values].apply(pd.to_numeric, errors="coerce")  # empty frames come back as object
        frame[dates] = frame[dates].apply(pd.to_datetime)
        out = pd.DataFrame({
            "cik": frame["cik"].astype(int), "company": frame["company"], "hq_location": frame["hq_location"],
            "year": year,
            # Filers tag revenue and interest expense two ways; take the first one present.
            "revenue_usd": frame["Revenues"].fillna(frame["RevenueFromContractWithCustomerExcludingAssessedTax"]),
            "operating_income_usd": frame["OperatingIncomeLoss"],
            "interest_expense_usd": frame["InterestExpense"].fillna(frame["InterestExpenseNonoperating"]),
            "net_income_usd": frame["NetIncomeLoss"],
            "total_assets_usd": frame["Assets"],
            "total_liabilities_usd": frame["Liabilities"],
            "long_term_debt_usd": frame["LongTermDebt"].fillna(frame["LongTermDebtNoncurrent"]),
            "cash_usd": frame["CashAndCashEquivalentsAtCarryingValue"],
            # SEC files each company's own fiscal year under the nearest calendar year (Walmart's 2022 ends
            # 31 Jan 2023), so keep the real dates to say which year a figure covers.
            "period_end": frame["Revenues_end"].where(frame["Revenues"].notna(),
                                                      frame["RevenueFromContractWithCustomerExcludingAssessedTax_end"])
                                               .fillna(frame["OperatingIncomeLoss_end"]),
            "balance_sheet_date": frame["Assets_end"],
        })
        keep = (out["revenue_usd"] >= MIN_REVENUE_USD) & out["operating_income_usd"].notna() & out["total_assets_usd"].notna()
        years.append(out[keep])
    table = pd.concat(years, ignore_index=True)
    # Filers change how they write their name between years ("MICROSOFT CORPORATION" vs "Microsoft
    # Corporation"); use each company's latest name so one company is one name.
    latest = table.sort_values("year").groupby("cik")[["company", "hq_location"]].last()
    table = table.drop(columns=["company", "hq_location"]).merge(latest, on="cik")
    return table[["cik", "company", "hq_location", *[c for c in table.columns if c not in ("cik", "company", "hq_location")]]] \
        .sort_values(["company", "year"]).reset_index(drop=True)


def drop_implausible_income(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blank income figures more than 10x revenue, and list them.

    A few filers tag a figure at the wrong scale: HighPeak Energy's 2023 net income reads as $216bn on
    $1.1bn of revenue. Real swings stay well under 10x (Teladoc's 2022 impairment loss was 5.7x revenue).
    """
    table, excluded = table.copy(), []
    for column in ("operating_income_usd", "net_income_usd"):
        broken = table[column].abs() > MAX_INCOME_TO_REVENUE * table["revenue_usd"]
        excluded.append(table.loc[broken, ["cik", "company", "year", "revenue_usd"]]
                        .assign(field=column, filed_value=table.loc[broken, column]))
        table.loc[broken, column] = None
    return table, pd.concat(excluded, ignore_index=True)


def revenue_by_product() -> pd.DataFrame:
    sales = pd.read_csv(RAW / "sales.csv", parse_dates=["month"])
    sales["year"] = sales["month"].dt.year
    sales["quarter"] = sales["month"].dt.year.astype(str) + "-Q" + sales["month"].dt.quarter.astype(str)
    sales["month"] = sales["month"].dt.strftime("%Y-%m")
    return sales[["month", "quarter", "year", "region", "product", "units", "revenue_usd", "cost_usd"]]


def build(path=WAREHOUSE) -> dict[str, int]:
    corporate, excluded = drop_implausible_income(corporate_credit())
    tables = {"sovereign_macro": sovereign_macro(), "corporate_credit": corporate,
              "revenue_by_product": revenue_by_product(), "corporate_exclusions": excluded}
    path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(path)) as con:
        for name, frame in tables.items():
            con.register("frame", frame)
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM frame")
            con.unregister("frame")
    return {name: len(frame) for name, frame in tables.items()}


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    counts = build()
    print(json.dumps({"warehouse": str(WAREHOUSE), "rows": counts}, indent=2))


if __name__ == "__main__":
    main()
