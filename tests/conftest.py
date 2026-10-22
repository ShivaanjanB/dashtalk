import duckdb
import pandas as pd
import pytest

from dashtalk.fetch import make_sales


def _sovereign() -> pd.DataFrame:
    rows = []
    for iso3, country, region, income, growth, inflation in [
        ("GBR", "United Kingdom", "Europe & Central Asia", "High income", 0.3, 6.8),
        ("TUR", "Turkiye", "Europe & Central Asia", "Upper middle income", 5.1, 53.9),
        ("KOR", "Korea, Rep.", "East Asia & Pacific", "High income", 1.4, 3.6),
        ("USA", "United States", "North America", "High income", 2.9, 4.1),
    ]:
        for year, shift in ((2022, 1.0), (2023, 0.0)):
            rows.append({"iso3": iso3, "country": country, "region": region, "income_group": income, "year": year,
                         "gdp_growth_pct": growth + shift, "inflation_pct": inflation + shift, "govt_debt_pct_gdp": None,
                         "current_account_pct_gdp": -2.0, "unemployment_pct": 4.0, "gdp_usd": 1e12,
                         "gdp_per_capita_usd": 40000.0})
    return pd.DataFrame(rows)


def _corporate() -> pd.DataFrame:
    rows = []
    for cik, company, revenue, op, interest, year_end in [
        (320193, "Apple Inc.", 383.3e9, 114.3e9, 3.93e9, "2023-09-30"),
        (1418121, "APPLE HOSPITALITY REIT, INC.", 1.34e9, 0.25e9, None, "2023-12-31"),  # interest not tagged
        (789019, "MICROSOFT CORPORATION", 211.9e9, 88.5e9, 1.97e9, "2023-06-30"),
    ]:
        rows.append({"cik": cik, "company": company, "hq_location": "US-CA", "year": 2023, "revenue_usd": revenue,
                     "operating_income_usd": op, "interest_expense_usd": interest, "net_income_usd": op * 0.8,
                     "total_assets_usd": revenue * 1.5, "total_liabilities_usd": revenue,
                     "long_term_debt_usd": revenue * 0.3, "cash_usd": revenue * 0.1,
                     "period_end": pd.Timestamp(year_end), "balance_sheet_date": pd.Timestamp("2023-12-31")})
    return pd.DataFrame(rows)


def _sales() -> pd.DataFrame:
    sales = make_sales()
    month = pd.to_datetime(sales["month"])
    sales["year"] = month.dt.year
    sales["quarter"] = month.dt.year.astype(str) + "-Q" + month.dt.quarter.astype(str)
    sales["month"] = month.dt.strftime("%Y-%m")
    return sales[["month", "quarter", "year", "region", "product", "units", "revenue_usd", "cost_usd"]]


@pytest.fixture(scope="session")
def warehouse(tmp_path_factory):
    path = tmp_path_factory.mktemp("wh") / "test.duckdb"
    with duckdb.connect(str(path)) as con:
        for name, frame in {"sovereign_macro": _sovereign(), "corporate_credit": _corporate(),
                            "revenue_by_product": _sales()}.items():
            con.register("frame", frame)
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM frame")
            con.unregister("frame")
    return path


@pytest.fixture
def talk(warehouse, tmp_path):
    from dashtalk.tools import DashTalk

    def make(user: str) -> DashTalk:
        return DashTalk(user, warehouse=warehouse, audit_path=tmp_path / "audit.jsonl")
    return make
