import pytest

from dashtalk.semantic import display, find_metric, load_dashboards


@pytest.mark.parametrize("asked, expected", [("growth", "gdp_growth"), ("CPI inflation", "inflation"),
                                             ("jobless_rate", "unemployment"), ("GDP per head", "gdp_per_capita")])
def test_metric_synonyms(asked, expected):
    assert find_metric(load_dashboards()["sovereign_macro"], asked).name == expected


@pytest.mark.parametrize("country, value", [("UK", 0.3), ("Turkey", 5.1), ("south korea", 1.4), ("usa", 2.9), ("GBR", 0.3)])
def test_country_aliases_and_codes(talk, country, value):
    result = talk("cfo").query_metric("sovereign_macro", "gdp_growth", {"country": country, "year": 2023})
    assert result["status"] == "ok" and result["rows"][0]["value"] == pytest.approx(value)


def test_partial_names_resolve_or_ask_back(talk):
    cfo = talk("cfo")
    one = cfo.query_metric("corporate_credit", "interest cover", {"company": "microsoft", "year": 2023})
    assert one["matched"]["company"] == ["MICROSOFT CORPORATION"]
    assert one["rows"][0]["value"] == pytest.approx(88.5 / 1.97, rel=1e-3) and one["rows"][0]["display"] == "44.92x (fiscal year ended 30 Jun 2023)"
    two = cfo.query_metric("corporate_credit", "revenue", {"company": "apple", "year": 2023})
    assert two["status"] == "ambiguous" and set(two["candidates"]) == {"Apple Inc.", "APPLE HOSPITALITY REIT, INC."}
    grouped = cfo.query_metric("corporate_credit", "revenue", {"company": "apple"}, group_by=["company"])
    assert grouped["status"] == "ok" and len(grouped["rows"]) == 2  # grouping by company makes "apple" fine


def test_sec_figures_name_the_year_they_cover(talk):
    cfo = talk("cfo")
    one = cfo.query_metric("corporate_credit", "revenue", {"company": "microsoft", "year": 2023})
    assert one["rows"][0]["display"] == "$211.9bn (fiscal year ended 30 Jun 2023)" and "Walmart" in one["period"]
    ranked = cfo.query_metric("corporate_credit", "revenue", {"year": 2023}, group_by=["company"])
    assert {r["company"]: r["display"] for r in ranked["rows"]}["Apple Inc."] == "$383.3bn (fiscal year ended 30 Sep 2023)"
    total = cfo.query_metric("corporate_credit", "revenue", {"year": 2023})
    assert total["rows"][0]["display"].endswith("(each company's fiscal year filed under 2023)")
    leverage = cfo.query_metric("corporate_credit", "leverage", {"company": "microsoft", "year": 2023})
    assert leverage["rows"][0]["display"].endswith("(balance sheet at 31 Dec 2023)")
    missing = cfo.query_metric("corporate_credit", "interest cover", {"company": "apple hospitality", "year": 2023})
    assert missing["rows"][0] == {"value": None, "display": "no data (fiscal year ended 31 Dec 2023)"}
    sovereign =cfo.query_metric("sovereign_macro", "inflation", {"country": "UK", "year": 2023})
    assert "(" not in sovereign["rows"][0]["display"] and "period" not in sovereign
    assert "Walmart" in cfo.describe_dashboard("corporate_credit")["period"]


def test_unknown_metric_dimension_and_value(talk):
    cfo = talk("cfo")
    assert cfo.query_metric("sovereign_macro", "happiness", {})["status"] == "not_found"
    assert cfo.query_metric("sovereign_macro", "inflation", {"planet": "Mars"})["status"] == "not_found"
    assert cfo.query_metric("sovereign_macro", "inflation", {"country": "Atlantis"})["status"] == "not_found"


def test_values_are_bound_not_spliced(talk):
    result = talk("cfo").query_metric("sovereign_macro", "inflation", {"country": "x'); DROP TABLE sovereign_macro; --"})
    assert result["status"] == "not_found"
    again = talk("cfo").query_metric("sovereign_macro", "inflation", {"country": "UK", "year": 2023})
    assert again["status"] == "ok"  # the table is still there
    assert "?" in again["sql"] and "United Kingdom" not in again["sql"]


def test_invalid_arguments(talk):
    cfo = talk("cfo")
    assert cfo.query_metric("sovereign_macro", "inflation", {}, limit=1000)["status"] == "invalid"
    assert cfo.query_metric("sovereign_macro", "inflation", {}, order="sideways")["status"] == "invalid"
    assert cfo.query_metric("sovereign_macro", "inflation", {"year": "last year"})["status"] == "invalid"


def test_grouping_by_time_sorts_by_time(talk):
    rows = talk("cfo").query_metric("sovereign_macro", "inflation", {"country": "UK"}, group_by=["year"])["rows"]
    assert [r["year"] for r in rows] == [2022, 2023]


@pytest.mark.parametrize("value, unit, shown", [(383.3e9, "usd", "$383.3bn"), (1.6e6, "usd", "$1.6m"),
                                                (5.049, "percent", "5.05%"), (-2.07, "percent of GDP", "-2.07% of GDP"),
                                                (44.918, "times", "44.92x"), (None, "usd", "no data"), (1234, "units", "1,234 units")])
def test_display(value, unit, shown):
    assert display(value, unit) == shown
