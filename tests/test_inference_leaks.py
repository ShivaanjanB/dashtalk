"""Leaks by arithmetic: a user limited to some regions must not be able to work out a hidden figure.

The attacks: ask for the company total and subtract your own region; rank all regions; ask for your
"share" of the total; break down by product or month with no region filter. The guarantee that blocks
all of them: every value the server returns to a scoped user equals the CFO's value over that user's
regions only, so no returned number depends on a hidden row.
"""

import pytest

SCOPES = {"analyst_emea": ["EMEA", "UK & Ireland"], "analyst_apac": ["APAC"]}
METRICS = ["revenue", "units", "gross_profit", "gross_margin"]
GROUPINGS = [[], ["product"], ["year"], ["quarter"], ["month"], ["region"], ["product", "year"], ["region", "product"]]


def by_group(result, group_by):
    return {tuple(row[g] for g in group_by): row["value"] for row in result["rows"]}


@pytest.mark.parametrize("group_by", GROUPINGS, ids=lambda g: "+".join(g) or "total")
@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("user", SCOPES)
def test_every_scoped_value_is_the_cfo_value_over_the_users_regions_only(talk, user, metric, group_by):
    mine = talk(user).query_metric("revenue_by_product", metric, {}, group_by=group_by, limit=100)
    reference = talk("cfo").query_metric("revenue_by_product", metric, {"region": SCOPES[user]}, group_by=group_by,
                                         limit=100)
    assert mine["status"] == "ok" and mine["rows"]
    assert by_group(mine, group_by) == pytest.approx(by_group(reference, group_by))


@pytest.mark.parametrize("metric", ["revenue", "units", "gross_profit"])
@pytest.mark.parametrize("user", SCOPES)
def test_the_company_total_is_never_returned_so_subtraction_gets_nothing(talk, user, metric):
    mine = talk(user).query_metric("revenue_by_product", metric, {"year": 2025})["rows"][0]["value"]
    company = talk("cfo").query_metric("revenue_by_product", metric, {"year": 2025})["rows"][0]["value"]
    assert mine < company
    own = talk(user).query_metric("revenue_by_product", metric, {"year": 2025, "region": SCOPES[user]})
    assert own["rows"][0]["value"] == pytest.approx(mine)  # "total minus my regions" is zero, not a hidden figure


@pytest.mark.parametrize("order", ["desc", "asc"])
@pytest.mark.parametrize("user", SCOPES)
def test_rankings_only_rank_regions_in_scope(talk, user, order):
    rows = talk(user).query_metric("revenue_by_product", "revenue", {"year": 2025}, group_by=["region"], order=order,
                                   limit=10)["rows"]
    assert {r["region"] for r in rows} <= set(SCOPES[user])


@pytest.mark.parametrize("region", ["APAC", "apac", "asia", "Asia Pacific", ["EMEA", "APAC"], ["UK & Ireland", "Americas"],
                                    "Americas", "north america"])
def test_hidden_regions_are_refused_however_they_are_asked_for(talk, region):
    result = talk("analyst_emea").query_metric("revenue_by_product", "revenue", {"region": region, "year": 2025})
    assert result["status"] == "denied" and "rows" not in result


def test_grouping_by_a_hidden_dimension_value_returns_nothing_hidden(talk):
    result = talk("analyst_apac").query_metric("revenue_by_product", "units", {"product": "ESG Scores"},
                                               group_by=["region", "quarter"], limit=100)
    assert {r["region"] for r in result["rows"]} == {"APAC"} and result["scope"]
