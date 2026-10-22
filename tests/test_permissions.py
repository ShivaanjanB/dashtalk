import json

import pytest


def test_dashboards_are_listed_only_if_allowed(talk):
    def names(user):
        return [d["name"] for d in talk(user).list_dashboards()["dashboards"]]
    assert names("cfo") == ["corporate_credit", "revenue_by_product", "sovereign_macro"]
    assert names("contractor") == ["sovereign_macro"]
    assert names("nobody") == []


@pytest.mark.parametrize("user, dashboard", [("contractor", "corporate_credit"), ("contractor", "revenue_by_product"),
                                             ("analyst_apac", "corporate_credit"), ("nobody", "sovereign_macro")])
def test_no_access_to_a_dashboard_is_denied(talk, user, dashboard):
    assert talk(user).query_metric(dashboard, "revenue" if dashboard != "sovereign_macro" else "inflation")["status"] == "denied"
    assert talk(user).describe_dashboard(dashboard)["status"] == "denied"


def test_regions_outside_scope_are_refused_not_filtered(talk):
    emea = talk("analyst_emea")
    refused = emea.query_metric("revenue_by_product", "revenue", {"region": "APAC", "year": 2025})
    assert refused["status"] == "denied" and "rows" not in refused
    mixed = emea.query_metric("revenue_by_product", "revenue", {"region": ["EMEA", "Americas"], "year": 2025})
    assert mixed["status"] == "denied"  # one forbidden region spoils the request
    alias = emea.query_metric("revenue_by_product", "revenue", {"region": "asia", "year": 2025})
    assert alias["status"] == "denied"  # "asia" resolves to APAC first, then is refused


def test_unfiltered_queries_are_limited_to_scope(talk):
    cfo, emea = talk("cfo"), talk("analyst_emea")
    by_region = emea.query_metric("revenue_by_product", "revenue", {"year": 2025}, group_by=["region"])
    assert {r["region"] for r in by_region["rows"]} == {"EMEA", "UK & Ireland"} and by_region["scope"]
    total = emea.query_metric("revenue_by_product", "revenue", {"year": 2025})["rows"][0]["value"]
    own = cfo.query_metric("revenue_by_product", "revenue", {"year": 2025, "region": ["EMEA", "UK & Ireland"]})
    assert total == pytest.approx(own["rows"][0]["value"])
    everything = cfo.query_metric("revenue_by_product", "revenue", {"year": 2025})["rows"][0]["value"]
    assert total < everything


def test_every_call_is_audited(talk, tmp_path):
    emea = talk("analyst_emea")
    emea.list_dashboards()
    emea.query_metric("revenue_by_product", "revenue", {"region": "APAC"})
    emea.query_metric("revenue_by_product", "revenue", {"region": "EMEA"})
    log = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert [(e["user"], e["tool"], e["status"]) for e in log] == [
        ("analyst_emea", "list_dashboards", "ok"), ("analyst_emea", "query_metric", "denied"),
        ("analyst_emea", "query_metric", "ok")]
    assert log[2]["rows"] == 1
