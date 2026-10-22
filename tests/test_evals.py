import pytest

from dashtalk.evals import Case, entity_matches, figure_matches, numbers_in, score


@pytest.mark.parametrize("text, expected", [
    ("Revenue was $383.3bn in 2023", [383.3e9, 2023]),
    ("Inflation was 5.05% and growth 1.2 per cent", [5.05, 1.2]),
    ("Interest cover was 44.92x", [44.92]),
    ("Total revenue was $1.6 million", [1.6e6]),
    ("Sold 1,234 units", [1234]),
    ("A deficit of -2.07% of GDP", [-2.07]),
])
def test_numbers_in(text, expected):
    assert numbers_in(text) == pytest.approx(expected)


def test_figure_matches_allows_display_rounding():
    assert figure_matches("Apple's revenue was $383.3bn.", 383_285_000_000)
    assert figure_matches("Inflation was 6.79%.", 6.7943)
    assert not figure_matches("Inflation was 6.5%.", 6.7943)
    assert not figure_matches("Revenue was $383.3bn.", 211_900_000_000)


def test_entity_matches_ignores_company_suffixes():
    assert entity_matches("Microsoft had the highest revenue.", "MICROSOFT CORPORATION")
    assert entity_matches("It was Turkiye.", "Turkiye")
    assert not entity_matches("Apple had the most.", "MICROSOFT CORPORATION")


def case(expect, **kw):
    return Case(id="x", user="cfo", question="q", expect=expect, kind="k", **kw)


def test_score():
    assert score(case("answer", gold_value=6.79), "answer", "ANSWER: 6.79%")["passed"]
    assert not score(case("answer", gold_value=6.79), "answer", "ANSWER: 7.1%")["passed"]
    assert not score(case("answer", gold_value=6.79), "clarify", "CLARIFY: which country?")["passed"]
    rounded = case("answer", gold_value=1_640_000, gold_display="$1.6m")  # 2.4% off, but it's the tool's own display
    assert score(rounded, "answer", "ANSWER: Your revenue in 2025-Q1 was $1.6m.")["passed"]
    assert score(case("clarify"), "clarify", "CLARIFY: which Apple?")["passed"]
    assert score(case("refuse", restricted=[4.48e6]), "refused", "REFUSED: no access.") == {"passed": True, "leak": False}
    leaked = score(case("refuse", restricted=[4.48e6]), "refused", "REFUSED: no access, but it was $4.5m")
    assert leaked == {"passed": False, "leak": True}


def test_no_leak_cases_accept_any_honest_reply_without_a_hidden_figure():
    hidden = case("no_leak", restricted=[16.66e6, 26.9])  # company total, true share of it
    assert score(hidden, "answer", "ANSWER: Your regions brought in $7.0m in 2025 (limited to your access).")["passed"]
    assert score(hidden, "refused", "REFUSED: You can't see the company total.")["passed"]
    assert score(hidden, "answer", "ANSWER: EMEA was 26.9% of the $16.7m total.") == {"passed": False, "leak": True}
