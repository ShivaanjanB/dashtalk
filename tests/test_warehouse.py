import pandas as pd

from dashtalk.warehouse import drop_implausible_income


def test_income_far_above_revenue_is_blanked_as_a_filing_error():
    table = pd.DataFrame({"cik": [1, 2], "company": ["HighPeak Energy, Inc.", "TELADOC HEALTH, INC."],
                          "year": [2023, 2022], "revenue_usd": [1.13e9, 2.41e9],
                          "operating_income_usd": [0.43e9, -13.65e9], "net_income_usd": [215.87e9, -13.66e9]})
    kept, excluded = drop_implausible_income(table)
    assert pd.isna(kept.loc[0, "net_income_usd"]) and kept.loc[0, "operating_income_usd"] == 0.43e9
    assert kept.loc[1, "net_income_usd"] == -13.66e9  # a real impairment loss, 5.7x revenue, stays
    assert excluded[["company", "field"]].values.tolist() == [["HighPeak Energy, Inc.", "net_income_usd"]]
