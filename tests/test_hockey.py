"""Regression tests for the task's main failure modes; no network required."""

import numpy as np
import pandas as pd
import pytest

from src.hockey import COLUMNS, build_pairs, parse_page, split_pairs, typed_raw, validate_raw


def make_raw(records):
    rows = []
    for team, year, rate in records:
        rows.append([team, year, 40, 30, None, rate, 240, 220, 20])
    return typed_raw(pd.DataFrame(rows, columns=COLUMNS))


def test_future_median_uses_unpaired_teams_and_strict_inequality():
    raw = make_raw([("A", 2000, .4), ("B", 2000, .3),
                    ("A", 2001, .5), ("B", 2001, .6), ("New", 2001, .9)])
    pairs, excluded = build_pairs(raw)
    assert len(pairs) == 2
    assert pairs.future_median.tolist() == [.6, .6]
    assert pairs.target.tolist() == [0, 0]  # B is equal to the full-year median.
    assert len(excluded) == 3


def test_missing_calendar_year_is_not_bridged_and_names_are_exact():
    raw = make_raw([("A", 2003, .4), ("A", 2005, .6),
                    ("Old Name", 2003, .3), ("New Name", 2004, .7)])
    pairs, excluded = build_pairs(raw)
    assert pairs.empty
    assert len(excluded) == 4


def test_duplicate_keys_fail_instead_of_multiplying_rows():
    raw = make_raw([("A", 2000, .4), ("A", 2000, .5)])
    with pytest.raises(ValueError, match="unique_team_year"):
        build_pairs(raw)


@pytest.mark.parametrize("column,value,check", [
    ("goal_diff", 99, "goal_diff_equals"),
    ("win_rate", 1.1, "win_rate_between"),
    ("wins", -1, "nonnegative_counts"),
    ("goals_for", pd.NA, "required_values_present"),
])
def test_source_inconsistencies_fail(column, value, check):
    raw = make_raw([("A", 2000, .4)])
    raw.loc[0, column] = value
    with pytest.raises(ValueError, match=check):
        validate_raw(raw)


def test_split_uses_target_year_inclusive_boundaries():
    years = [2005, 2006, 2008, 2009, 2011]
    pairs = pd.DataFrame({"year": np.repeat(np.array(years) - 1, 2),
                          "target_year": np.repeat(years, 2),
                          "target": [0, 1] * len(years)})
    split = split_pairs(pairs)
    assert set(split["train"].target_year) == {2005}
    assert set(split["validation"].target_year) == {2006, 2008}
    assert set(split["test"].target_year) == {2009, 2011}
    assert len(set.union(*(set(x.index) for x in split.values()))) == len(pairs)


def test_html_parser_preserves_empty_ot_and_reads_pagination():
    css = ["name", "year", "wins", "losses", "ot-losses", "pct", "gf", "ga", "diff"]
    values = [" A ", "2000", "40", "30", "", "0.5", "240", "220", "20"]
    cells = "".join(f'<td class="{c}">{v}</td>' for c, v in zip(css, values))
    html = (f'<table><tr class="team">{cells}</tr></table>'
            '<ul class="pagination"><li><a href="?page_num=3">3</a></li></ul>')
    rows, pages = parse_page(html)
    assert rows[0]["team"] == "A"
    assert rows[0]["ot_losses"] is None
    assert pages == {1, 3}


def test_changed_or_empty_html_is_not_silently_accepted():
    with pytest.raises(ValueError, match="no hockey rows"):
        parse_page("<html>Service unavailable</html>")
    with pytest.raises(ValueError, match="Missing HTML field"):
        parse_page('<tr class="team"><td class="name">A</td></tr>')


def test_future_edits_do_not_change_past_relative_features():
    raw = make_raw([("A", 2000, .4), ("B", 2000, .3),
                    ("A", 2001, .5), ("B", 2001, .6)])
    raw.loc[1, ["goals_for", "goal_diff"]] = [260, 40]
    original, _ = build_pairs(raw)
    raw.loc[2, ["goals_for", "goal_diff"]] = [900, 680]
    changed, _ = build_pairs(raw)
    pd.testing.assert_series_equal(original.goals_for_relative, changed.goals_for_relative)
