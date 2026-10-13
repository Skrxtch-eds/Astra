"""Collection, validation and calendar-safe construction of hockey examples."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

URL = "https://www.scrapethissite.com/pages/forms/"
COLUMNS = ["team", "year", "wins", "losses", "ot_losses", "win_rate",
           "goals_for", "goals_against", "goal_diff"]
CSS = ["name", "year", "wins", "losses", "ot-losses", "pct",
       "gf", "ga", "diff"]
INTEGER_COLUMNS = ["year", "wins", "losses", "ot_losses", "goals_for",
                   "goals_against", "goal_diff"]


def parse_page(html: str) -> tuple[list[dict], set[int]]:
    """Read rows by field names and page numbers from the pagination element."""
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for tr in soup.select("tr.team"):
        record = {}
        for column, css in zip(COLUMNS, CSS):
            td = tr.select_one(f"td.{css}")
            if td is None:
                raise ValueError(f"Missing HTML field: {css}")
            value = td.get_text(strip=True)
            if column == "team":
                if not value:
                    raise ValueError("Empty team name")
                record[column] = value
            elif value == "" and column == "ot_losses":
                record[column] = None
            elif column == "win_rate":
                record[column] = float(value)
            else:
                record[column] = int(value)
        rows.append(record)
    if not rows:
        raise ValueError("Page contains no hockey rows")
    pages = {1}
    for link in soup.select(".pagination a[href]"):
        query = parse_qs(urlparse(link["href"]).query)
        if "page_num" in query:
            number = int(query["page_num"][0])
            if number < 1:
                raise ValueError("Invalid page number")
            pages.add(number)
    return rows, pages


def typed_raw(frame: pd.DataFrame) -> pd.DataFrame:
    """Preserve the source schema and structural OT missing values."""
    if list(frame.columns) != COLUMNS:
        raise ValueError(f"Expected columns {COLUMNS}, got {list(frame.columns)}")
    result = frame.copy()
    result["team"] = result["team"].astype("string").str.strip()
    for column in INTEGER_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="raise").astype("Int64")
    result["win_rate"] = pd.to_numeric(result["win_rate"], errors="raise").astype(float)
    return result


def validate_raw(raw: pd.DataFrame) -> pd.Series:
    """Fail on violations that would make pairing or classification ambiguous."""
    checks = {
        "nonempty_table": len(raw) > 0,
        "unique_team_year": not raw.duplicated(["team", "year"]).any(),
        "required_values_present": not raw.drop(columns="ot_losses").isna().any().any(),
        "nonempty_team_names": raw["team"].str.len().gt(0).all(),
        "win_rate_between_0_and_1": raw["win_rate"].between(0, 1).all(),
        "nonnegative_counts": raw[["wins", "losses", "ot_losses", "goals_for",
                                    "goals_against"]].ge(0).fillna(True).all().all(),
        "goal_diff_equals_GF_minus_GA": raw["goal_diff"].eq(
            raw["goals_for"] - raw["goals_against"]).all(),
    }
    report = pd.Series(checks, name="passed", dtype=bool)
    if not report.all():
        raise ValueError(f"Source validation failed: {report[~report].index.tolist()}")
    return report


def collect_data(data_dir: Path, *, delay: float = 0.25) -> pd.DataFrame:
    """Follow every discovered page; publish CSV only after complete validation."""
    if delay < 0:
        raise ValueError("delay must be nonnegative")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"], respect_retry_after_header=True)
    rows, log = [], []
    pending, seen = {1}, set()
    with requests.Session() as session:
        session.headers["User-Agent"] = "HockeyResearch/1.0 (educational assignment)"
        session.mount("https://", HTTPAdapter(max_retries=retry))
        while pending:
            page = min(pending)
            pending.remove(page)
            response = session.get(URL, params={"page_num": page}, timeout=(10, 45))
            response.raise_for_status()
            records, discovered = parse_page(response.text)
            rows.extend(records)
            seen.add(page)
            pending.update(discovered - seen)
            log.append({"page": page, "rows": len(records), "url": response.url})
            print(f"Page {page}: {len(records)} rows", flush=True)
            if pending:
                time.sleep(delay)
    if seen != set(range(1, max(seen) + 1)):
        raise ValueError("Pagination is not consecutive")
    raw = typed_raw(pd.DataFrame(rows, columns=COLUMNS))
    validate_raw(raw)
    payload = raw.to_csv(index=False, lineterminator="\n").encode("utf-8")
    metadata = {
        "source_url": URL,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "page_count": len(log), "row_count": len(raw), "pages": log,
        "csv_sha256": hashlib.sha256(payload).hexdigest(),
    }
    temporary = data_dir / "hockey_raw.csv.tmp"
    temporary.write_bytes(payload)
    temporary.replace(data_dir / "hockey_raw.csv")
    (data_dir / "collection_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return raw


def load_raw(data_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Load the offline snapshot and verify its provenance checksum."""
    data_dir = Path(data_dir)
    payload = (data_dir / "hockey_raw.csv").read_bytes()
    metadata = json.loads((data_dir / "collection_metadata.json").read_text(encoding="utf-8"))
    if hashlib.sha256(payload).hexdigest() != metadata["csv_sha256"]:
        raise ValueError("CSV checksum does not match collection metadata")
    raw = typed_raw(pd.read_csv(data_dir / "hockey_raw.csv"))
    validate_raw(raw)
    if len(raw) != metadata["row_count"]:
        raise ValueError("Unexpected CSV row count")
    return raw, metadata


def build_pairs(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute future medians BEFORE joining, matching exact names and t + 1."""
    validate_raw(raw)
    current = raw.copy()
    # All current-season league data is available when the season has ended.
    for column in ["goals_for", "goals_against"]:
        grouped = current.groupby("year")[column]
        scale = grouped.transform("std").replace(0, np.nan)
        current[f"{column}_relative"] = (
            (current[column] - grouped.transform("mean")) / scale).astype(float)
    current["target_year"] = current["year"] + 1
    future = raw[["team", "year", "win_rate"]].rename(
        columns={"year": "target_year", "win_rate": "future_win_rate"})
    medians = raw.groupby("year")["win_rate"].median()
    future["future_median"] = future["target_year"].map(medians)
    merged = current.merge(future, on=["team", "target_year"], how="left",
                           validate="one_to_one", indicator=True)
    excluded = merged.loc[merged["_merge"].eq("left_only"), COLUMNS + ["target_year"]].copy()
    pairs = merged.loc[merged["_merge"].eq("both")].drop(columns="_merge").copy()
    pairs["target"] = pairs["future_win_rate"].gt(pairs["future_median"]).astype(int)
    pairs = pairs.sort_values(["target_year", "team"]).reset_index(drop=True)
    if len(pairs) + len(excluded) != len(raw):
        raise ValueError("Pair accounting failed")
    return pairs, excluded


def split_pairs(pairs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Fixed split by the year whose outcome is predicted."""
    year = pairs["target_year"]
    if year.gt(2011).any():
        raise ValueError("Years after 2011 require an explicit split policy")
    result = {"train": pairs.loc[year.le(2005)].copy(),
              "validation": pairs.loc[year.between(2006, 2008)].copy(),
              "test": pairs.loc[year.between(2009, 2011)].copy()}
    if sum(map(len, result.values())) != len(pairs):
        raise ValueError("Split does not cover all pairs")
    for name, frame in result.items():
        if frame.empty or frame["target"].nunique() != 2:
            raise ValueError(f"{name} needs both target classes")
    return result


if __name__ == "__main__":
    collect_data(Path(__file__).resolve().parents[1] / "data")
