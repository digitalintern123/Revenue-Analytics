"""
REVENUE ANALYTICS SYSTEM — ONE-CLICK SETUP
===========================================
Download this ONE file, then run:
    python setup_project.py

It will create the entire project folder with all files inside it.
Then:
    cd Revenue_Analytics_System
    pip install -r requirements.txt
    python -m streamlit run app.py
"""

import os

# === Project root ===
ROOT = "Revenue_Analytics_System"

# === Define all files ===
FILES = {}

# ---------------------------------------------------------------
# requirements.txt
# ---------------------------------------------------------------
FILES["requirements.txt"] = """streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
openpyxl>=3.1.0
pdfplumber>=0.10.0
plotly>=5.18.0
sqlalchemy>=2.0.0
"""

# ---------------------------------------------------------------
# packages.txt
# ---------------------------------------------------------------
FILES["packages.txt"] = """build-essential
"""

# ---------------------------------------------------------------
# .streamlit/config.toml
# ---------------------------------------------------------------
FILES[os.path.join(".streamlit", "config.toml")] = """[server]
headless = true
port = 8501
maxUploadSize = 50

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#4F46E5"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F8FAFC"
textColor = "#1E293B"
font = "sans serif"
"""

# ---------------------------------------------------------------
# .gitignore
# ---------------------------------------------------------------
FILES[".gitignore"] = """__pycache__/
*.pyc
*.pyo
.venv/
venv/
data/*.db
.env
*.egg-info/
.DS_Store
"""

# ---------------------------------------------------------------
# modules/__init__.py
# ---------------------------------------------------------------
FILES[os.path.join("modules", "__init__.py")] = '"""Revenue Analytics System — core modules."""\n'

# ---------------------------------------------------------------
# modules/database.py
# ---------------------------------------------------------------
FILES[os.path.join("modules", "database.py")] = r'''"""
database.py — SQLite persistence layer.
Stores every uploaded report so history builds automatically.
UNIQUE constraint on (date, segment, outlet, location) prevents duplicates.
"""
from __future__ import annotations
import datetime as dt
import logging
from pathlib import Path
from typing import Optional
import pandas as pd
from sqlalchemy import (
    Column, Date, Float, Integer, String, Table, MetaData,
    create_engine, select, func,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

logger = logging.getLogger(__name__)
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "revenue.db"
_metadata = MetaData()

revenue_master = Table(
    "revenue_master", _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("date", Date, nullable=False, index=True),
    Column("segment", String(128), nullable=False, index=True),
    Column("outlet", String(128), nullable=False, index=True),
    Column("location", String(128), nullable=False, index=True),
    Column("pax", Float, nullable=True),
    Column("revenue", Float, nullable=False),
    Column("source_file", String(256), nullable=True),
    Column("uploaded_at", String(64), nullable=True),
)

def get_engine(db_path: Optional[Path] = None) -> Engine:
    db_path = db_path or DEFAULT_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", echo=False, future=True)

def init_db(engine: Optional[Engine] = None) -> Engine:
    engine = engine or get_engine()
    _metadata.create_all(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_revenue_record "
            "ON revenue_master (date, segment, outlet, location)"
        )
    return engine

def save_dataframe(df: pd.DataFrame, engine: Optional[Engine] = None, source_file: Optional[str] = None) -> dict:
    engine = engine or init_db()
    inserted, skipped, errors = 0, 0, 0
    upload_ts = dt.datetime.now().isoformat(timespec="seconds")
    with engine.begin() as conn:
        for _, row in df.iterrows():
            try:
                conn.execute(revenue_master.insert().values(
                    date=row["Date"], segment=row["Segment"], outlet=row["Outlet"],
                    location=row["Location"], pax=row.get("Pax"), revenue=row["Revenue"],
                    source_file=source_file, uploaded_at=upload_ts,
                ))
                inserted += 1
            except IntegrityError:
                skipped += 1
            except Exception as exc:
                logger.warning("Row insert error: %s", exc)
                errors += 1
    return {"inserted": inserted, "skipped_duplicates": skipped, "errors": errors}

def load_all(engine: Optional[Engine] = None) -> pd.DataFrame:
    engine = engine or init_db()
    with engine.connect() as conn:
        df = pd.read_sql_table("revenue_master", conn)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df

def load_by_date(target_date: dt.date, engine: Optional[Engine] = None) -> pd.DataFrame:
    engine = engine or init_db()
    query = select(revenue_master).where(revenue_master.c.date == target_date)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df

def get_available_dates(engine: Optional[Engine] = None) -> list:
    engine = engine or init_db()
    with engine.connect() as conn:
        result = conn.execute(select(revenue_master.c.date).distinct().order_by(revenue_master.c.date))
        return [row[0] for row in result.fetchall()]

def get_row_count(engine: Optional[Engine] = None) -> int:
    engine = engine or init_db()
    with engine.connect() as conn:
        return conn.execute(select(func.count()).select_from(revenue_master)).scalar_one()

def reset_database(engine: Optional[Engine] = None) -> None:
    engine = engine or get_engine()
    _metadata.drop_all(engine)
    init_db(engine)
'''

# ---------------------------------------------------------------
# modules/pdf_parser.py
# ---------------------------------------------------------------
FILES[os.path.join("modules", "pdf_parser.py")] = r'''"""
pdf_parser.py — Custom parser for Encalm Group revenue PDF reports.

Handles the specific format:
    Page 1: Summary by segment (For the Day / MTD / YTD)
    Page 2: Detailed outlet breakdown — Delhi/Hyderabad/Goa columns

Melts the wide format into:  Date | Segment | Outlet | Location | Pax | Revenue
"""
from __future__ import annotations
import io, logging, re
from typing import BinaryIO, List, Optional, Union
import numpy as np
import pandas as pd
import pdfplumber

logger = logging.getLogger(__name__)

KNOWN_SEGMENTS = {"lounges & spa", "atithya", "others", "others #", "subsidiary"}


def _clean_num(value) -> float:
    """Convert Indian-format numbers: '49,35,256' or '6 0,81,280' → 4935256.0"""
    if value is None:
        return 0.0
    s = str(value).strip()
    if s in ("", "-", "\u2014", "N/A", "n/a", "None"):
        return 0.0
    s = re.sub(r"[\u20b9$\u20ac\u00a3,\s]", "", s)
    try:
        return float(s)
    except ValueError:
        return 0.0


def _is_segment_header(row: list) -> Optional[str]:
    """Row is a segment header if col 0 has a known segment name and the rest are empty."""
    if not row or not row[0]:
        return None
    name = str(row[0]).strip()
    other = [c for c in row[1:] if c is not None and str(c).strip() not in ("", "None")]
    if other:
        return None
    if name.lower().replace("#", "").strip() in {s.replace("#", "").strip() for s in KNOWN_SEGMENTS}:
        return name.replace("#", "").strip()
    return None


def _is_skip_row(row: list) -> bool:
    if not row:
        return True
    first = str(row[0] or "").strip().lower()
    if not first:
        return True
    if first in ("total", "grand total"):
        return True
    return False


def _find_detailed_table(raw_tables: list):
    """Find the page-2 detailed table (has 'Outlet' in header)."""
    for tbl in raw_tables:
        for row in tbl[:3]:
            cell = str(row[0] or "").strip().lower()
            # Match "Outlet / Business" but NOT "Business Segment"
            if "outlet" in cell:
                return tbl
    return None


def _find_date(raw_tables: list) -> str:
    """Extract the report date from any table header."""
    for tbl in raw_tables:
        for row in tbl[:5]:
            for cell in row:
                if cell:
                    m = re.search(r"(\d{2}-\d{2}-\d{4})", str(cell))
                    if m:
                        return m.group(1)
    return "01-01-2026"


def _parse_detailed_table(table_rows: list, report_date: str) -> pd.DataFrame:
    """
    Parse page 2's detailed table. Columns:
        [0] Outlet   [1] Delhi PAX   [2] Delhi Rev
        [3] Hyd PAX  [4] Hyd Rev     [5] Goa PAX
        [6] Goa Rev  [7] Total Rev
    """
    records = []
    current_segment = "Unknown"

    for row in table_rows:
        if not row or len(row) < 2:
            continue

        seg = _is_segment_header(row)
        if seg:
            current_segment = seg
            continue

        if _is_skip_row(row):
            continue

        outlet = str(row[0]).strip()
        if not outlet:
            continue

        while len(row) < 8:
            row.append(None)

        for loc, pax_idx, rev_idx in [("Delhi", 1, 2), ("Hyderabad", 3, 4), ("Goa", 5, 6)]:
            pax = _clean_num(row[pax_idx])
            rev = _clean_num(row[rev_idx])
            if rev > 0:
                records.append({
                    "Date": report_date,
                    "Segment": current_segment,
                    "Outlet": outlet,
                    "Location": loc,
                    "Pax": int(pax) if pax > 0 else 0,
                    "Revenue": rev,
                })

    return pd.DataFrame(records)


# ── Public API (same signature as before) ──

def extract_pdf_tables(file: Union[BinaryIO, str, bytes]) -> List:
    if isinstance(file, bytes):
        stream = io.BytesIO(file)
    elif isinstance(file, str):
        stream = open(file, "rb")
    else:
        file.seek(0)
        stream = io.BytesIO(file.read())
    tables = []
    with pdfplumber.open(stream) as pdf:
        for page in pdf.pages:
            for tbl in (page.extract_tables() or []):
                if tbl and len(tbl) > 2:
                    tables.append(tbl)
    return tables


def clean_pdf_data(raw_tables: List) -> pd.DataFrame:
    return pd.DataFrame({"_raw": [raw_tables]})


def convert_to_master_format(df_or_raw) -> pd.DataFrame:
    if isinstance(df_or_raw, pd.DataFrame) and "_raw" in df_or_raw.columns:
        raw_tables = df_or_raw["_raw"].iloc[0]
    elif isinstance(df_or_raw, list):
        raw_tables = df_or_raw
    else:
        return df_or_raw

    if not raw_tables:
        return pd.DataFrame(columns=["Date", "Segment", "Outlet", "Location", "Pax", "Revenue"])

    report_date = _find_date(raw_tables)
    detailed = _find_detailed_table(raw_tables)

    if detailed is None:
        logger.warning("No detailed outlet table found — using first table")
        detailed = raw_tables[0]

    # Skip header rows (first 2 rows)
    data_rows = detailed[2:]
    result = _parse_detailed_table(data_rows, report_date)

    if not result.empty:
        result["Date"] = pd.to_datetime(result["Date"], format="%d-%m-%Y", errors="coerce")
        result["Revenue"] = pd.to_numeric(result["Revenue"], errors="coerce").round(0).astype("Int64")
        result["Pax"] = pd.to_numeric(result["Pax"], errors="coerce").round(0).astype("Int64")

    logger.info("Extracted %d rows from PDF", len(result))
    return result
'''

# ---------------------------------------------------------------
# modules/excel_parser.py
# ---------------------------------------------------------------
FILES[os.path.join("modules", "excel_parser.py")] = r'''"""
excel_parser.py — Read and normalize revenue data from Excel uploads.
"""
from __future__ import annotations
import io, logging, re
from typing import BinaryIO, Dict, Optional, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_COLUMN_MAP = {
    r"(?i)date": "Date",
    r"(?i)segment|business.?segment|vertical": "Segment",
    r"(?i)outlet|unit|store": "Outlet",
    r"(?i)location|city|airport": "Location",
    r"(?i)pax|footfall|customers|covers|guests": "Pax",
    r"(?i)revenue|sales|amount|turnover|net.?rev": "Revenue",
}

def _map_col(raw: str) -> Optional[str]:
    for pattern, canonical in _COLUMN_MAP.items():
        if re.search(pattern, str(raw).strip()): return canonical
    return None

def _clean_num(value) -> float:
    if pd.isna(value): return np.nan
    s = str(value).strip()
    if s in ("", "-", "\u2014", "N/A", "n/a", "NA", "null", "None"): return np.nan
    s = re.sub(r"[\u20b9$\u20ac\u00a3,\s]", "", s)
    try: return float(s)
    except ValueError: return np.nan

def read_excel_file(file: Union[BinaryIO, str, bytes]) -> Dict[str, pd.DataFrame]:
    if isinstance(file, bytes): stream = io.BytesIO(file)
    elif isinstance(file, str): stream = file
    else: file.seek(0); stream = io.BytesIO(file.read())
    sheets = pd.read_excel(stream, sheet_name=None, engine="openpyxl")
    return sheets

def clean_excel_data(sheet_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for name, df in sheet_dfs.items():
        if df.empty: continue
        df.columns = [str(c).strip() for c in df.columns]
        frames.append(df.copy())
    if not frames: return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined.replace("", np.nan, inplace=True)
    combined.dropna(how="all", inplace=True)
    rename = {}
    for col in combined.columns:
        mapped = _map_col(col)
        if mapped and mapped not in rename.values(): rename[col] = mapped
    combined.rename(columns=rename, inplace=True)
    for col in ("Pax", "Revenue"):
        if col in combined.columns: combined[col] = combined[col].apply(_clean_num)
    if "Date" in combined.columns:
        combined["Date"] = pd.to_datetime(combined["Date"], dayfirst=True, errors="coerce")
    combined.dropna(subset=["Revenue"], inplace=True)
    return combined.reset_index(drop=True)

def convert_to_master_format(df: pd.DataFrame) -> pd.DataFrame:
    master = ["Date", "Segment", "Outlet", "Location", "Pax", "Revenue"]
    for col in master:
        if col not in df.columns: df[col] = np.nan if col in ("Pax", "Revenue") else "Unknown"
    df = df[master].copy()
    df["Pax"] = pd.to_numeric(df["Pax"], errors="coerce")
    df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df
'''

# ---------------------------------------------------------------
# modules/data_processor.py
# ---------------------------------------------------------------
FILES[os.path.join("modules", "data_processor.py")] = r'''"""
data_processor.py — Orchestration: file detection, parsing, validation, persistence.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import BinaryIO, Optional, Tuple, Union
import pandas as pd
from modules.pdf_parser import extract_pdf_tables, clean_pdf_data, convert_to_master_format as pdf_master
from modules.excel_parser import read_excel_file, clean_excel_data, convert_to_master_format as excel_master
from modules.database import init_db, save_dataframe

logger = logging.getLogger(__name__)
MASTER_COLUMNS = ["Date", "Segment", "Outlet", "Location", "Pax", "Revenue"]

def validate_schema(df: pd.DataFrame) -> Tuple[bool, list]:
    missing = [c for c in MASTER_COLUMNS if c not in df.columns]
    return (len(missing) == 0, missing)

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["Date", "Segment", "Outlet", "Location"], keep="last").reset_index(drop=True)

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("Segment", "Outlet", "Location"):
        if col in df.columns: df[col] = df[col].astype(str).str.strip().str.title()
    if "Date" in df.columns: df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

def parse_file(file: Union[BinaryIO, bytes], filename: str) -> pd.DataFrame:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        raw = extract_pdf_tables(file)
        if not raw: raise ValueError("No tables found in PDF.")
        master = pdf_master(clean_pdf_data(raw))
    elif ext in (".xlsx", ".xls"):
        sheets = read_excel_file(file)
        if not sheets: raise ValueError("Excel file appears empty.")
        master = excel_master(clean_excel_data(sheets))
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Upload .pdf or .xlsx")
    if master.empty: raise ValueError("Zero usable rows after cleaning.")
    is_valid, missing = validate_schema(master)
    if not is_valid: raise ValueError(f"Missing columns: {missing}")
    return remove_duplicates(standardize_columns(master))

def process_and_save(file: Union[BinaryIO, bytes], filename: str, engine=None) -> Tuple[pd.DataFrame, dict]:
    master = parse_file(file, filename)
    engine = engine or init_db()
    summary = save_dataframe(master, engine=engine, source_file=filename)
    return master, summary
'''

# ---------------------------------------------------------------
# modules/revenue_analysis.py
# ---------------------------------------------------------------
FILES[os.path.join("modules", "revenue_analysis.py")] = r'''"""
revenue_analysis.py — Hybrid comparison engine.
Uses uploaded files FIRST, falls back to database history.
Implements DoD, MoM, YoY, Revenue/PAX, Volume vs Spend analysis.
"""
from __future__ import annotations
import datetime as dt
import logging
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def _pct_change(current: float, previous: float) -> Optional[float]:
    if pd.isna(previous) or previous == 0: return None
    return round((current - previous) / previous * 100, 2)

def _abs_change(current: float, previous: float) -> Optional[float]:
    if pd.isna(previous): return None
    return round(current - previous, 2)

def classify_trend(pct: Optional[float]) -> str:
    if pct is None: return "No Data"
    if pct > 5: return "\U0001f4c8 Revenue Increase"
    if pct < -5: return "\U0001f4c9 Revenue Decline"
    return "\u27a1\ufe0f Stable"

def aggregate_revenue(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=["Segment", "Outlet", "Location", "Revenue", "Pax"])
    df = df.copy()
    col_map = {c: c.title() for c in df.columns}
    df.rename(columns=col_map, inplace=True)
    group_cols = ["Segment", "Outlet", "Location"]
    agg_dict = {}
    if "Revenue" in df.columns: agg_dict["Revenue"] = "sum"
    if "Pax" in df.columns: agg_dict["Pax"] = "sum"
    if not agg_dict: return pd.DataFrame()
    return df.groupby(group_cols, as_index=False).agg(agg_dict)

def compare_dataframes(today_df, compare_df, label_today="Today", label_compare="Compare"):
    today_agg = aggregate_revenue(today_df)
    compare_agg = aggregate_revenue(compare_df)
    if today_agg.empty and compare_agg.empty: return pd.DataFrame()
    today_agg = today_agg.rename(columns={"Revenue": f"Revenue_{label_today}", "Pax": f"Pax_{label_today}"})
    compare_agg = compare_agg.rename(columns={"Revenue": f"Revenue_{label_compare}", "Pax": f"Pax_{label_compare}"})
    keys = ["Segment", "Outlet", "Location"]
    merged = pd.merge(today_agg, compare_agg, on=keys, how="outer")
    rev_t, rev_c = f"Revenue_{label_today}", f"Revenue_{label_compare}"
    merged["Abs_Change"] = merged.apply(lambda r: _abs_change(r.get(rev_t,0) or 0, r.get(rev_c,0) or 0), axis=1)
    merged["Pct_Change"] = merged.apply(lambda r: _pct_change(r.get(rev_t,0) or 0, r.get(rev_c,0) or 0), axis=1)
    merged["Trend"] = merged["Pct_Change"].apply(classify_trend)
    pax_t, pax_c = f"Pax_{label_today}", f"Pax_{label_compare}"
    merged["Pax_Pct_Change"] = merged.apply(lambda r: _pct_change(r.get(pax_t,0) or 0, r.get(pax_c,0) or 0), axis=1)
    merged["Pax_Trend"] = merged["Pax_Pct_Change"].apply(classify_trend)
    return merged

def get_comparison_data(uploaded_df, target_date, db_history=None):
    if uploaded_df is not None and not uploaded_df.empty: return uploaded_df
    if db_history is not None and not db_history.empty and target_date is not None:
        db_hist = db_history.copy()
        date_col = "date" if "date" in db_hist.columns else "Date"
        if date_col in db_hist.columns:
            db_hist[date_col] = pd.to_datetime(db_hist[date_col])
            mask = db_hist[date_col].dt.date == target_date
            filtered = db_hist[mask]
            if not filtered.empty: return filtered
    return None

def build_full_comparison(today_df, yesterday_df=None, last_month_df=None, last_year_df=None):
    keys = ["Segment", "Outlet", "Location"]
    today_agg = aggregate_revenue(today_df)
    out = today_agg.rename(columns={"Revenue": "Today_Revenue", "Pax": "Today_Pax"})
    if yesterday_df is not None and not yesterday_df.empty:
        yest_agg = aggregate_revenue(yesterday_df).rename(columns={"Revenue": "Yesterday_Revenue", "Pax": "Yesterday_Pax"})
        out = out.merge(yest_agg[keys + ["Yesterday_Revenue"]], on=keys, how="outer")
        out["DoD %"] = out.apply(lambda r: _pct_change(r.get("Today_Revenue",0) or 0, r.get("Yesterday_Revenue",0) or 0), axis=1)
    else:
        out["Yesterday_Revenue"] = np.nan; out["DoD %"] = np.nan
    if last_month_df is not None and not last_month_df.empty:
        mom_agg = aggregate_revenue(last_month_df).rename(columns={"Revenue": "LastMonth_Revenue", "Pax": "LastMonth_Pax"})
        out = out.merge(mom_agg[keys + ["LastMonth_Revenue"]], on=keys, how="outer")
        out["MoM %"] = out.apply(lambda r: _pct_change(r.get("Today_Revenue",0) or 0, r.get("LastMonth_Revenue",0) or 0), axis=1)
    else:
        out["LastMonth_Revenue"] = np.nan; out["MoM %"] = np.nan
    if last_year_df is not None and not last_year_df.empty:
        yoy_agg = aggregate_revenue(last_year_df).rename(columns={"Revenue": "LastYear_Revenue", "Pax": "LastYear_Pax"})
        out = out.merge(yoy_agg[keys + ["LastYear_Revenue"]], on=keys, how="outer")
        out["YoY %"] = out.apply(lambda r: _pct_change(r.get("Today_Revenue",0) or 0, r.get("LastYear_Revenue",0) or 0), axis=1)
    else:
        out["LastYear_Revenue"] = np.nan; out["YoY %"] = np.nan
    # Add PAX columns
    if yesterday_df is not None and not yesterday_df.empty:
        yp = aggregate_revenue(yesterday_df).rename(columns={"Pax": "Yesterday_Pax"})
        out = out.merge(yp[keys + ["Yesterday_Pax"]], on=keys, how="left")
        out["DoD PAX %"] = out.apply(lambda r: _pct_change(r.get("Today_Pax",0) or 0, r.get("Yesterday_Pax",0) or 0), axis=1)
    else:
        out["Yesterday_Pax"] = np.nan; out["DoD PAX %"] = np.nan
    if last_month_df is not None and not last_month_df.empty:
        mp = aggregate_revenue(last_month_df).rename(columns={"Pax": "LastMonth_Pax"})
        out = out.merge(mp[keys + ["LastMonth_Pax"]], on=keys, how="left")
        out["MoM PAX %"] = out.apply(lambda r: _pct_change(r.get("Today_Pax",0) or 0, r.get("LastMonth_Pax",0) or 0), axis=1)
    else:
        out["LastMonth_Pax"] = np.nan; out["MoM PAX %"] = np.nan
    if last_year_df is not None and not last_year_df.empty:
        yp2 = aggregate_revenue(last_year_df).rename(columns={"Pax": "LastYear_Pax"})
        out = out.merge(yp2[keys + ["LastYear_Pax"]], on=keys, how="left")
        out["YoY PAX %"] = out.apply(lambda r: _pct_change(r.get("Today_Pax",0) or 0, r.get("LastYear_Pax",0) or 0), axis=1)
    else:
        out["LastYear_Pax"] = np.nan; out["YoY PAX %"] = np.nan
    out.rename(columns={"Today_Revenue": "Today Revenue", "Yesterday_Revenue": "Yesterday Revenue",
        "LastMonth_Revenue": "Last Month Revenue", "LastYear_Revenue": "Last Year Revenue",
        "Today_Pax": "Today PAX", "Yesterday_Pax": "Yesterday PAX",
        "LastMonth_Pax": "Last Month PAX", "LastYear_Pax": "Last Year PAX"}, inplace=True)
    return out

def revenue_per_pax(df):
    agg = aggregate_revenue(df)
    if agg.empty: return agg
    agg["Rev_Per_Pax"] = np.where(agg["Pax"] > 0, (agg["Revenue"] / agg["Pax"]).round(2), np.nan)
    return agg

def volume_vs_spend_analysis(today_df, compare_df):
    today_agg = aggregate_revenue(today_df)
    comp_agg = aggregate_revenue(compare_df)
    if today_agg.empty or comp_agg.empty: return pd.DataFrame()
    keys = ["Segment", "Outlet", "Location"]
    merged = today_agg.merge(comp_agg, on=keys, suffixes=("_Today", "_Prev"), how="outer").fillna(0)
    merged["Pax_Chg%"] = merged.apply(lambda r: _pct_change(r["Pax_Today"], r["Pax_Prev"]), axis=1)
    merged["Rev_Chg%"] = merged.apply(lambda r: _pct_change(r["Revenue_Today"], r["Revenue_Prev"]), axis=1)
    merged["RPP_Today"] = np.where(merged["Pax_Today"] > 0, merged["Revenue_Today"] / merged["Pax_Today"], np.nan)
    merged["RPP_Prev"] = np.where(merged["Pax_Prev"] > 0, merged["Revenue_Prev"] / merged["Pax_Prev"], np.nan)
    merged["RPP_Chg%"] = merged.apply(lambda r: _pct_change(r["RPP_Today"], r["RPP_Prev"]), axis=1)
    def _driver(row):
        p, r = abs(row["Pax_Chg%"] or 0), abs(row["RPP_Chg%"] or 0)
        if p == 0 and r == 0: return "No Change"
        return "\U0001f4ca Volume Driven" if p >= r else "\U0001f4b0 Spend Driven"
    merged["Driver"] = merged.apply(_driver, axis=1)
    return merged

def executive_summary(today_df, yesterday_df=None):
    today_agg = aggregate_revenue(today_df)
    total_today = today_agg["Revenue"].sum() if not today_agg.empty else 0
    pax_today = today_agg["Pax"].sum() if not today_agg.empty else 0
    total_yest, pax_yest, growth_pct, pax_pct = 0, 0, None, None
    if yesterday_df is not None and not yesterday_df.empty:
        yest_agg = aggregate_revenue(yesterday_df)
        total_yest = yest_agg["Revenue"].sum() if not yest_agg.empty else 0
        pax_yest = yest_agg["Pax"].sum() if not yest_agg.empty else 0
        growth_pct = _pct_change(total_today, total_yest)
        pax_pct = _pct_change(pax_today, pax_yest)
    top_seg, bot_seg = "N/A", "N/A"
    if yesterday_df is not None and not yesterday_df.empty:
        comp = compare_dataframes(today_df, yesterday_df, "Today", "Yesterday")
        if not comp.empty and "Pct_Change" in comp.columns:
            seg_agg = comp.groupby("Segment", as_index=False).agg({"Revenue_Today": "sum", "Revenue_Yesterday": "sum"})
            seg_agg["pct"] = seg_agg.apply(lambda r: _pct_change(r["Revenue_Today"], r["Revenue_Yesterday"]), axis=1)
            seg_agg.dropna(subset=["pct"], inplace=True)
            if not seg_agg.empty:
                top_seg = seg_agg.loc[seg_agg["pct"].idxmax(), "Segment"]
                bot_seg = seg_agg.loc[seg_agg["pct"].idxmin(), "Segment"]
    return {"total_revenue_today": total_today, "total_revenue_yesterday": total_yest,
        "growth_pct": growth_pct, "total_pax_today": pax_today, "total_pax_yesterday": pax_yest,
        "pax_growth_pct": pax_pct, "top_growing_segment": top_seg, "top_declining_segment": bot_seg}
'''

# ---------------------------------------------------------------
# modules/insights.py
# ---------------------------------------------------------------
FILES[os.path.join("modules", "insights.py")] = r'''"""
insights.py — Generate management-level business insights.
"""
from __future__ import annotations
import datetime as dt
import logging
from typing import Optional
import numpy as np
import pandas as pd
from modules.revenue_analysis import (
    compare_dataframes, volume_vs_spend_analysis, executive_summary, classify_trend,
)

logger = logging.getLogger(__name__)

def _fmt_inr(value):
    if pd.isna(value) or value == 0: return "\u20b90"
    sign = "-" if value < 0 else ""
    return f"{sign}\u20b9{abs(value):,.0f}"

def _fmt_pct(value):
    if value is None or pd.isna(value): return "N/A"
    s = f"{value:+.2f}"
    s = s.rstrip("0").rstrip(".")
    return s + "%"

def generate_insights(today_df, yesterday_df=None, last_month_df=None, last_year_df=None):
    lines = []
    summary = executive_summary(today_df, yesterday_df)
    total_today = summary["total_revenue_today"]
    total_yest = summary["total_revenue_yesterday"]
    growth = summary["growth_pct"]

    lines.append("## \U0001f4ca Executive Summary\n")
    lines.append(f"**Total Revenue Today:** {_fmt_inr(total_today)}")

    if yesterday_df is not None:
        lines.append(f"**Yesterday Revenue:** {_fmt_inr(total_yest)}")
        lines.append(f"**Day-over-Day Growth:** {_fmt_pct(growth)}\n")
        if growth is not None:
            if growth > 0: lines.append(f"Revenue **increased** by {_fmt_pct(growth)} compared to yesterday.")
            elif growth < 0: lines.append(f"Revenue **declined** by {_fmt_pct(growth)} compared to yesterday.")
            else: lines.append("Revenue is **flat** compared to yesterday.")
    lines.append("")

    if yesterday_df is not None and not yesterday_df.empty:
        dod = compare_dataframes(today_df, yesterday_df, "Today", "Yesterday")
        valid = dod.dropna(subset=["Pct_Change"])
        if not valid.empty:
            lines.append("### \U0001f51d Top Performers (Day-over-Day)\n")
            for _, row in valid.nlargest(3, "Pct_Change").iterrows():
                lines.append(f"- **{row['Outlet']}** ({row['Location']}): {_fmt_pct(row['Pct_Change'])} \u2192 {_fmt_inr(row.get('Revenue_Today', 0))}")
            lines.append("")
            lines.append("### \u26a0\ufe0f Underperformers (Day-over-Day)\n")
            for _, row in valid.nsmallest(3, "Pct_Change").iterrows():
                lines.append(f"- **{row['Outlet']}** ({row['Location']}): {_fmt_pct(row['Pct_Change'])} \u2192 {_fmt_inr(row.get('Revenue_Today', 0))}")
            lines.append("")

    if yesterday_df is not None and not yesterday_df.empty:
        vs = volume_vs_spend_analysis(today_df, yesterday_df)
        if not vs.empty:
            vol = vs[vs["Driver"].str.contains("Volume")]
            spend = vs[vs["Driver"].str.contains("Spend")]
            lines.append("### \U0001f50d Revenue Driver Analysis\n")
            if not vol.empty: lines.append(f"**Volume-Driven** (PAX change dominates): {', '.join(vol['Outlet'].head(3).tolist())}")
            if not spend.empty: lines.append(f"**Spend-Driven** (Rev/Pax change dominates): {', '.join(spend['Outlet'].head(3).tolist())}")
            lines.append("")

    pax_pct = summary["pax_growth_pct"]
    if pax_pct is not None:
        lines.append("### \U0001f465 Footfall Insight\n")
        if pax_pct < -5: lines.append(f"PAX dropped by {_fmt_pct(pax_pct)}, indicating **lower customer footfall**.")
        elif pax_pct > 5: lines.append(f"PAX grew by {_fmt_pct(pax_pct)}, suggesting **higher customer throughput**.")
        else: lines.append(f"PAX is broadly stable ({_fmt_pct(pax_pct)}).")
        lines.append("")

    if last_month_df is not None and not last_month_df.empty:
        mom = compare_dataframes(today_df, last_month_df, "Today", "LastMonth")
        if not mom.empty:
            t = mom.get("Revenue_Today", pd.Series([0])).sum()
            p = mom.get("Revenue_LastMonth", pd.Series([0])).sum()
            lines.append(f"### \U0001f4c5 Month-over-Month: {_fmt_pct(round((t-p)/p*100,2) if p else None)} vs same date last month\n")

    if last_year_df is not None and not last_year_df.empty:
        yoy = compare_dataframes(today_df, last_year_df, "Today", "LastYear")
        if not yoy.empty:
            t = yoy.get("Revenue_Today", pd.Series([0])).sum()
            p = yoy.get("Revenue_LastYear", pd.Series([0])).sum()
            lines.append(f"### \U0001f4c6 Year-over-Year: {_fmt_pct(round((t-p)/p*100,2) if p else None)} vs same date last year\n")

    lines.append("### \U0001f3e2 Segment Summary\n")
    lines.append(f"**Top Growing Segment:** {summary['top_growing_segment']}")
    lines.append(f"**Top Declining Segment:** {summary['top_declining_segment']}")
    return "\n".join(lines)
'''

# ---------------------------------------------------------------
# app.py — Main Streamlit Dashboard
# ---------------------------------------------------------------
FILES["app.py"] = r'''"""
app.py — Revenue Performance Monitoring System (Fixed)

Fix: Data persists across page navigation by loading from database,
not just session_state. Session state stores the analysis date reference;
actual data is reloaded from SQLite on every page.
"""
from __future__ import annotations
import datetime as dt
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

_modules_dir = PROJECT_ROOT / "modules"
if not _modules_dir.is_dir() or not (_modules_dir / "__init__.py").exists():
    import streamlit as st
    st.set_page_config(page_title="Setup Error", page_icon="\u26a0\ufe0f")
    st.error("## \u26a0\ufe0f Folder structure problem")
    st.markdown(f"The **`modules/`** folder was not found next to `app.py`.\n\n"
                f"**Current path:** `{PROJECT_ROOT}`\n\n"
                "Run `python setup_project.py` first to create the project structure.")
    st.stop()

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.database import init_db, load_all, load_by_date, get_available_dates, get_row_count, reset_database
from modules.data_processor import parse_file, process_and_save
from modules.revenue_analysis import (
    build_full_comparison, compare_dataframes, executive_summary,
    revenue_per_pax, volume_vs_spend_analysis, get_comparison_data,
    aggregate_revenue, classify_trend,
)
from modules.insights import generate_insights

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Revenue Performance Monitor",
    page_icon="\U0001f4ca",
    layout="wide",
    initial_sidebar_state="expanded",
)

engine = init_db()

# ── Session state: store DATES, not DataFrames ──
# This is the key fix: dates survive page navigation reliably,
# and we reload data from the database on each page.
if "analysis_ready" not in st.session_state:
    st.session_state.analysis_ready = False
if "today_date" not in st.session_state:
    st.session_state.today_date = None
if "yesterday_date" not in st.session_state:
    st.session_state.yesterday_date = None
if "last_month_date" not in st.session_state:
    st.session_state.last_month_date = None
if "last_year_date" not in st.session_state:
    st.session_state.last_year_date = None

# ── Sidebar ──
st.sidebar.title("\U0001f4ca Revenue Monitor")
page = st.sidebar.radio(
    "Navigate",
    ["\U0001f4e4 Upload & Analyze",
     "\U0001f4c8 Executive Summary",
     "\U0001f504 Revenue Comparison",
     "\U0001f3ea Outlet Performance",
     "\U0001f916 Business Insights"],
)
st.sidebar.markdown("---")
row_count = get_row_count(engine)
st.sidebar.caption(f"\U0001f4c1 Database rows: **{row_count:,}**")
dates = get_available_dates(engine)
if dates:
    st.sidebar.caption(f"\U0001f4c5 History: **{len(dates)}** dates stored")
if st.session_state.analysis_ready:
    st.sidebar.success(f"\u2705 Analysis active: {st.session_state.today_date}")


# ── Helper: load data for a date from database ──
def load_date_data(target_date):
    """Load all rows for a specific date from the database."""
    if target_date is None:
        return None
    df = load_by_date(target_date, engine)
    if df.empty:
        return None
    return df


# ── Helper: get all 4 period DataFrames ──
def get_analysis_data():
    """Reload all period data from the database using stored dates."""
    today_df = load_date_data(st.session_state.today_date)
    yesterday_df = load_date_data(st.session_state.yesterday_date)
    last_month_df = load_date_data(st.session_state.last_month_date)
    last_year_df = load_date_data(st.session_state.last_year_date)
    return today_df, yesterday_df, last_month_df, last_year_df


# ── Styling helpers ──
def _color_pct(val):
    try:
        v = float(val)
    except (ValueError, TypeError):
        return ""
    if v > 0:
        return "color: #16a34a; font-weight: 600"
    elif v < 0:
        return "color: #dc2626; font-weight: 600"
    return ""


def _fmt_pct_val(v):
    try:
        s = f"{float(v):+.2f}".rstrip("0").rstrip(".")
        return s
    except (ValueError, TypeError):
        return v


def _fmt_metric_pct(v):
    if v is None:
        return None
    s = f"{v:+.1f}".rstrip("0").rstrip(".")
    return s + "%"


def _style_table(df):
    df = df.copy()
    for c in df.columns:
        if "%" in c:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(2)
        elif "Revenue" in c or "Change" in c:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(0).astype("Int64")
        elif "PAX" in c and "%" not in c and "Trend" not in c:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(0).astype("Int64")
    pct_cols = [c for c in df.columns if "%" in c]
    rev_cols = [c for c in df.columns if "Revenue" in c]
    pax_cols = [c for c in df.columns if "PAX" in c and "%" not in c and "Trend" not in c]
    styler = df.style
    if pct_cols:
        styler = styler.map(_color_pct, subset=pct_cols)
        styler = styler.format({c: _fmt_pct_val for c in pct_cols}, na_rep="\\u2014")
    if rev_cols:
        styler = styler.format({c: "{:,.0f}" for c in rev_cols}, na_rep="\\u2014")
    if pax_cols:
        styler = styler.format({c: "{:,.0f}" for c in pax_cols}, na_rep="\\u2014")
    return styler

def _check_data():
    if not st.session_state.analysis_ready or st.session_state.today_date is None:
        st.warning("\U0001f4e4 Please upload today's report on the **Upload & Analyze** page first.")
        st.stop()


# ===================================================================
# PAGE 1 — Upload & Analyze
# ===================================================================
if page == "\U0001f4e4 Upload & Analyze":
    st.title("\U0001f4e4 Upload Revenue Reports")
    st.markdown(
        "Upload **today's report** (required) and optionally upload comparison reports. "
        "If you skip a comparison file, the system checks database history."
    )
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("\U0001f4cb Today's Report *")
        today_file = st.file_uploader("Today's revenue report (required)", type=["pdf", "xlsx", "xls"], key="upload_today")
    with col2:
        st.subheader("\U0001f4cb Yesterday's Report")
        yesterday_file = st.file_uploader("Yesterday's report (optional)", type=["pdf", "xlsx", "xls"], key="upload_yesterday")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("\U0001f4cb Last Month's Report")
        last_month_file = st.file_uploader("Same date last month (optional)", type=["pdf", "xlsx", "xls"], key="upload_last_month")
    with col4:
        st.subheader("\U0001f4cb Last Year's Report")
        last_year_file = st.file_uploader("Same date last year (optional)", type=["pdf", "xlsx", "xls"], key="upload_last_year")

    st.markdown("---")

    if today_file is not None:
        if st.button("\U0001f680 Process & Analyze", type="primary", use_container_width=True):
            try:
                # ── Process today's report ──
                with st.spinner("Processing today's report..."):
                    today_df, summary = process_and_save(today_file, today_file.name, engine)

                st.success(
                    f"\u2705 Today's report: **{summary['inserted']}** rows inserted, "
                    f"**{summary['skipped_duplicates']}** duplicates skipped."
                )

                # Extract the report date and store it
                report_date = None
                if today_df["Date"].notna().any():
                    ref = today_df["Date"].dropna().iloc[0]
                    report_date = ref.date() if hasattr(ref, "date") else ref
                st.session_state.today_date = report_date

                # ── Process yesterday ──
                if yesterday_file:
                    with st.spinner("Processing yesterday's report..."):
                        ydf, ys = process_and_save(yesterday_file, yesterday_file.name, engine)
                    if ydf["Date"].notna().any():
                        ref = ydf["Date"].dropna().iloc[0]
                        st.session_state.yesterday_date = ref.date() if hasattr(ref, "date") else ref
                    st.success(f"\u2705 Yesterday's report: {ys['inserted']} rows")
                elif report_date:
                    yd = report_date - dt.timedelta(days=1)
                    fb = load_by_date(yd, engine)
                    if not fb.empty:
                        st.session_state.yesterday_date = yd
                        st.info("\U0001f4c2 Yesterday's data loaded from database history")
                    else:
                        st.session_state.yesterday_date = None
                        st.warning("\u26a0\ufe0f No yesterday data available")

                # ── Process last month ──
                if last_month_file:
                    with st.spinner("Processing last month's report..."):
                        mdf, ms = process_and_save(last_month_file, last_month_file.name, engine)
                    if mdf["Date"].notna().any():
                        ref = mdf["Date"].dropna().iloc[0]
                        st.session_state.last_month_date = ref.date() if hasattr(ref, "date") else ref
                    st.success(f"\u2705 Last month's report: {ms['inserted']} rows")
                elif report_date:
                    m = report_date.month - 1 if report_date.month > 1 else 12
                    y = report_date.year if report_date.month > 1 else report_date.year - 1
                    lm_date = dt.date(y, m, min(report_date.day, 28))
                    fb = load_by_date(lm_date, engine)
                    if not fb.empty:
                        st.session_state.last_month_date = lm_date
                        st.info("\U0001f4c2 Last month's data loaded from database history")
                    else:
                        st.session_state.last_month_date = None

                # ── Process last year ──
                if last_year_file:
                    with st.spinner("Processing last year's report..."):
                        lydf, lys = process_and_save(last_year_file, last_year_file.name, engine)
                    if lydf["Date"].notna().any():
                        ref = lydf["Date"].dropna().iloc[0]
                        st.session_state.last_year_date = ref.date() if hasattr(ref, "date") else ref
                    st.success(f"\u2705 Last year's report: {lys['inserted']} rows")
                elif report_date:
                    ly_date = dt.date(report_date.year - 1, report_date.month, min(report_date.day, 28))
                    fb = load_by_date(ly_date, engine)
                    if not fb.empty:
                        st.session_state.last_year_date = ly_date
                        st.info("\U0001f4c2 Last year's data loaded from database history")
                    else:
                        st.session_state.last_year_date = None

                # Mark analysis as ready
                st.session_state.analysis_ready = True

                # Preview
                st.markdown("---")
                st.subheader("\U0001f4cb Extracted Data Preview")
                st.dataframe(today_df.head(30), use_container_width=True)
                st.success("\u2705 **Analysis ready!** Navigate to other pages using the sidebar. Your data is saved and will persist across pages.")

            except ValueError as exc:
                st.error(f"\u26a0\ufe0f {exc}")
            except Exception as exc:
                logging.exception("Upload error")
                st.error(f"\u274c Unexpected error: {exc}")
    else:
        st.info("\U0001f446 Upload at least **today's report** to begin analysis.")

    # Database admin
    with st.expander("\U0001f5c4\ufe0f Database Management"):
        st.write(f"**Total rows stored:** {get_row_count(engine):,}")
        if dates:
            st.write(f"**Date range:** {min(dates)} \u2192 {max(dates)}")
        if st.button("\U0001f5d1\ufe0f Reset Database", type="secondary"):
            reset_database(engine)
            st.session_state.analysis_ready = False
            st.session_state.today_date = None
            st.success("Database cleared.")
            st.rerun()


# ===================================================================
# PAGE 2 — Executive Summary
# ===================================================================
elif page == "\U0001f4c8 Executive Summary":
    st.title("\U0001f4c8 Executive Summary")
    _check_data()

    today_df, yesterday_df, last_month_df, last_year_df = get_analysis_data()
    if today_df is None:
        st.error("Could not load today's data from database.")
        st.stop()

    summary = executive_summary(today_df, yesterday_df)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Revenue Today", f"\u20b9{summary['total_revenue_today']:,.0f}",
                   delta=_fmt_metric_pct(summary['growth_pct']))
    with c2:
        if summary["total_revenue_yesterday"]:
            st.metric("Yesterday Revenue", f"\u20b9{summary['total_revenue_yesterday']:,.0f}")
        else:
            st.metric("Yesterday Revenue", "No data")
    with c3:
        st.metric("PAX Today", f"{summary['total_pax_today']:,.0f}",
                   delta=_fmt_metric_pct(summary['pax_growth_pct']))
    with c4:
        g = summary["growth_pct"]
        st.metric("DoD Growth", _fmt_metric_pct(g) if g is not None else "N/A")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.success(f"\U0001f53a Top Growing Segment: **{summary['top_growing_segment']}**")
    with c2:
        st.error(f"\U0001f53b Top Declining Segment: **{summary['top_declining_segment']}**")

    st.subheader("Revenue by Segment")
    today_agg = aggregate_revenue(today_df)
    if not today_agg.empty:
        seg_pie = today_agg.groupby("Segment", as_index=False)["Revenue"].sum()
        fig = px.pie(seg_pie, names="Segment", values="Revenue", hole=0.4)
        fig.update_layout(height=380, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("PAX by Segment")
    if not today_agg.empty:
        pax_pie = today_agg.groupby("Segment", as_index=False)["Pax"].sum()
        fig_pax = px.pie(pax_pie, names="Segment", values="Pax", hole=0.4)
        fig_pax.update_layout(height=380, margin=dict(t=10, b=10))
        st.plotly_chart(fig_pax, use_container_width=True)

    st.subheader("Revenue by Location")
    if not today_agg.empty:
        loc_bar = today_agg.groupby("Location", as_index=False)["Revenue"].sum().sort_values("Revenue", ascending=True)
        fig2 = px.bar(loc_bar, x="Revenue", y="Location", orientation="h", labels={"Revenue": "Revenue (\u20b9)"})
        fig2.update_layout(height=350, margin=dict(t=10, l=10))
        st.plotly_chart(fig2, use_container_width=True)


# ===================================================================
# PAGE 3 — Revenue Comparison
# ===================================================================
elif page == "\U0001f504 Revenue Comparison":
    st.title("\U0001f504 Revenue Comparison")
    _check_data()

    today_df, yesterday_df, last_month_df, last_year_df = get_analysis_data()
    if today_df is None:
        st.error("Could not load today's data.")
        st.stop()

    s1, s2, s3 = st.columns(3)
    with s1:
        st.write("**DoD:**", "\u2705 Available" if yesterday_df is not None else "\u274c No data")
    with s2:
        st.write("**MoM:**", "\u2705 Available" if last_month_df is not None else "\u274c No data")
    with s3:
        st.write("**YoY:**", "\u2705 Available" if last_year_df is not None else "\u274c No data")

    st.markdown("---")
    tab_dod, tab_mom, tab_yoy, tab_full = st.tabs(["Day over Day", "Month over Month", "Year over Year", "Full Comparison"])

    with tab_dod:
        st.subheader("Today vs Yesterday")
        if yesterday_df is not None:
            dod = compare_dataframes(today_df, yesterday_df, "Today", "Yesterday")
            d = dod[["Segment", "Outlet", "Location", "Revenue_Today", "Revenue_Yesterday", "Pct_Change", "Trend", "Pax_Today", "Pax_Yesterday", "Pax_Pct_Change", "Pax_Trend"]].copy()
            d.rename(columns={"Revenue_Today": "Today Revenue", "Revenue_Yesterday": "Yesterday Revenue", "Pct_Change": "Rev DoD %", "Pax_Today": "Today PAX", "Pax_Yesterday": "Yesterday PAX", "Pax_Pct_Change": "PAX DoD %", "Pax_Trend": "PAX Trend"}, inplace=True)
            st.dataframe(_style_table(d), use_container_width=True, height=500)
        else:
            st.info("Upload yesterday's report for DoD comparison.")

    with tab_mom:
        st.subheader("Today vs Last Month")
        if last_month_df is not None:
            mom = compare_dataframes(today_df, last_month_df, "Today", "LastMonth")
            d = mom[["Segment", "Outlet", "Location", "Revenue_Today", "Revenue_LastMonth", "Pct_Change", "Trend", "Pax_Today", "Pax_LastMonth", "Pax_Pct_Change", "Pax_Trend"]].copy()
            d.rename(columns={"Revenue_Today": "Today Revenue", "Revenue_LastMonth": "Last Month Revenue", "Pct_Change": "Rev MoM %", "Pax_Today": "Today PAX", "Pax_LastMonth": "Last Month PAX", "Pax_Pct_Change": "PAX MoM %", "Pax_Trend": "PAX Trend"}, inplace=True)
            st.dataframe(_style_table(d), use_container_width=True, height=500)
        else:
            st.info("Upload last month's report for MoM comparison.")

    with tab_yoy:
        st.subheader("Today vs Last Year")
        if last_year_df is not None:
            yoy = compare_dataframes(today_df, last_year_df, "Today", "LastYear")
            d = yoy[["Segment", "Outlet", "Location", "Revenue_Today", "Revenue_LastYear", "Pct_Change", "Trend", "Pax_Today", "Pax_LastYear", "Pax_Pct_Change", "Pax_Trend"]].copy()
            d.rename(columns={"Revenue_Today": "Today Revenue", "Revenue_LastYear": "Last Year Revenue", "Pct_Change": "Rev YoY %", "Pax_Today": "Today PAX", "Pax_LastYear": "Last Year PAX", "Pax_Pct_Change": "PAX YoY %", "Pax_Trend": "PAX Trend"}, inplace=True)
            st.dataframe(_style_table(d), use_container_width=True, height=500)
        else:
            st.info("Upload last year's report for YoY comparison.")

    with tab_full:
        st.subheader("Full Comparison Table")
        full = build_full_comparison(today_df, yesterday_df, last_month_df, last_year_df)
        if not full.empty:
            st.dataframe(_style_table(full), use_container_width=True, height=500)
        else:
            st.info("Upload comparison reports to see the full table.")


# ===================================================================
# PAGE 4 — Outlet Performance
# ===================================================================
elif page == "\U0001f3ea Outlet Performance":
    st.title("\U0001f3ea Outlet Performance")
    _check_data()

    today_df, yesterday_df, _, _ = get_analysis_data()
    if today_df is None:
        st.error("Could not load today's data.")
        st.stop()

    today_agg = aggregate_revenue(today_df)
    today_agg["Label"] = today_agg["Outlet"] + " \u2014 " + today_agg["Location"]

    col_t, col_b = st.columns(2)
    with col_t:
        st.subheader("\U0001f51d Top 10 Outlets")
        top10 = today_agg.nlargest(10, "Revenue")
        fig_t = px.bar(top10, x="Revenue", y="Label", orientation="h", color="Revenue",
                       color_continuous_scale=["#fbbf24", "#16a34a"], labels={"Revenue": "Revenue (\u20b9)", "Label": ""})
        fig_t.update_layout(height=420, margin=dict(l=10, t=10), showlegend=False)
        st.plotly_chart(fig_t, use_container_width=True)

    with col_b:
        st.subheader("\u26a0\ufe0f Bottom 10 Outlets")
        bot10 = today_agg.nsmallest(10, "Revenue")
        fig_b = px.bar(bot10, x="Revenue", y="Label", orientation="h", color="Revenue",
                       color_continuous_scale=["#dc2626", "#fbbf24"], labels={"Revenue": "Revenue (\u20b9)", "Label": ""})
        fig_b.update_layout(height=420, margin=dict(l=10, t=10), showlegend=False)
        st.plotly_chart(fig_b, use_container_width=True)

    if yesterday_df is not None and not yesterday_df.empty:
        st.markdown("---")
        st.subheader("\U0001f4ca DoD Revenue Change by Outlet")
        dod = compare_dataframes(today_df, yesterday_df, "Today", "Yesterday").dropna(subset=["Pct_Change"])
        dod["Label"] = dod["Outlet"] + " \u2014 " + dod["Location"]
        dod = dod.sort_values("Pct_Change", ascending=True)
        fig_dod = px.bar(dod, x="Pct_Change", y="Label", orientation="h", color="Pct_Change",
                         color_continuous_scale=["#dc2626", "#f5f5f5", "#16a34a"], color_continuous_midpoint=0,
                         labels={"Pct_Change": "Change %", "Label": ""})
        fig_dod.update_layout(height=max(400, len(dod) * 25), margin=dict(l=10, t=10))
        st.plotly_chart(fig_dod, use_container_width=True)

    st.markdown("---")
    st.subheader("\U0001f4b0 Revenue per PAX")
    rpp = revenue_per_pax(today_df)
    if not rpp.empty:
        rpp_d = rpp.dropna(subset=["Rev_Per_Pax"]).sort_values("Rev_Per_Pax", ascending=False)
        rpp_d = rpp_d.rename(columns={"Rev_Per_Pax": "Rev/PAX (\u20b9)", "Revenue": "Revenue (\u20b9)"})
        st.dataframe(rpp_d.head(20), use_container_width=True)


# ===================================================================
# PAGE 5 — Business Insights
# ===================================================================
elif page == "\U0001f916 Business Insights":
    st.title("\U0001f916 Business Insights")
    _check_data()

    today_df, yesterday_df, last_month_df, last_year_df = get_analysis_data()
    if today_df is None:
        st.error("Could not load today's data.")
        st.stop()

    if st.button("\U0001f4dd Generate Management Summary", type="primary", use_container_width=True):
        with st.spinner("Analyzing..."):
            report = generate_insights(today_df, yesterday_df, last_month_df, last_year_df)
        st.markdown(report)
    else:
        st.info("Click the button above to generate the management summary.")

    if yesterday_df is not None and not yesterday_df.empty:
        st.markdown("---")
        st.subheader("\U0001f50d Volume vs Spend Driver Analysis")
        vs = volume_vs_spend_analysis(today_df, yesterday_df)
        if not vs.empty:
            d = vs[["Segment", "Outlet", "Location", "Pax_Chg%", "Rev_Chg%", "RPP_Chg%", "Driver"]].copy()
            d.rename(columns={"Pax_Chg%": "PAX \u0394%", "Rev_Chg%": "Revenue \u0394%", "RPP_Chg%": "Rev/Pax \u0394%"}, inplace=True)
            pct_vs_cols = [c for c in d.columns if "%" in c]
            vs_styler = d.style.format({c: _fmt_pct_val for c in pct_vs_cols}, na_rep="\\u2014")
            st.dataframe(vs_styler, use_container_width=True, height=400)
'''

# ---------------------------------------------------------------
# README.md
# ---------------------------------------------------------------
FILES["README.md"] = """# Revenue Performance Monitoring System

## Quick Start
```
python setup_project.py
cd Revenue_Analytics_System
pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy to Streamlit Cloud
1. Push the `Revenue_Analytics_System` folder to GitHub
2. Go to share.streamlit.io -> New app -> Select repo -> Main file: app.py -> Deploy

## Upload Slots
- **Today's Report** (required)
- **Yesterday's Report** (optional - for DoD)
- **Last Month's Report** (optional - for MoM)
- **Last Year's Report** (optional - for YoY)

If you skip a comparison file, the system checks database history.
"""

# ===============================================================
# BUILD THE PROJECT
# ===============================================================

def main():
    print("=" * 50)
    print("  Revenue Analytics System — Project Setup")
    print("=" * 50)
    print()

    if os.path.exists(ROOT):
        print(f"[!] Folder '{ROOT}' already exists. Overwriting files...")
    else:
        os.makedirs(ROOT)
        print(f"[+] Created folder: {ROOT}")

    for filepath, content in FILES.items():
        full_path = os.path.join(ROOT, filepath)
        folder = os.path.dirname(full_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
            print(f"[+] Created folder: {folder}")

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[+] Created: {filepath}")

    print()
    print("=" * 50)
    print("  SETUP COMPLETE!")
    print("=" * 50)
    print()
    print("  Next steps:")
    print()
    print(f"    cd {ROOT}")
    print("    pip install -r requirements.txt")
    print("    python -m streamlit run app.py")
    print()
    print("  Then open: http://localhost:8501")
    print("=" * 50)


if __name__ == "__main__":
    main()
