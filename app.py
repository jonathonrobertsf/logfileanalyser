# Streamlit Log File Analysis for SEO (with debug output)

import io
import re
import gzip
from datetime import datetime, timezone
from typing import List

import pandas as pd
import streamlit as st
import altair as alt
from bs4 import BeautifulSoup

st.set_page_config(
    page_title="Log File Analysis for SEO",
    layout="wide",
    initial_sidebar_state="expanded",
)

alt.data_transformers.disable_max_rows()

COMBINED_REGEX = re.compile(
    r"^(?P<ip>\S+)\s+\S+\s+\S+\s+\[(?P<time>[^\]]+)\]\s+\"(?P<method>[A-Z]+)\s+(?P<path>[^\s]+)\s+HTTP/(?P<httpver>[0-9.]+)\"\s+(?P<status>\d{3})\s+(?P<bytes>\S+)(?:\s+\"(?P<referrer>[^\"]*)\"\s+\"(?P<ua>[^\"]*)\")?"
)

BOT_PATTERNS = {
    "Googlebot": re.compile(r"googlebot", re.I),
    "Bingbot": re.compile(r"bingbot", re.I),
    "YandexBot": re.compile(r"yandex(bot)?", re.I),
    "DuckDuckBot": re.compile(r"duckduck(bot)?", re.I),
    "Baiduspider": re.compile(r"baiduspider", re.I),
    "AhrefsBot": re.compile(r"ahrefs(bot)?", re.I),
    "SemrushBot": re.compile(r"semrush(bot)?", re.I),
    "MJ12bot": re.compile(r"mj12bot", re.I),
    "Applebot": re.compile(r"applebot", re.I),
    "Screaming Frog": re.compile(r"screaming\s*frog|seo\s*spider", re.I),
    "SeznamBot": re.compile(r"seznambot", re.I),
    "FacebookBot": re.compile(r"facebookexternalhit|facebookbot", re.I),
}

STATUS_BUCKETS = {
    "2xx": range(200, 300),
    "3xx": range(300, 400),
    "4xx": range(400, 500),
    "5xx": range(500, 600),
}

TIME_FORMATS = [
    "%d/%b/%Y:%H:%M:%S %z",
    "%d/%b/%Y:%H:%M:%S %z (%Z)",
]

PARAM_RE = re.compile(r"[\?&]", re.I)
DISALLOW_RE = re.compile(r"^Disallow:\s*(?P<path>\S*)", re.I)


@st.cache_data(show_spinner=False)
def _read_file(file) -> List[str]:
    name = file.name.lower()
    data = file.read()
    if name.endswith(".gz"):
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
            content = gz.read().decode("utf-8", errors="replace")
    else:
        content = data.decode("utf-8", errors="replace")
    return content.splitlines()


@st.cache_data(show_spinner=False)
def parse_logs(files: List) -> pd.DataFrame:
    rows = []
    for f in files:
        lines = _read_file(f)
        for line in lines:
            m = COMBINED_REGEX.match(line)
            if not m:
                continue
            d = m.groupdict()
            ts = None
            for fmt in TIME_FORMATS:
                try:
                    ts = datetime.strptime(d["time"], fmt)
                    break
                except Exception:
                    pass
            if ts is None:
                continue
            ua = d.get("ua") or ""
            bot = None
            for name, pat in BOT_PATTERNS.items():
                if pat.search(ua):
                    bot = name
                    break
            is_bot = bot is not None
            try:
                size = int(d.get("bytes") or 0) if d.get("bytes") != "-" else 0
            except Exception:
                size = 0
            rows.append(
                {
                    "file": f.name,
                    "ip": d["ip"],
                    "ts": ts.astimezone(timezone.utc),
                    "method": d["method"],
                    "path": d["path"],
                    "httpver": d["httpver"],
                    "status": int(d["status"]),
                    "bytes": size,
                    "referrer": d.get("referrer") or "",
                    "ua": ua,
                    "bot": bot if bot else ("Human" if ua else "Unknown"),
                    "is_bot": is_bot,
                    "is_param": bool(PARAM_RE.search(d["path"])),
                }
            )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = df["ts"].dt.date  # plain Python date objects
    df["hour"] = df["ts"].dt.hour

    def _bucket(s):
        for name, rng in STATUS_BUCKETS.items():
            if s in rng:
                return name
        return "other"

    df["status_bucket"] = df["status"].apply(_bucket)
    return df


# ----------------------------
# UI
# ----------------------------

st.title("🔎 Log File Analysis for SEO")

log_files = st.sidebar.file_uploader(
    "Upload server access logs",
    type=["log", "txt", "gz"],
    accept_multiple_files=True,
)

if not log_files:
    st.info("Add at least one log file in the sidebar to get started.")
    st.stop()

with st.spinner("Parsing logs…"):
    df = parse_logs(log_files)

if df.empty:
    st.warning("No valid log lines were parsed. Check the log format (Apache/Nginx combined).")
    st.stop()

# Debug info to see why filters might remove everything
st.write("Parsed date range:", df["date"].min(), "to", df["date"].max())
st.write("Status bucket counts:")
st.write(df["status_bucket"].value_counts())

# Sidebar filters
min_date, max_date = df["date"].min(), df["date"].max()
col1, col2 = st.sidebar.columns([1, 1])
with col1:
    date_range = st.date_input(
        "Date range", (min_date, max_date), min_value=min_date, max_value=max_date
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date
with col2:
    method_sel = st.multiselect(
        "HTTP methods", sorted(df["method"].unique()), default=sorted(df["method"].unique())
    )

status_select = st.sidebar.multiselect(
    "Status buckets", ["2xx", "3xx", "4xx", "5xx"], default=["2xx", "3xx", "4xx", "5xx"]
)
bot_select = st.sidebar.multiselect(
    "Bots/Agents", list(BOT_PATTERNS.keys()) + ["Human", "Unknown"]
)
path_filter = st.sidebar.text_input("Path contains")

st.write("Active filters:", start_date, end_date, method_sel, status_select, bot_select, path_filter)

# Apply filters
mask = (
    (df["date"] >= start_date)
    & (df["date"] <= end_date)
    & (df["method"].isin(method_sel))
    & (df["status_bucket"].isin(status_select))
)
if status_select:
    mask &= df["status_bucket"].isin(status_select)
if bot_select:
    mask &= df["bot"].isin(bot_select)
if path_filter:
    mask &= df["path"].str.contains(re.escape(path_filter), case=False, na=False)

df_filtered = df[mask]

st.write("Rows before filtering:", len(df))
st.write("Rows after filtering:", len(df_filtered))

st.subheader("Parsed log sample (pre-filter)")
st.dataframe(df.head(50))

st.subheader("Filtered log sample")
st.dataframe(df_filtered.head(50))
