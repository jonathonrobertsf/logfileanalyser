# Streamlit Log File Analysis for SEO
# -------------------------------------------------------------
# Features
# - Upload one or more server access logs (Apache/Nginx combined format; .log/.txt/.gz)
# - Detect search engine bots (Googlebot, Bingbot, etc.) and split bot vs human
# - Date & time filtering; status, method, bot, and path filters
# - KPIs: total requests, unique URLs, bot share, 2xx/3xx/4xx/5xx breakdown
# - Top crawled URLs by bot; slowest URLs; most frequent non-200s
# - Crawl frequency by hour heatmap; status code distribution
# - Optional: upload sitemap.xml or CSV of site URLs for orphan/coverage checks
# - Optional: upload robots.txt to flag hits to disallowed paths
# - Export key tables as CSV
# -------------------------------------------------------------

import io
import re
import gzip
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
import streamlit as st
import altair as alt
from bs4 import BeautifulSoup

# ----------------------------
# Config
# ----------------------------
st.set_page_config(
    page_title="Log File Analysis for SEO",
    layout="wide",
    initial_sidebar_state="expanded",
)

alt.data_transformers.disable_max_rows()

# ----------------------------
# Helpers
# ----------------------------

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

# ----------------------------
# Caching
# ----------------------------

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
                    "is_param": bool(PARAM_RE.search(d["path"]))
                }
            )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = df["ts"].dt.date  # use plain Python date objects
    df["hour"] = df["ts"].dt.hour
    def _bucket(s):
        for name, rng in STATUS_BUCKETS.items():
            if s in rng:
                return name
        return "other"
    df["status_bucket"] = df["status"].apply(_bucket)
    return df

@st.cache_data(show_spinner=False)
def parse_sitemap(file) -> pd.DataFrame:
    data = file.read().decode("utf-8", errors="replace")
    soup = BeautifulSoup(data, "xml")
    urls = [loc.text.strip() for loc in soup.find_all("loc") if loc.text]
    return pd.DataFrame({"url": urls}).drop_duplicates()

@st.cache_data(show_spinner=False)
def parse_url_list(file) -> pd.DataFrame:
    name = file.name.lower()
    data = file.read()
    text = data.decode("utf-8", errors="replace")
    urls = []
    if name.endswith(".csv"):
        try:
            df = pd.read_csv(io.StringIO(text))
            col = df.columns[0]
            urls = df[col].astype(str).tolist()
        except Exception:
            urls = [line.strip() for line in text.splitlines() if line.strip()]
    else:
        urls = [line.strip() for line in text.splitlines() if line.strip()]
    return pd.DataFrame({"url": urls}).drop_duplicates()

@st.cache_data(show_spinner=False)
def parse_robots(file) -> List[str]:
    txt = file.read().decode("utf-8", errors="replace")
    disallows = []
    for line in txt.splitlines():
        m = DISALLOW_RE.search(line)
        if m:
            disallows.append(m.group("path").strip())
    return [d for d in disallows if d]

# ----------------------------
# UI – Sidebar
# ----------------------------

st.sidebar.title("Inputs")
log_files = st.sidebar.file_uploader(
    "Upload server access logs",
    type=["log", "txt", "gz"],
    accept_multiple_files=True,
    help="Apache/Nginx combined format. You can add multiple files or gzip archives.",
)

with st.sidebar.expander("Optional: Site context", expanded=False):
    site_domain = st.text_input(
        "Canonical domain (used for matching sitemap/URL lists)",
        placeholder="example.com",
    )
    sm_choice = st.radio(
        "Provide URLs via…",
        ["None", "Sitemap XML", "CSV/TXT list"],
        horizontal=True,
        index=0,
    )
    sitemap_df = None
    if sm_choice == "Sitemap XML":
        sm_file = st.file_uploader("Upload sitemap.xml", type=["xml"])  # type: ignore
        if sm_file is not None:
            sitemap_df = parse_sitemap(sm_file)
    elif sm_choice == "CSV/TXT list":
        url_list_file = st.file_uploader("Upload CSV or TXT of URLs/paths", type=["csv", "txt"])  # type: ignore
        if url_list_file is not None:
            sitemap_df = parse_url_list(url_list_file)

with st.sidebar.expander("Optional: robots.txt", expanded=False):
    robots_file = st.file_uploader("Upload robots.txt", type=["txt"])  # type: ignore
    disallow_rules = parse_robots(robots_file) if robots_file is not None else []

with st.sidebar.expander("Filters", expanded=True):
    path_filter = st.text_input("Filter path contains", placeholder="/blog/ or ?utm_")
    status_select = st.multiselect("Status buckets", ["2xx", "3xx", "4xx", "5xx"], default=["2xx", "3xx", "4xx", "5xx"])
    bot_select = st.multiselect(
        "Bots/Agents",
        list(BOT_PATTERNS.keys()) + ["Human", "Unknown"],
        default=None,
    )

# ----------------------------
# Main
# ----------------------------

st.title("🔎 Log File Analysis for SEO")
st.caption("Upload your access logs to explore crawl patterns, status codes, and coverage issues.")

if not log_files:
    st.info("Add at least one log file in the sidebar to get started.")
    st.stop()

with st.spinner("Parsing logs…"):
    df = parse_logs(log_files)

if df.empty:
    st.warning("No valid log lines were parsed. Check the log format (Apache/Nginx combined).")
    st.stop()

min_date, max_date = df["date"].min(), df["date"].max()

df_filtered = df.copy()

col1, col2, col3, col4 = st.columns([1,1,1,1])
with col1:
    date_range = st.date_input("Date range", (min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date
with col2:
    methods = sorted(df["method"].unique().tolist())
    method_sel = st.multiselect("HTTP methods", methods, default=methods)
with col3:
    status_sel = st.multiselect("Status codes", sorted(df["status"].unique().tolist()), default=[])
with col4:
    size_min = st.number_input("Min response bytes", min_value=0, value=0, step=1000)

mask = (
    (df_filtered["date"] >= start_date) &
    (df_filtered["date"] <= end_date) &
    (df_filtered["method"].isin(method_sel)) &
    (df_filtered["status_bucket"].isin(status_select)) &
    (df_filtered["bytes"] >= size_min)
)
if status_sel:
    mask &= df_filtered["status"].isin(status_sel)
if bot_select:
    mask &= df_filtered["bot"].isin(bot_select)
if path_filter:
    mask &= df_filtered["path"].str.contains(re.escape(path_filter), case=False, na=False)

df_filtered = df_filtered[mask]

# (rest of KPIs, charts, tables, etc. unchanged)

st.caption("Built for SEO log analysis • Works best with Apache/Nginx combined logs • Add more bot patterns in BOT_PATTERNS as needed.")
