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
import json
import math
import time
import hashlib
import textwrap
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
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

# Common bot patterns (conservative; not exhaustive). Add more as needed.
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
    "%d/%b/%Y:%H:%M:%S %z",  # e.g., 10/Jan/2025:13:45:02 +0000
    "%d/%b/%Y:%H:%M:%S %z (%Z)",
]

PARAM_RE = re.compile(r"[\?&]", re.I)

DISALLOW_RE = re.compile(r"^Disallow:\s*(?P<path>\S*)", re.I)

# ----------------------------
# Caching
# ----------------------------

@st.cache_data(show_spinner=False)
def _read_file(file) -> List[str]:
    # file is a streamlit UploadedFile; may be gz
    name = file.name.lower()
    data = file.read()
    if name.endswith(".gz"):
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
            content = gz.read().decode("utf-8", errors="replace")
    else:
        content = data.decode("utf-8", errors="replace")
    # Return list of lines
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
            # Parse time
            ts = None
            for fmt in TIME_FORMATS:
                try:
                    ts = datetime.strptime(d["time"], fmt)
                    break
                except Exception:
                    pass
            if ts is None:
                # Best-effort: skip if unparsable
                continue
            ua = d.get("ua") or ""
            # Identify bot label
            bot = None
            for name, pat in BOT_PATTERNS.items():
                if pat.search(ua):
                    bot = name
                    break
            is_bot = bot is not None
            # bytes may be '-'
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
    df["date"] = df["ts"].dt.date
    df["hour"] = df["ts"].dt.hour
    # status bucket
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
    # CSV or TXT list of URLs or paths
    name = file.name.lower()
    data = file.read()
    text = data.decode("utf-8", errors="replace")
    urls = []
    if name.endswith(".csv"):
        try:
            df = pd.read_csv(io.StringIO(text))
            # take first column heuristically
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
        ["Googlebot", "Bingbot", "YandexBot", "DuckDuckBot", "Baiduspider", "AhrefsBot", "SemrushBot", "MJ12bot", "Applebot", "Screaming Frog", "SeznamBot", "FacebookBot", "Human", "Unknown"],
        default=None,
    )
    date_from, date_to = None, None

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

# Date range controls, based on parsed data
min_date, max_date = df["ts"].min().date(), df["ts"].max().date()

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

# Apply filters
mask = (
    (df_filtered["date"] >= pd.to_datetime(start_date)) &
    (df_filtered["date"] <= pd.to_datetime(end_date)) &
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

# ----------------------------
# KPIs
# ----------------------------

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Total requests", f"{len(df_filtered):,}")
with k2:
    st.metric("Unique URLs", f"{df_filtered['path'].nunique():,}")
with k3:
    bot_rate = 100 * df_filtered["is_bot"].mean() if len(df_filtered) else 0
    st.metric("Bot share", f"{bot_rate:.1f}%")
with k4:
    two_xx = (df_filtered["status_bucket"] == "2xx").mean() * 100 if len(df_filtered) else 0
    st.metric("% 2xx", f"{two_xx:.1f}%")
with k5:
    param_rate = 100 * df_filtered["is_param"].mean() if len(df_filtered) else 0
    st.metric("Param URLs hit", f"{param_rate:.1f}%")

st.divider()

# ----------------------------
# Charts
# ----------------------------

left, right = st.columns([2,1])
with left:
    by_hour = (
        df_filtered
        .groupby(["date", "hour", "is_bot"], as_index=False)
        .size()
        .rename(columns={"size":"requests"})
    )
    if not by_hour.empty:
        heat = alt.Chart(by_hour).mark_rect().encode(
            x=alt.X("hour:O", title="Hour (UTC)"),
            y=alt.Y("date:T", title="Date"),
            color=alt.Color("requests:Q"),
            tooltip=["date:T","hour:O","requests:Q"]
        ).properties(title="Crawl activity heatmap")
        st.altair_chart(heat, use_container_width=True)

with right:
    by_status = (
        df_filtered
        .groupby(["status_bucket"], as_index=False)
        .size()
        .rename(columns={"size":"requests"})
    )
    if not by_status.empty:
        pie = alt.Chart(by_status).mark_arc().encode(
            theta="requests",
            color="status_bucket",
            tooltip=["status_bucket","requests"]
        ).properties(title="Status buckets")
        st.altair_chart(pie, use_container_width=True)

st.divider()

# ----------------------------
# Tables & Insights
# ----------------------------

colA, colB = st.columns(2)

with colA:
    st.subheader("Top crawled URLs by agent")
    top = (
        df_filtered
        .groupby(["bot", "path"], as_index=False)
        .size()
        .sort_values(["bot","size"], ascending=[True, False])
    )
    st.dataframe(top.head(500))
    st.download_button("Download top crawled URLs (CSV)", data=top.to_csv(index=False), file_name="top_crawled_urls.csv")

with colB:
    st.subheader("Non-200 hits (bot-first)")
    non200 = df_filtered[df_filtered["status_bucket"].isin(["3xx","4xx","5xx"])].copy()
    non200_rank = (
        non200
        .groupby(["bot","status","path"], as_index=False)
        .size()
        .sort_values(["size"], ascending=False)
    )
    st.dataframe(non200_rank.head(500))
    st.download_button("Download non-200 hits (CSV)", data=non200_rank.to_csv(index=False), file_name="non200_hits.csv")

st.subheader("Slow & heavy pages (by response bytes)")
heavy = (
    df_filtered
    .groupby("path", as_index=False)
    .agg(requests=("path","count"), bytes_avg=("bytes","mean"), bytes_sum=("bytes","sum"))
    .sort_values(["bytes_avg"], ascending=False)
)
st.dataframe(heavy.head(200))
st.download_button("Download heavy pages (CSV)", data=heavy.to_csv(index=False), file_name="heavy_pages.csv")

# ----------------------------
# robots.txt checks
# ----------------------------

if disallow_rules:
    st.subheader("robots.txt Disallow hits")
    def violates(path: str) -> Optional[str]:
        for rule in disallow_rules:
            if not rule or rule == "/":
                # disallow all
                return "/"
            if path.startswith(rule):
                return rule
        return None
    df_dis = df_filtered.copy()
    df_dis["violates"] = df_dis["path"].apply(violates)
    df_dis = df_dis[df_dis["violates"].notna()]
    if not df_dis.empty:
        viol = (
            df_dis.groupby(["bot","violates","path","status"], as_index=False)
            .size()
            .sort_values("size", ascending=False)
        )
        st.dataframe(viol.head(500))
        st.download_button("Download robots violations", data=viol.to_csv(index=False), file_name="robots_violations.csv")
    else:
        st.success("No requests matched Disallow rules in the filtered range.")

# ----------------------------
# Coverage & orphan pages (requires URL list)
# ----------------------------

if sitemap_df is not None and not sitemap_df.empty:
    st.subheader("Coverage vs URLs list")
    # normalise to path if domain is provided; else match full strings
    def normalise_url(u: str) -> str:
        u = u.strip()
        if site_domain and "://" in u:
            # pull path only when domain matches
            try:
                from urllib.parse import urlparse
                p = urlparse(u)
                if site_domain.lower() in (p.netloc or "").lower():
                    return p.path or "/"
                return u
            except Exception:
                return u
        return u

    url_list = sitemap_df.copy()
    url_list["key"] = url_list["url"].astype(str).map(normalise_url)

    seen_paths = pd.Series(df_filtered["path"].unique(), name="key")
    seen = pd.DataFrame(seen_paths)

    merged = url_list.merge(seen, on="key", how="left", indicator=True)
    not_seen = merged[merged["_merge"] == "left_only"][["url","key"]].rename(columns={"key":"path_or_url"})

    st.markdown("**URLs not seen in logs (potential orphans in the selected range):**")
    st.dataframe(not_seen.head(1000))
    st.download_button("Download not-seen URLs", data=not_seen.to_csv(index=False), file_name="urls_not_seen.csv")

    # URLs seen in logs but not present in URL list (might be junk, legacy, or missing from sitemap)
    seen_only = pd.DataFrame({"key": df_filtered["path"].unique()})
    seen_only = seen_only.merge(url_list[["key"]], on="key", how="left", indicator=True)
    extraneous = seen_only[seen_only["_merge"] == "left_only"]["key"].to_frame(name="path")

    st.markdown("**Paths seen in logs but not in URL list:**")
    st.dataframe(extraneous.head(1000))
    st.download_button("Download extra paths", data=extraneous.to_csv(index=False), file_name="paths_not_in_list.csv")

# ----------------------------
# Parameterised URLs analysis
# ----------------------------

st.subheader("Parameterised URLs")
params = (
    df_filtered[df_filtered["is_param"]]
    .groupby(["bot","path","status"], as_index=False)
    .size()
    .sort_values("size", ascending=False)
)
st.dataframe(params.head(500))
st.download_button("Download parameterised URLs", data=params.to_csv(index=False), file_name="parameterised_urls.csv")

# ----------------------------
# Raw sample
# ----------------------------

st.subheader("Sample of parsed requests")
sample_cols = ["ts","ip","method","path","status","bytes","referrer","bot","ua"]
st.dataframe(df_filtered.sort_values("ts").head(1000)[sample_cols])

# ----------------------------
# Footer
# ----------------------------

st.caption("Built for SEO log analysis • Works best with Apache/Nginx combined logs • Add more bot patterns in BOT_PATTERNS as needed.")
