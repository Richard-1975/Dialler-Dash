"""Insurance Operations Report — decision-making dashboard."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta, datetime
import calendar as cal_mod
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
import os

import db

# ── Disk cache directory (persists across server restarts) ───────────────────────
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Font injection (Axiform — insert CDN/file URL when available) ───────────────
st.markdown("""<style>
/* Activate Axiform once CDN URL is available:
   @font-face { font-family: 'Axiform'; src: url('INSERT_URL_HERE'); }
   html, body, [class*="css"] { font-family: 'Axiform', -apple-system, sans-serif; }
*/
div[data-testid="stMetric"] label { font-size: 0.8rem; color: #888; }
</style>""", unsafe_allow_html=True)

# ── Campaign definitions ────────────────────────────────────────────────────────
MAIN_INSURANCE_CAMPAIGN = "16291"
FC_STANDALONE_EXTRA_IDS = ("25735",)
CAMPAIGNS_BY_ID = {
    "Mobi Leads 365": ("365",),
    "Mobi Leads 366": ("366",),
    "Retention":      ("25697","25824","25825","25865","25866",
                       "26052","26053","99914","FIN_RISK"),
    "Mobi Active":    ("800000","800002"),
}
CAMPAIGN_ORDER = [
    "Mobi Leads 365", "Mobi Leads 366", "Mobi Active",
    "FC Standalone", "HC Standalone",
    "Upgrades", "Lapse", "Email Opens",
    "Retention",
]
DATA_FLOOR       = "2025-12-01"   # Hard lower bound — helps query optimiser use date indexes
PROMO_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache", "list_promo_mapping.parquet")

def _all_campaign_ids():
    ids = {MAIN_INSURANCE_CAMPAIGN}
    ids.update(FC_STANDALONE_EXTRA_IDS)
    for v in CAMPAIGNS_BY_ID.values():
        ids.update(v)
    return ids

@st.cache_data(ttl=86400, show_spinner=False)
def _load_list_promo_mapping():
    """Fetch the list_id → Promotion Description mapping once per day.
    Results are persisted to disk so a VPN timeout can fall back to the
    previous day's data rather than crashing the app.

    The query is split into two parts:
      Part 1 — base promotions from tblPromotion_Generation (fast, always attempted).
      Part 2 — ViciDial re-dial lists > 100000000 joined via LIKE (slow, best-effort).
    If Part 2 times out over VPN, Part 1 alone is used so Upgrades/Lapse/HC Standalone
    etc. still classify correctly for non-redial list IDs."""
    # Return from disk immediately if file exists (epoch check deletes it at 08:00)
    if os.path.exists(PROMO_CACHE_PATH):
        return pd.read_parquet(PROMO_CACHE_PATH)

    # Only fetch promotions whose Description matches a pattern we actually use in
    # _campaign_case(). The full tblPromotion_Generation table has thousands of rows
    # going back to 2007 — injecting all of them as a VALUES literal crashes SQL Server.
    _RELEVANT_DESC = (
        "Description LIKE '%FC STANDALONE%'"
        " OR Description LIKE '%FC_STANDALONE%'"
        " OR Description LIKE '%FC ACTIVE STANDALONE%'"
        " OR Description LIKE '%FC PAID UP STANDALONE%'"
        " OR Description LIKE '%FC_PU STANDALONE%'"
        " OR Description LIKE '%EXT STANDALONE%'"
        " OR Description LIKE '%HC STANDALONE%'"
        " OR Description LIKE '%HC_STANDALONE%'"
        " OR Description LIKE '%UPGRADE%'"
        " OR Description LIKE '%LAPSE%'"
        " OR Description LIKE '%EMAIL OPEN%'"
        " OR Description LIKE '%EMAILOPEN%'"
        " OR Description LIKE '%Accident Cover Email%'"
    )

    # Part 1 — fast: direct table scan, no cross-DB join
    try:
        base_df = db.query(f"""
            SELECT CAST(Promotion_ID AS VARCHAR) AS list_id_str, Description
            FROM CreditEase_Snapshot..tblPromotion_Generation
            WHERE {_RELEVANT_DESC}
        """)
    except Exception:
        # Can't even reach the DB — fall back to stale disk data or empty
        if os.path.exists(PROMO_CACHE_PATH):
            return pd.read_parquet(PROMO_CACHE_PATH)
        return pd.DataFrame(columns=["list_id_str", "Description"])

    # Part 2 — slow: cross-DB LIKE join for ViciDial redial lists (best-effort, 45s cap)
    try:
        redial_df = db.query(f"""
            SELECT CAST(vl.list_id AS VARCHAR) AS list_id_str, p.Description
            FROM extract.vicidial_vicidial_lists vl
            JOIN CreditEase_Snapshot..tblPromotion_Generation p
                ON (   vl.list_name LIKE 'CAMPAIGN'  + CAST(p.Promotion_ID AS VARCHAR) + ' %'
                    OR vl.list_name LIKE 'CAMPAIGN:' + CAST(p.Promotion_ID AS VARCHAR) + ': %')
            WHERE vl.list_id > 100000000
              AND vl.list_lastcalldate >= '2025-12-01'
              AND ({_RELEVANT_DESC.replace('Description', 'p.Description')})
        """, timeout=45)
    except Exception:
        redial_df = pd.DataFrame(columns=["list_id_str", "Description"])

    df = (pd.concat([base_df, redial_df], ignore_index=True)
            .dropna(subset=["Description"])
            .drop_duplicates(subset=["list_id_str"])
            .reset_index(drop=True))
    df.to_parquet(PROMO_CACHE_PATH, index=False)
    return df


def _list_promo_cte():
    """Build the list_promo CTE by injecting the cached mapping as SQL VALUES literals.
    The expensive cross-DB LIKE join runs at most once per hour instead of every query."""
    df = _load_list_promo_mapping()
    if df.empty:
        # Fallback: empty table so LEFT JOIN produces NULLs for all rows
        return ("list_promo (list_id_str, Description) AS "
                "(SELECT TOP 0 CAST('' AS VARCHAR(50)), CAST('' AS VARCHAR(500)))")
    lids   = df["list_id_str"].astype(str).str.replace("'", "''", regex=False)
    descs  = df["Description"].astype(str).str.replace("'", "''", regex=False)
    values = ",".join("('" + lids + "','" + descs + "')")
    return (
        "list_promo (list_id_str, Description) AS (\n"
        "        SELECT v.list_id_str, v.Description\n"
        f"        FROM (VALUES {values}) AS v (list_id_str, Description)\n"
        "    )"
    )

def _campaign_case():
    ret_ids  = ",".join(f"'{i}'" for i in CAMPAIGNS_BY_ID["Retention"])
    mobi_ids = ",".join(f"'{i}'" for i in CAMPAIGNS_BY_ID["Mobi Active"])
    fc_extra = ",".join(f"'{i}'" for i in FC_STANDALONE_EXTRA_IDS)
    return f"""CASE
        WHEN a.campaign_id = '365'  THEN 'Mobi Leads 365'
        WHEN a.campaign_id = '366'  THEN 'Mobi Leads 366'
        WHEN a.campaign_id IN ({mobi_ids}) THEN 'Mobi Active'
        WHEN a.campaign_id IN ({ret_ids}) THEN 'Retention'
        WHEN a.campaign_id IN ({fc_extra})
          OR lp.Description LIKE '%FC STANDALONE%'
          OR lp.Description LIKE '%FC_STANDALONE%'
          OR lp.Description LIKE '%FC ACTIVE STANDALONE%'
          OR lp.Description LIKE '%FC PAID UP STANDALONE%'
          OR lp.Description LIKE '%FC_PU STANDALONE%'
          OR lp.Description LIKE '%EXT STANDALONE%'
          OR (a.campaign_id = '{MAIN_INSURANCE_CAMPAIGN}' AND lp.Description IS NULL)
                                    THEN 'FC Standalone'
        WHEN lp.Description LIKE '%HC STANDALONE%'
          OR lp.Description LIKE '%HC_STANDALONE%'
                                    THEN 'HC Standalone'
        WHEN lp.Description LIKE '%UPGRADE%'
          OR lp.Description LIKE '%Funeral Upgrades%'
          OR lp.Description LIKE '%Insurance Upgrades%'
                                    THEN 'Upgrades'
        WHEN lp.Description LIKE '%LAPSE%'
          OR lp.Description LIKE '%Funeral Lapses%'
          OR lp.Description LIKE '%Insurance Lapses%'
                                    THEN 'Lapse'
        WHEN lp.Description LIKE '%EMAIL OPEN%'
          OR lp.Description LIKE '%EMAILOPEN%'
          OR lp.Description LIKE '%Accident Cover Email%'
                                    THEN 'Email Opens'
        ELSE 'Other'
    END"""

# ── Cache epoch — refreshes automatically at 08:00 each day ─────────────────────
def _business_epoch() -> str:
    """Returns the current business-day epoch string.
    Before 08:00 the previous day's date is returned so overnight sessions
    continue to use the prior day's cache until the refresh window opens."""
    now = datetime.now()
    if now.hour < 8:
        return str((now - timedelta(days=1)).date())
    return str(now.date())


# ── Date helpers ────────────────────────────────────────────────────────────────
def get_comparison_dates(d_from, d_to, comp_type):
    n_days = (d_to - d_from).days + 1
    if comp_type == "Previous period":
        c_to   = d_from - timedelta(days=1)
        c_from = c_to - timedelta(days=n_days - 1)
    elif comp_type == "WTD":
        c_from = d_from - timedelta(days=7)
        c_to   = d_to   - timedelta(days=7)
    else:  # MTD
        pm = d_from.month - 1 if d_from.month > 1 else 12
        py = d_from.year if d_from.month > 1 else d_from.year - 1
        ld = cal_mod.monthrange(py, pm)[1]
        c_from = d_from.replace(year=py, month=pm, day=min(d_from.day, ld))
        c_to   = d_to.replace(year=py,   month=pm, day=min(d_to.day,   ld))
    return c_from, c_to

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Insurance")

    view = st.radio(
        "Navigate to",
        ["Operations", "Campaign", "Agent"],
        label_visibility="collapsed",
    )

    st.divider()

    date_from = st.date_input("From", value=date(2026, 1, 1), min_value=date(2026, 1, 1))
    date_to   = st.date_input("To",   value=date.today(),    min_value=date(2026, 1, 1))

    comparison_type = st.selectbox(
        "Compare to",
        ["Previous period", "WTD", "MTD"],
    )
    comp_from, comp_to = get_comparison_dates(date_from, date_to, comparison_type)
    st.caption(f"Comparison: {comp_from:%d %b} – {comp_to:%d %b %Y}")

    st.divider()
    st.caption(f"DB: {db.DATABASE}")

# ── Cache tier helpers ────────────────────────────────────────────────────────────
RECENT_CUTOFF_DAYS = 7  # Data within this many days of today is considered "live"


def _is_historical(ct: str, pt: str) -> bool:
    """Return True when both period end-dates fall outside the live window.
    Historical data never changes so it can be cached indefinitely."""
    cutoff = str(date.today() - timedelta(days=RECENT_CUTOFF_DAYS))
    return ct < cutoff and pt < cutoff


def _disk_path(name: str, cf: str, ct: str, pf: str, pt: str) -> str:
    """Deterministic Parquet file path for a historical (frozen) query result."""
    return os.path.join(CACHE_DIR, f"{name}__{cf}__{ct}__{pf}__{pt}.parquet")


def _live_disk_path(name: str, cf: str, ct: str, pf: str, pt: str) -> str:
    """Parquet path for the most-recent live query result (deleted at 08:00 epoch)."""
    return os.path.join(CACHE_DIR, f"live__{name}__{cf}__{ct}__{pf}__{pt}.parquet")


def _is_connection_error(exc: Exception) -> bool:
    """Return True when the exception is a DB server unreachable error (e.g. VPN down)."""
    return "20009" in str(exc) or "Unable to connect" in str(exc)


def _load_from_disk_or_db(name: str, sql_fn, cf: str, ct: str, pf: str, pt: str) -> pd.DataFrame:
    """Return historical data from disk if already cached, otherwise query the DB,
    persist the result to disk, and return it. The DB is never contacted again for
    the same parameters once the Parquet file exists."""
    path = _disk_path(name, cf, ct, pf, pt)
    if os.path.exists(path):
        return pd.read_parquet(path)
    df = db.query(sql_fn(cf, ct, pf, pt))
    df.to_parquet(path, index=False)
    return df


# ── Data loaders ────────────────────────────────────────────────────────────────
# Each loader has:
#   _<name>_sql()   — builds the SQL string (shared between tiers)
#   _<name>_frozen  — ttl=None  (historical: never expires)
#   _<name>_live    — ttl=86400 (recent: refreshes at 08:00 via epoch check)
#   load_<name>()   — dispatcher: routes to the right tier

def _insurance_sql(cf, ct, pf, pt):
    case_expr = _campaign_case()
    cte       = _list_promo_cte()
    all_ids   = ",".join(f"'{i}'" for i in _all_campaign_ids())
    return f"""
        WITH {cte},
        base_calls AS (
            SELECT
                CASE WHEN a.call_date >= '{cf}' AND a.call_date < DATEADD(day,1,CAST('{ct}' AS DATE))
                     THEN 'current' ELSE 'comparison' END AS Period,
                {case_expr}                               AS Campaign,
                a.lead_id
            FROM extract.vicidial_vicidial_log a WITH (NOLOCK)
            LEFT JOIN list_promo lp
                ON CAST(a.list_id AS VARCHAR) = lp.list_id_str
            WHERE a.campaign_id IN ({all_ids})
              AND a.call_date >= '{DATA_FLOOR}'
              AND (
                  (a.call_date >= '{cf}' AND a.call_date < DATEADD(day,1,CAST('{ct}' AS DATE)))
               OR (a.call_date >= '{pf}' AND a.call_date < DATEADD(day,1,CAST('{pt}' AS DATE)))
              )
        ),
        call_counts AS (
            SELECT Period, Campaign,
                   COUNT(*)                  AS Calls,
                   COUNT(DISTINCT lead_id)   AS UniqueCustomers
            FROM base_calls
            GROUP BY Period, Campaign
        ),
        base_agent AS (
            SELECT
                CASE WHEN a.event_time >= '{cf}' AND a.event_time < DATEADD(day,1,CAST('{ct}' AS DATE))
                     THEN 'current' ELSE 'comparison' END                  AS Period,
                {case_expr}                                                 AS Campaign,
                CAST(a.talk_time AS BIGINT)                                AS talk_time,
                CASE WHEN m.isConnect = 1 OR a.status = 'ACSALE' THEN 1 ELSE 0 END AS is_connect,
                CASE WHEN m.isRPC     = 1                         THEN 1 ELSE 0 END AS is_rpc,
                CASE WHEN vs.sale     = 'Y'                       THEN 1 ELSE 0 END AS is_sale
            FROM transform.trfmDiallerAgentLog a WITH (NOLOCK)
            LEFT JOIN list_promo lp
                ON CAST(a.list_id AS VARCHAR) = lp.list_id_str
            LEFT JOIN extract.mds_vwFCVicidialStatusMapping m WITH (NOLOCK)
                ON a.status = m.DiallerStatus
            LEFT JOIN extract.vicidial_vicidial_statuses vs WITH (NOLOCK)
                ON a.status = vs.status
            WHERE a.campaign_id IN ({all_ids})
              AND a.lead_id > 0
              AND a.event_time >= '{DATA_FLOOR}'
              AND (
                  (a.event_time >= '{cf}' AND a.event_time < DATEADD(day,1,CAST('{ct}' AS DATE)))
               OR (a.event_time >= '{pf}' AND a.event_time < DATEADD(day,1,CAST('{pt}' AS DATE)))
              )
        ),
        agent_metrics AS (
            SELECT Period, Campaign,
                   ISNULL(SUM(talk_time), 0) / 3600.0 AS HoursWorked,
                   SUM(is_connect)                     AS Connects,
                   SUM(is_rpc)                         AS RPC,
                   SUM(is_sale)                        AS Sales
            FROM base_agent
            GROUP BY Period, Campaign
        )
        SELECT
            c.Period,
            c.Campaign,
            ISNULL(am.HoursWorked, 0) AS HoursWorked,
            c.UniqueCustomers,
            c.Calls,
            ISNULL(am.Connects, 0)    AS Connects,
            ISNULL(am.RPC, 0)         AS RPC,
            ISNULL(am.Sales, 0)       AS Sales
        FROM call_counts c
        LEFT JOIN agent_metrics am
            ON c.Period = am.Period AND c.Campaign = am.Campaign
    """

@st.cache_data(ttl=None, show_spinner=False)
def _load_insurance_frozen(cf, ct, pf, pt):
    return _load_from_disk_or_db("insurance", _insurance_sql, cf, ct, pf, pt)

@st.cache_data(ttl=86400, show_spinner=False)
def _load_insurance_live(cf, ct, pf, pt):
    path = _live_disk_path("insurance", cf, ct, pf, pt)
    try:
        df = db.query(_insurance_sql(cf, ct, pf, pt))
        df.to_parquet(path, index=False)
        return df, False
    except Exception as e:
        if os.path.exists(path):
            return pd.read_parquet(path), True
        raise

def load_insurance(cf, ct, pf, pt):
    if _is_historical(ct, pt):
        return _load_insurance_frozen(cf, ct, pf, pt), False
    return _load_insurance_live(cf, ct, pf, pt)


def _funnel_sql(cf, ct, pf, pt):
    # Exclude 365/366 — those campaigns generate tens of millions of rows in
    # vicidial_vicidial_log and cause DISTINCT-count scans to time out.
    all_ids = ",".join(f"'{i}'" for i in _all_campaign_ids() - {"365", "366"})
    return f"""
        WITH funnel_calls_base AS (
            SELECT
                CASE WHEN call_date >= '{cf}' AND call_date < DATEADD(day,1,CAST('{ct}' AS DATE))
                     THEN 'current' ELSE 'comparison' END AS Period,
                lead_id,
                status
            FROM extract.vicidial_vicidial_log WITH (NOLOCK)
            WHERE campaign_id IN ({all_ids})
              AND call_date >= '{DATA_FLOOR}'
              AND (
                  (call_date >= '{cf}' AND call_date < DATEADD(day,1,CAST('{ct}' AS DATE)))
               OR (call_date >= '{pf}' AND call_date < DATEADD(day,1,CAST('{pt}' AS DATE)))
              )
        ),
        funnel_calls AS (
            SELECT
                Period,
                COUNT(DISTINCT lead_id)                                              AS UniqueCustomers,
                COUNT(DISTINCT CASE WHEN status NOT IN ('AA','NA','AB','AL','ADC','PDROP')
                                    THEN lead_id END)                                AS UniqueHit
            FROM funnel_calls_base
            GROUP BY Period
        ),
        funnel_agent_base AS (
            SELECT
                CASE WHEN a.event_time >= '{cf}' AND a.event_time < DATEADD(day,1,CAST('{ct}' AS DATE))
                     THEN 'current' ELSE 'comparison' END                           AS Period,
                CASE WHEN m.isConnect = 1 OR a.status = 'ACSALE' THEN a.lead_id END AS connect_lead,
                CASE WHEN m.isRPC = 1                             THEN a.lead_id END AS rpc_lead,
                CASE WHEN vs.sale = 'Y'                           THEN a.lead_id END AS sale_lead
            FROM transform.trfmDiallerAgentLog a WITH (NOLOCK)
            LEFT JOIN extract.mds_vwFCVicidialStatusMapping m WITH (NOLOCK)
                ON a.status = m.DiallerStatus
            LEFT JOIN extract.vicidial_vicidial_statuses vs WITH (NOLOCK)
                ON a.status = vs.status
            WHERE a.campaign_id IN ({all_ids})
              AND a.lead_id > 0
              AND a.event_time >= '{DATA_FLOOR}'
              AND (
                  (a.event_time >= '{cf}' AND a.event_time < DATEADD(day,1,CAST('{ct}' AS DATE)))
               OR (a.event_time >= '{pf}' AND a.event_time < DATEADD(day,1,CAST('{pt}' AS DATE)))
              )
        ),
        funnel_agent AS (
            SELECT
                Period,
                COUNT(DISTINCT connect_lead) AS UniqueConnects,
                COUNT(DISTINCT rpc_lead)     AS UniqueRPC,
                COUNT(DISTINCT sale_lead)    AS UniqueSales
            FROM funnel_agent_base
            GROUP BY Period
        )
        SELECT
            fc.Period,
            fc.UniqueCustomers,
            fc.UniqueHit,
            ISNULL(fa.UniqueConnects, 0) AS UniqueConnects,
            ISNULL(fa.UniqueRPC,      0) AS UniqueRPC,
            ISNULL(fa.UniqueSales,    0) AS UniqueSales
        FROM funnel_calls fc
        LEFT JOIN funnel_agent fa ON fc.Period = fa.Period
    """

@st.cache_data(ttl=None, show_spinner=False)
def _load_funnel_frozen(cf, ct, pf, pt):
    return _load_from_disk_or_db("funnel", _funnel_sql, cf, ct, pf, pt)

@st.cache_data(ttl=86400, show_spinner=False)
def _load_funnel_live(cf, ct, pf, pt):
    path = _live_disk_path("funnel", cf, ct, pf, pt)
    try:
        df = db.query(_funnel_sql(cf, ct, pf, pt))
        df.to_parquet(path, index=False)
        return df, False
    except Exception as e:
        if os.path.exists(path):
            return pd.read_parquet(path), True
        raise

def load_funnel(cf, ct, pf, pt):
    if _is_historical(ct, pt):
        return _load_funnel_frozen(cf, ct, pf, pt), False
    return _load_funnel_live(cf, ct, pf, pt)


def _daily_trend_sql(cf, ct, pf, pt):
    # Same exclusion as funnel — 365/366 volume causes vicidial_vicidial_log scans to time out.
    all_ids = ",".join(f"'{i}'" for i in _all_campaign_ids() - {"365", "366"})
    return f"""
        WITH daily_calls_base AS (
            SELECT
                CASE WHEN call_date >= '{cf}' AND call_date < DATEADD(day,1,CAST('{ct}' AS DATE))
                     THEN 'current' ELSE 'comparison' END AS Period,
                CAST(call_date AS DATE)                   AS Date,
                lead_id
            FROM extract.vicidial_vicidial_log WITH (NOLOCK)
            WHERE campaign_id IN ({all_ids})
              AND call_date >= '{DATA_FLOOR}'
              AND (
                  (call_date >= '{cf}' AND call_date < DATEADD(day,1,CAST('{ct}' AS DATE)))
               OR (call_date >= '{pf}' AND call_date < DATEADD(day,1,CAST('{pt}' AS DATE)))
              )
        ),
        daily_calls AS (
            SELECT Period, Date,
                   COUNT(*)                AS Calls,
                   COUNT(DISTINCT lead_id) AS UniqueCustomers
            FROM daily_calls_base
            GROUP BY Period, Date
        ),
        daily_agent_base AS (
            SELECT
                CASE WHEN a.event_time >= '{cf}' AND a.event_time < DATEADD(day,1,CAST('{ct}' AS DATE))
                     THEN 'current' ELSE 'comparison' END        AS Period,
                CAST(a.event_time AS DATE)                       AS Date,
                CASE WHEN m.isConnect = 1 OR a.status = 'ACSALE'
                     THEN 1 ELSE 0 END                           AS is_connect,
                CASE WHEN vs.sale = 'Y' THEN 1 ELSE 0 END        AS is_sale
            FROM transform.trfmDiallerAgentLog a WITH (NOLOCK)
            LEFT JOIN extract.mds_vwFCVicidialStatusMapping m WITH (NOLOCK)
                ON a.status = m.DiallerStatus
            LEFT JOIN extract.vicidial_vicidial_statuses vs WITH (NOLOCK)
                ON a.status = vs.status
            WHERE a.campaign_id IN ({all_ids})
              AND a.lead_id > 0
              AND a.event_time >= '{DATA_FLOOR}'
              AND (
                  (a.event_time >= '{cf}' AND a.event_time < DATEADD(day,1,CAST('{ct}' AS DATE)))
               OR (a.event_time >= '{pf}' AND a.event_time < DATEADD(day,1,CAST('{pt}' AS DATE)))
              )
        ),
        daily_agent AS (
            SELECT Period, Date,
                   SUM(is_connect) AS Connects,
                   SUM(is_sale)    AS Sales
            FROM daily_agent_base
            GROUP BY Period, Date
        )
        SELECT
            dc.Period,
            dc.Date,
            dc.Calls,
            dc.UniqueCustomers,
            ISNULL(da.Connects, 0) AS Connects,
            ISNULL(da.Sales, 0)    AS Sales
        FROM daily_calls dc
        LEFT JOIN daily_agent da ON dc.Period = da.Period AND dc.Date = da.Date
        ORDER BY dc.Period, dc.Date
    """

@st.cache_data(ttl=None, show_spinner=False)
def _load_daily_trend_frozen(cf, ct, pf, pt):
    return _load_from_disk_or_db("daily_trend", _daily_trend_sql, cf, ct, pf, pt)

@st.cache_data(ttl=86400, show_spinner=False)
def _load_daily_trend_live(cf, ct, pf, pt):
    path = _live_disk_path("daily_trend", cf, ct, pf, pt)
    try:
        df = db.query(_daily_trend_sql(cf, ct, pf, pt))
        df.to_parquet(path, index=False)
        return df, False
    except Exception as e:
        if os.path.exists(path):
            return pd.read_parquet(path), True
        raise

def load_daily_trend(cf, ct, pf, pt):
    if _is_historical(ct, pt):
        return _load_daily_trend_frozen(cf, ct, pf, pt), False
    return _load_daily_trend_live(cf, ct, pf, pt)


def _agent_sql(cf, ct, pf, pt):
    all_ids = ",".join(f"'{i}'" for i in _all_campaign_ids())
    return f"""
        WITH base_agent AS (
            -- Agent-level metrics: connects, RPC, sales, talk time
            SELECT
                a.[user],
                CASE WHEN a.event_time >= '{cf}' AND a.event_time < DATEADD(day,1,CAST('{ct}' AS DATE))
                     THEN 'current' ELSE 'comparison' END                  AS Period,
                CAST(a.talk_time AS BIGINT)                                AS talk_time,
                a.lead_id,
                CASE WHEN m.isConnect = 1 OR a.status = 'ACSALE' THEN 1 ELSE 0 END AS is_connect,
                CASE WHEN m.isRPC     = 1                         THEN 1 ELSE 0 END AS is_rpc,
                CASE WHEN vs.sale     = 'Y'                       THEN 1 ELSE 0 END AS is_sale
            FROM transform.trfmDiallerAgentLog a WITH (NOLOCK)
            LEFT JOIN extract.mds_vwFCVicidialStatusMapping m WITH (NOLOCK)
                ON a.status = m.DiallerStatus
            LEFT JOIN extract.vicidial_vicidial_statuses vs WITH (NOLOCK)
                ON a.status = vs.status
            WHERE a.campaign_id IN ({all_ids})
              AND a.lead_id > 0
              AND a.event_time >= '{DATA_FLOOR}'
              AND (
                  (a.event_time >= '{cf}' AND a.event_time < DATEADD(day,1,CAST('{ct}' AS DATE)))
               OR (a.event_time >= '{pf}' AND a.event_time < DATEADD(day,1,CAST('{pt}' AS DATE)))
              )
        ),
        base_avail AS (
            -- Agent login/availability segments from the agent log
            SELECT
                [user],
                CASE WHEN event_time >= '{cf}' AND event_time < DATEADD(day,1,CAST('{ct}' AS DATE))
                     THEN 'current' ELSE 'comparison' END AS Period,
                ISNULL(talk_sec, 0)  AS talk_sec,
                ISNULL(wait_sec, 0)  AS wait_sec,
                ISNULL(dispo_sec, 0) AS dispo_sec,
                ISNULL(pause_sec, 0) AS pause_sec
            FROM extract.vicidial_vicidial_agent_log WITH (NOLOCK)
            WHERE campaign_id IN ({all_ids})
              AND [user] IS NOT NULL AND [user] != ''
              AND event_time >= '{DATA_FLOOR}'
              AND (
                  (event_time >= '{cf}' AND event_time < DATEADD(day,1,CAST('{ct}' AS DATE)))
               OR (event_time >= '{pf}' AND event_time < DATEADD(day,1,CAST('{pt}' AS DATE)))
              )
        ),
        agent_metrics AS (
            SELECT
                [user], Period,
                ISNULL(SUM(talk_time), 0) / 3600.0 AS TalkHrs,
                COUNT(DISTINCT lead_id)             AS UniqueCustomers,
                SUM(is_connect)                     AS Connects,
                SUM(is_rpc)                         AS RPC,
                SUM(is_sale)                        AS Sales
            FROM base_agent
            GROUP BY [user], Period
        ),
        avail_metrics AS (
            -- Calls   = every call event routed to the agent (incl. quick hangups before connect)
            -- LoggedHrs = all time on dialler (talk + wait + dispo + pause)
            -- AvailHrs  = time available to take calls (excludes pause)
            SELECT
                [user], Period,
                COUNT(*)                                                  AS Calls,
                SUM(talk_sec + wait_sec + dispo_sec + pause_sec) / 3600.0 AS LoggedHrs,
                SUM(talk_sec + wait_sec + dispo_sec)             / 3600.0 AS AvailHrs
            FROM base_avail
            GROUP BY [user], Period
        )
        SELECT
            am.[user]                                        AS Agent,
            ISNULL(u.full_name, am.[user])                   AS AgentName,
            ISNULL(ug.group_name, ISNULL(u.user_group, ''))  AS Team,
            am.Period,
            ISNULL(av.Calls,     0)                 AS Calls,
            am.TalkHrs,
            ISNULL(av.AvailHrs,  0)                 AS AvailHrs,
            ISNULL(av.LoggedHrs, 0)                 AS LoggedHrs,
            am.UniqueCustomers,
            am.Connects,
            am.RPC,
            am.Sales
        FROM agent_metrics am
        LEFT JOIN avail_metrics av
            ON am.[user] = av.[user] AND am.Period = av.Period
        LEFT JOIN extract.vicidial_vicidial_users u WITH (NOLOCK)
            ON am.[user] = u.[user]
        LEFT JOIN extract.vicidial_vicidial_user_groups ug WITH (NOLOCK)
            ON u.user_group = ug.user_group
    """

@st.cache_data(ttl=None, show_spinner=False)
def _load_agent_frozen(cf, ct, pf, pt):
    return _load_from_disk_or_db("agent", _agent_sql, cf, ct, pf, pt)

@st.cache_data(ttl=86400, show_spinner=False)
def _load_agent_live(cf, ct, pf, pt):
    path = _live_disk_path("agent", cf, ct, pf, pt)
    try:
        df = db.query(_agent_sql(cf, ct, pf, pt))
        df.to_parquet(path, index=False)
        return df, False
    except Exception as e:
        if os.path.exists(path):
            return pd.read_parquet(path), True
        raise

def load_agent(cf, ct, pf, pt):
    if _is_historical(ct, pt):
        return _load_agent_frozen(cf, ct, pf, pt), False
    return _load_agent_live(cf, ct, pf, pt)


# ── Background pre-warm ──────────────────────────────────────────────────────────
def _prewarm():
    """Populate the disk cache for the full historical range (DATA_FLOOR → cutoff).
    Skips any query whose Parquet file already exists on disk.
    Runs once per server lifetime in a daemon thread — never blocks the UI."""
    try:
        cutoff  = date.today() - timedelta(days=RECENT_CUTOFF_DAYS)
        ct_date = cutoff - timedelta(days=1)
        cf      = DATA_FLOOR
        ct      = str(ct_date)
        if ct < cf:
            return  # No historical data available yet
        n_days  = (ct_date - date.fromisoformat(cf)).days + 1
        pt_date = date.fromisoformat(cf) - timedelta(days=1)
        pf_date = pt_date - timedelta(days=n_days - 1)
        pf, pt  = str(pf_date), str(pt_date)
        tasks = []
        with ThreadPoolExecutor(max_workers=4) as ex:
            if not os.path.exists(_disk_path("insurance",   cf, ct, pf, pt)):
                tasks.append(ex.submit(_load_insurance_frozen,   cf, ct, pf, pt))
            if not os.path.exists(_disk_path("funnel",      cf, ct, pf, pt)):
                tasks.append(ex.submit(_load_funnel_frozen,      cf, ct, pf, pt))
            if not os.path.exists(_disk_path("daily_trend", cf, ct, pf, pt)):
                tasks.append(ex.submit(_load_daily_trend_frozen, cf, ct, pf, pt))
            if not os.path.exists(_disk_path("agent",       cf, ct, pf, pt)):
                tasks.append(ex.submit(_load_agent_frozen,       cf, ct, pf, pt))
    except Exception:
        pass  # Pre-warm is best-effort; failures are silent


@st.cache_resource
def _start_prewarm():
    """cache_resource runs once per server lifetime — ensures pre-warm fires exactly once."""
    Thread(target=_prewarm, daemon=True).start()

_start_prewarm()


# ── Normalise + split by period ─────────────────────────────────────────────────
MAIN_COLS = ["HoursWorked", "UniqueCustomers", "Calls", "Connects", "RPC", "Sales"]

def normalise_split(df):
    """Split the combined Period dataframe into (current_df, comparison_df),
    both indexed by Campaign with zero rows for inactive campaigns."""
    empty = pd.DataFrame(0, index=CAMPAIGN_ORDER, columns=MAIN_COLS)
    if df is None or df.empty:
        return empty.copy(), empty.copy()

    df = df[df["Campaign"] != "Other"].copy()
    for col in MAIN_COLS:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)

    def _norm(period_df):
        if period_df.empty:
            return empty.copy()
        return (period_df.drop(columns=["Period"])
                         .set_index("Campaign")
                         .reindex(CAMPAIGN_ORDER, fill_value=0))

    curr = _norm(df[df["Period"] == "current"].copy())
    comp = _norm(df[df["Period"] == "comparison"].copy())
    return curr, comp


# ── Refresh button (placed here so loader functions are already defined) ─────────
with st.sidebar:
    if st.button("🔄 Refresh data"):
        _load_insurance_live.clear()
        _load_funnel_live.clear()
        _load_daily_trend_live.clear()
        _load_agent_live.clear()
        _load_list_promo_mapping.clear()
        if os.path.exists(PROMO_CACHE_PATH):
            os.remove(PROMO_CACHE_PATH)
        for f in os.listdir(CACHE_DIR):
            if f.startswith("live__"):
                os.remove(os.path.join(CACHE_DIR, f))
        st.rerun()

# ── Auto-refresh live cache at 08:00 ─────────────────────────────────────────────
# Only the live (recent) tier is cleared — frozen historical data is never touched.
_epoch = _business_epoch()
if st.session_state.get("cache_epoch") != _epoch:
    _load_insurance_live.clear()
    _load_funnel_live.clear()
    _load_daily_trend_live.clear()
    _load_agent_live.clear()
    _load_list_promo_mapping.clear()
    if os.path.exists(PROMO_CACHE_PATH):
        os.remove(PROMO_CACHE_PATH)
    for _f in os.listdir(CACHE_DIR):
        if _f.startswith("live__"):
            os.remove(os.path.join(CACHE_DIR, _f))
    st.session_state["cache_epoch"] = _epoch

cf, ct = str(date_from), str(date_to)
pf, pt = str(comp_from), str(comp_to)


def _load_series(combined):
    raw, comp = normalise_split(combined)
    hours     = raw["HoursWorked"].astype(float)
    customers = raw["UniqueCustomers"].astype(float)
    calls     = raw["Calls"].astype(float)
    connects  = raw["Connects"].astype(float)
    rpc       = raw["RPC"].astype(float)
    sales     = raw["Sales"].astype(float)
    return raw, comp, hours, customers, calls, connects, rpc, sales

# ── Helpers ─────────────────────────────────────────────────────────────────────
def safe_div(num, den):
    if hasattr(den, "where"):
        return num.where(den > 0, other=0) / den.where(den > 0, other=1)
    return (num / den) if den > 0 else 0

def _delta_str(curr, prev, is_pct=False):
    d = curr - prev
    if is_pct:
        return f"{d:+.1f}pp"
    if abs(d) >= 10:
        return f"{d:+,.0f}"
    return f"{d:+.2f}"

def kpi(col, label, curr, prev, fmt, is_pct=False, inverse=False):
    col.metric(
        label=label,
        value=fmt(curr),
        delta=_delta_str(curr, prev, is_pct),
        delta_color="inverse" if inverse else "normal",
    )

# ── Totals ──────────────────────────────────────────────────────────────────────
def _totals(r):
    T = {c: float(r[c].sum()) for c in MAIN_COLS}
    T["ConnectRate"]  = safe_div(T["Connects"],  T["Calls"])    * 100
    T["RPCRate"]      = safe_div(T["RPC"],        T["Connects"]) * 100
    T["SalesPerRPC"]  = safe_div(T["Sales"],      T["RPC"])      * 100
    T["Sales1k"]      = safe_div(T["Sales"] * 1000, T["Calls"])
    T["SalesPerConn"] = safe_div(T["Sales"],      T["Connects"]) * 100
    T["SalesPerHr"]   = safe_div(T["Sales"],      T["HoursWorked"])
    T["ConnPerHr"]    = safe_div(T["Connects"],   T["HoursWorked"])
    return T


# ══════════════════════════════════════════════════════════════════════════════
# OPERATIONS VIEW
# ══════════════════════════════════════════════════════════════════════════════
if view == "Operations":
    with st.spinner("Loading data…"):
        with ThreadPoolExecutor(max_workers=3) as ex:
            f_ins    = ex.submit(load_insurance,    cf, ct, pf, pt)
            f_funnel = ex.submit(load_funnel,       cf, ct, pf, pt)
            f_trend  = ex.submit(load_daily_trend,  cf, ct, pf, pt)
        try:
            combined,  ins_stale   = f_ins.result(timeout=300)
            funnel_df, fun_stale   = f_funnel.result(timeout=300)
            trend_df,  trend_stale = f_trend.result(timeout=300)
        except TimeoutError:
            st.error("Query timed out after 5 minutes. Try a shorter date range.")
            st.stop()
        except Exception as e:
            if _is_connection_error(e):
                st.error("Cannot connect to the database server. Check your VPN connection and try refreshing.")
            else:
                st.error(f"Query failed: {e}")
            st.stop()

    if ins_stale or fun_stale or trend_stale:
        st.warning("Database server unreachable (VPN may be disconnected). Showing data from the last successful load.")

    raw, comp, hours, customers, calls, connects, rpc, sales = _load_series(combined)
    T  = _totals(raw)
    CT = _totals(comp)

    st.title("Operations Overview")
    st.caption(
        f"{date_from:%d %b} – {date_to:%d %b %Y}"
        f"  |  vs {comp_from:%d %b} – {comp_to:%d %b %Y}  ({comparison_type})"
    )

    # ── KPI header ─────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    kpi(k1, "Unique Customers", T["UniqueCustomers"], CT["UniqueCustomers"], lambda v: f"{int(v):,}")
    kpi(k2, "Total Calls",      T["Calls"],           CT["Calls"],           lambda v: f"{int(v):,}")
    kpi(k3, "Connects",         T["Connects"],        CT["Connects"],        lambda v: f"{int(v):,}")
    kpi(k4, "Connect Rate",     T["ConnectRate"],     CT["ConnectRate"],     lambda v: f"{v:.1f}%", is_pct=True)
    kpi(k5, "Total RPC",        T["RPC"],             CT["RPC"],             lambda v: f"{int(v):,}")
    kpi(k6, "Total Sales",      T["Sales"],           CT["Sales"],           lambda v: f"{int(v):,}")
    kpi(k7, "RPC Rate",         T["RPCRate"],         CT["RPCRate"],         lambda v: f"{v:.1f}%", is_pct=True)

    # ── Derived KPIs ───────────────────────────────────────────────────────────
    d1, d2, d3, d4 = st.columns(4)
    kpi(d1, "Sales per 1,000 Calls", T["Sales1k"],      CT["Sales1k"],      lambda v: f"{v:.1f}")
    kpi(d2, "Sales per Connect",     T["SalesPerConn"], CT["SalesPerConn"], lambda v: f"{v:.1f}%", is_pct=True)
    kpi(d3, "Sales per Hour",        T["SalesPerHr"],   CT["SalesPerHr"],   lambda v: f"{v:.2f}")
    kpi(d4, "Connects per Hour",     T["ConnPerHr"],    CT["ConnPerHr"],    lambda v: f"{v:.2f}")

    st.divider()

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab_report, tab_charts, tab_funnel = st.tabs(["📋 Report", "📈 Charts", "🔽 Funnel"])

    # ── Tab: Report ─────────────────────────────────────────────────────────────
    with tab_report:
        with st.expander("Metric definitions"):
            st.markdown("""
**Total Calls** — All outbound dial attempts from `vicidial_vicidial_log` including
no-answers, busy, voicemail, and agent-handled calls. Includes ViciDial auto-generated
re-dial lists (e.g. `CAMPAIGN26018 REDIAL`).

**Connects / Connect Rate** — Calls where an agent spoke to the customer
(`isConnect = 1` or `ACSALE` in `trfmDiallerAgentLog`). Connect Rate = Connects ÷ Calls.

**RPC / RPC %** — Right Party Contact. RPC % = RPC ÷ Connects.

**Sales** — `sale = 'Y'` dispositions (includes ACSALE). *Excel reports exclude ACSALE.*

**Hours Worked** — Agent talk time in hours from `trfmDiallerAgentLog`.
            """)

        def _sdiv(num, den):
            return num.where(den > 0, other=0) / den.where(den > 0, other=1)

        op_rows = {
            "Unique Customers":   customers,
            "Hours Worked":       hours,
            "Total Calls":        calls,
            "Calls per Customer": _sdiv(calls, customers),
            "Total Connects":     connects,
            "Connect Rate":       _sdiv(connects, calls) * 100,
            "Total RPC":          rpc,
            "RPC %":              _sdiv(rpc, connects) * 100,
            "Total Sales":        sales,
            "Sales per RPC":      _sdiv(sales, rpc) * 100,
            "Connects per Hour":  _sdiv(connects, hours),
            "RPC per Hour":       _sdiv(rpc, hours),
            "Sales per Hour":     _sdiv(sales, hours),
        }
        pct_m = {"Connect Rate", "RPC %", "Sales per RPC"}
        dp2_m = {"Hours Worked", "Connects per Hour", "RPC per Hour",
                 "Sales per Hour", "Calls per Customer"}

        op_matrix = pd.DataFrame(op_rows, index=CAMPAIGN_ORDER).T

        tot = {}
        for m, s in op_rows.items():
            if m not in pct_m and m not in dp2_m:
                tot[m] = s.sum()
        tot["Calls per Customer"] = safe_div(calls.sum(), customers.sum())
        tot["Connect Rate"]       = safe_div(connects.sum(), calls.sum())    * 100
        tot["RPC %"]              = safe_div(rpc.sum(), connects.sum())      * 100
        tot["Sales per RPC"]      = safe_div(sales.sum(), rpc.sum())         * 100
        tot["Connects per Hour"]  = safe_div(connects.sum(), hours.sum())
        tot["RPC per Hour"]       = safe_div(rpc.sum(), hours.sum())
        tot["Sales per Hour"]     = safe_div(sales.sum(), hours.sum())
        op_matrix["Total / Blended"] = pd.Series(tot)

        def fmt_op(val, metric):
            if pd.isna(val) or val == 0:
                return "-"
            if metric in pct_m:
                return f"{val:.2f}%"
            if metric in dp2_m:
                return f"{val:.2f}"
            return f"{int(val):,}"

        display_op = op_matrix.copy().astype(object)
        for metric in display_op.index:
            for col in display_op.columns:
                display_op.loc[metric, col] = fmt_op(op_matrix.loc[metric, col], metric)

        st.dataframe(display_op, use_container_width=True, height=430)

        with st.expander("Raw numbers"):
            st.dataframe(op_matrix.round(2), use_container_width=True)

        with st.expander("Download CSV"):
            st.download_button(
                "Download CSV",
                data=raw.reset_index().to_csv(index=False),
                file_name=f"insurance_ops_{date_from}_{date_to}.csv",
                mime="text/csv",
            )

    # ── Tab: Charts ─────────────────────────────────────────────────────────────
    with tab_charts:
        if trend_df.empty:
            st.warning("No trend data for the selected period.")
        else:
            trend_df["Date"]        = pd.to_datetime(trend_df["Date"])
            trend_df["Calls"]       = pd.to_numeric(trend_df["Calls"],    errors="coerce").fillna(0)
            trend_df["Connects"]    = pd.to_numeric(trend_df["Connects"], errors="coerce").fillna(0)
            trend_df["Sales"]       = pd.to_numeric(trend_df["Sales"],    errors="coerce").fillna(0)
            trend_df["ConnectRate"] = (trend_df["Connects"] /
                                       trend_df["Calls"].where(trend_df["Calls"] > 0, 1) * 100).round(1)

            curr_trend = trend_df[trend_df["Period"] == "current"]
            comp_trend = trend_df[trend_df["Period"] == "comparison"]
            period_label = f"{date_from:%d %b} – {date_to:%d %b %Y}"
            comp_label   = f"{comp_from:%d %b} – {comp_to:%d %b %Y}"

            def _trend_chart(title, col_name, color_curr, color_cmp):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=curr_trend["Date"], y=curr_trend[col_name],
                    name=period_label, line=dict(color=color_curr, width=2),
                    mode="lines+markers",
                ))
                if not comp_trend.empty:
                    fig.add_trace(go.Scatter(
                        x=comp_trend["Date"], y=comp_trend[col_name],
                        name=comp_label,
                        line=dict(color=color_cmp, width=1.5, dash="dash"),
                        mode="lines+markers",
                    ))
                fig.update_layout(
                    title=title, xaxis_title=None,
                    legend=dict(orientation="h", y=-0.25, font=dict(size=11)),
                    margin=dict(t=40, b=10),
                )
                return fig

            c1, c2, c3 = st.columns(3)
            with c1:
                st.plotly_chart(_trend_chart("Daily Calls", "Calls", "#1f77b4", "#aec7e8"),
                                use_container_width=True)
            with c2:
                st.plotly_chart(_trend_chart("Daily Connect Rate (%)", "ConnectRate",
                                             "#2ca02c", "#98df8a"),
                                use_container_width=True)
            with c3:
                st.plotly_chart(_trend_chart("Daily Sales", "Sales", "#d62728", "#ff9896"),
                                use_container_width=True)

        st.divider()

        # Campaign bar charts (no extra query — reuse loaded data)
        active = raw[raw["Calls"] > 0].copy().reset_index()
        if not active.empty:
            active["ConnectRate"] = (active["Connects"] /
                                     active["Calls"].where(active["Calls"] > 0, 1) * 100).round(1)
            active["SalesPerRPC"] = (active["Sales"] /
                                     active["RPC"].where(active["RPC"] > 0, 1) * 100).round(2)
            active["SalesPerHr"]  = (active["Sales"] /
                                     active["HoursWorked"].where(active["HoursWorked"] > 0, 1)).round(2)
            ca1, ca2 = st.columns(2)
            with ca1:
                fig = px.bar(active.sort_values("Sales", ascending=False),
                             x="Campaign", y="Sales", title="Sales by Campaign",
                             color="Sales", color_continuous_scale="Blues")
                fig.update_layout(xaxis_tickangle=-30, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
                fig = px.bar(active.sort_values("ConnectRate", ascending=False),
                             x="Campaign", y="ConnectRate",
                             title="Connect Rate % by Campaign",
                             labels={"ConnectRate": "Connect %"})
                fig.update_layout(xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)
            with ca2:
                fig = px.bar(active.sort_values("SalesPerRPC", ascending=False),
                             x="Campaign", y="SalesPerRPC",
                             title="Sales per RPC % by Campaign",
                             labels={"SalesPerRPC": "Sales/RPC %"},
                             color="SalesPerRPC", color_continuous_scale="Greens")
                fig.update_layout(xaxis_tickangle=-30, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
                fig = px.bar(active.sort_values("SalesPerHr", ascending=False),
                             x="Campaign", y="SalesPerHr",
                             title="Sales per Hour by Campaign",
                             labels={"SalesPerHr": "Sales/Hr"})
                fig.update_layout(xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)

    # ── Tab: Funnel ─────────────────────────────────────────────────────────────
    with tab_funnel:
        st.subheader("Customer Journey Funnel")

        def _funnel_row(df, period):
            row = df[df["Period"] == period]
            if row.empty:
                return {c: 0 for c in ["UniqueCustomers","UniqueHit","UniqueConnects","UniqueSales"]}
            r = row.iloc[0]
            return {c: int(pd.to_numeric(r.get(c, 0), errors="coerce") or 0)
                    for c in ["UniqueCustomers","UniqueHit","UniqueConnects","UniqueSales"]}

        fn  = _funnel_row(funnel_df, "current")
        cfn = _funnel_row(funnel_df, "comparison")

        def _cvr(a, b): return safe_div(a, b) * 100

        hit_rate      = _cvr(fn["UniqueHit"],      fn["UniqueCustomers"])
        conn_rate_fn  = _cvr(fn["UniqueConnects"],  fn["UniqueHit"])
        sale_rate_fn  = _cvr(fn["UniqueSales"],     fn["UniqueConnects"])
        overall_cvr   = _cvr(fn["UniqueSales"],     fn["UniqueCustomers"])

        c_hit_rate     = _cvr(cfn["UniqueHit"],      cfn["UniqueCustomers"])
        c_conn_rate_fn = _cvr(cfn["UniqueConnects"],  cfn["UniqueHit"])
        c_sale_rate_fn = _cvr(cfn["UniqueSales"],     cfn["UniqueConnects"])
        c_overall_cvr  = _cvr(cfn["UniqueSales"],     cfn["UniqueCustomers"])

        funnel_col, stats_col = st.columns([2, 1])
        with funnel_col:
            fig = go.Figure(go.Funnel(
                y=["Unique Customers", "Unique Hit", "Unique Connects", "Unique Sales"],
                x=[fn["UniqueCustomers"], fn["UniqueHit"],
                   fn["UniqueConnects"],  fn["UniqueSales"]],
                textposition="inside",
                textinfo="value+percent initial",
                marker=dict(color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]),
                connector=dict(line=dict(color="rgba(0,0,0,0.1)", width=1)),
            ))
            fig.update_layout(margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with stats_col:
            st.markdown("#### Step Conversion Rates")
            st.markdown(f"""
| Step | Rate | vs prior |
|------|------|----------|
| Hit / Customers | {hit_rate:.1f}% | {hit_rate - c_hit_rate:+.1f}pp |
| Connect / Hit | {conn_rate_fn:.1f}% | {conn_rate_fn - c_conn_rate_fn:+.1f}pp |
| Sale / Connect | {sale_rate_fn:.1f}% | {sale_rate_fn - c_sale_rate_fn:+.1f}pp |
| **Overall CVR** | **{overall_cvr:.2f}%** | **{overall_cvr - c_overall_cvr:+.2f}pp** |
            """)
            st.markdown("#### Volume")
            st.markdown(f"""
| Stage | Current | Prior | Δ |
|-------|---------|-------|---|
| Customers | {fn['UniqueCustomers']:,} | {cfn['UniqueCustomers']:,} | {fn['UniqueCustomers']-cfn['UniqueCustomers']:+,} |
| Hit | {fn['UniqueHit']:,} | {cfn['UniqueHit']:,} | {fn['UniqueHit']-cfn['UniqueHit']:+,} |
| Connects | {fn['UniqueConnects']:,} | {cfn['UniqueConnects']:,} | {fn['UniqueConnects']-cfn['UniqueConnects']:+,} |
| Sales | {fn['UniqueSales']:,} | {cfn['UniqueSales']:,} | {fn['UniqueSales']-cfn['UniqueSales']:+,} |
            """)

        st.caption(
            "**Unique Hit** — auto-dialler detected a human answer (status not AA/NA/AB/AL/ADC/PDROP). "
            "**Unique Connects** — distinct customers where an agent connected. "
            "**Unique Sales** — distinct customers with a confirmed sale."
        )


# ══════════════════════════════════════════════════════════════════════════════
# CAMPAIGN VIEW
# ══════════════════════════════════════════════════════════════════════════════
elif view == "Campaign":
    with st.spinner("Loading data…"):
        try:
            combined, ins_stale = load_insurance(cf, ct, pf, pt)
        except Exception as e:
            if _is_connection_error(e):
                st.error("Cannot connect to the database server. Check your VPN connection and try refreshing.")
            else:
                st.error(f"Query failed: {e}")
            st.stop()

    if ins_stale:
        st.warning("Database server unreachable (VPN may be disconnected). Showing data from the last successful load.")

    raw, comp, hours, customers, calls, connects, rpc, sales = _load_series(combined)
    T  = _totals(raw)
    CT = _totals(comp)

    st.title("Campaign Performance")
    st.caption(
        f"{date_from:%d %b} – {date_to:%d %b %Y}"
        f"  |  vs {comp_from:%d %b} – {comp_to:%d %b %Y}  ({comparison_type})"
    )

    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Total Sales",      T["Sales"],           CT["Sales"],      lambda v: f"{int(v):,}")
    kpi(k2, "Unique Customers", T["UniqueCustomers"], CT["UniqueCustomers"], lambda v: f"{int(v):,}")
    kpi(k3, "Connect Rate",     T["ConnectRate"],     CT["ConnectRate"], lambda v: f"{v:.1f}%", is_pct=True)
    kpi(k4, "Sales per 1K",     T["Sales1k"],         CT["Sales1k"],    lambda v: f"{v:.1f}")

    st.divider()

    # Unique RPC from the loaded data
    # Note: UniqueRPC is not in MAIN_COLS — it was in the old load_insurance.
    # For the Campaign view we approximate with RPC (total count).
    # True Unique RPC requires load_funnel which is per-total not per-campaign.
    # We show RPC as the campaign-level column and note the distinction.
    camp_rows = {
        "Unique Customers":   customers,
        "Calls":              calls,
        "Calls per Customer": safe_div(calls, customers),
        "Connects":           connects,
        "Connect %":          safe_div(connects, customers) * 100,
        "":                   pd.Series(float("nan"), index=CAMPAIGN_ORDER),
        "RPC":                rpc,
        "RPC %":              safe_div(rpc, connects)       * 100,
        " ":                  pd.Series(float("nan"), index=CAMPAIGN_ORDER),
        "Sales":              sales,
        "Sales per RPC":      safe_div(sales, rpc)          * 100,
        "Sales Conversion":   safe_div(sales, customers)    * 100,
    }
    camp_pct = {"Connect %", "RPC %", "Sales Conversion", "Sales per RPC"}
    camp_dp2 = {"Calls per Customer"}

    camp_matrix = pd.DataFrame(camp_rows, index=CAMPAIGN_ORDER).T
    camp_totals = {
        "Unique Customers":   customers.sum(),
        "Calls":              calls.sum(),
        "Calls per Customer": safe_div(calls.sum(), customers.sum()),
        "Connects":           connects.sum(),
        "Connect %":          safe_div(connects.sum(), customers.sum()) * 100,
        "":                   float("nan"),
        "RPC":                rpc.sum(),
        "RPC %":              safe_div(rpc.sum(), connects.sum())       * 100,
        " ":                  float("nan"),
        "Sales":              sales.sum(),
        "Sales per RPC":      safe_div(sales.sum(), rpc.sum())          * 100,
        "Sales Conversion":   safe_div(sales.sum(), customers.sum())    * 100,
    }
    camp_matrix["Total / Blended"] = pd.Series(camp_totals)

    def fmt_camp(val, metric):
        if pd.isna(val) or metric.strip() == "":
            return ""
        if metric in camp_pct:
            return f"{val:.2f}%"
        if metric in camp_dp2:
            return f"{val:.2f}"
        return f"{int(val):,}"

    display_camp = camp_matrix.copy().astype(object)
    for metric in display_camp.index:
        for col in display_camp.columns:
            display_camp.loc[metric, col] = fmt_camp(camp_matrix.loc[metric, col], metric)

    st.dataframe(display_camp, use_container_width=True, height=460)

    with st.expander("Metric definitions"):
        st.markdown("""
**Sales Conversion** — Total Sales ÷ Unique Customers. Overall sales yield on the dialled population.

**Sales per RPC** — Conversion rate of right-party contacts that resulted in a sale.
        """)

    with st.expander("Download CSV"):
        st.download_button(
            "Download CSV",
            data=raw.reset_index().to_csv(index=False),
            file_name=f"insurance_campaign_{date_from}_{date_to}.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# AGENT VIEW
# ══════════════════════════════════════════════════════════════════════════════
elif view == "Agent":
    with st.spinner("Loading data…"):
        try:
            agent_df, ag_stale = load_agent(cf, ct, pf, pt)
        except Exception as e:
            if _is_connection_error(e):
                st.error("Cannot connect to the database server. Check your VPN connection and try refreshing.")
            else:
                st.error(f"Query failed: {e}")
            st.stop()

    if ag_stale:
        st.warning("Database server unreachable. Showing data from the last successful load.")

    st.title("Agent Analytics")
    st.caption(
        f"{date_from:%d %b} – {date_to:%d %b %Y}"
        f"  |  vs {comp_from:%d %b} – {comp_to:%d %b %Y}  ({comparison_type})"
    )

    # ── Split periods and derive metrics ────────────────────────────────────
    for col in ["Calls", "TalkHrs", "AvailHrs", "LoggedHrs", "UniqueCustomers", "Connects", "RPC", "Sales"]:
        agent_df[col] = pd.to_numeric(agent_df[col], errors="coerce").fillna(0)

    curr_ag = agent_df[agent_df["Period"] == "current"].copy()
    comp_ag = agent_df[agent_df["Period"] == "comparison"].copy()

    # Filter to agents with meaningful activity (exclude system/test accounts)
    curr_ag = curr_ag[curr_ag["Calls"] >= 5].copy()
    comp_ag = comp_ag[comp_ag["Calls"] >= 5].copy()

    if curr_ag.empty:
        st.info("No agent data for the selected period.")
        st.stop()

    # curr_ag_all / comp_ag_all used for Team Summary tab (always unfiltered)
    curr_ag_all = curr_ag.copy()
    comp_ag_all = comp_ag.copy()

    def _pdiv(num, den):
        return num.where(den > 0, other=0) / den.where(den > 0, other=1)

    curr_ag["ConnectPct"]  = (_pdiv(curr_ag["Connects"],    curr_ag["Calls"])        * 100)
    curr_ag["RPCPct"]      = (_pdiv(curr_ag["RPC"],          curr_ag["Connects"])     * 100)
    curr_ag["SalesPerRPC"] = (_pdiv(curr_ag["Sales"],        curr_ag["RPC"])          * 100)
    curr_ag["SalesHr"]      = (_pdiv(curr_ag["Sales"],   curr_ag["AvailHrs"]))
    curr_ag["CallsHr"]      = (_pdiv(curr_ag["Calls"],   curr_ag["AvailHrs"]))
    curr_ag["Adherence"]    = (_pdiv(curr_ag["AvailHrs"], curr_ag["LoggedHrs"]) * 100)

    # ── Team totals ──────────────────────────────────────────────────────────
    def _team_totals(df):
        T = {c: float(df[c].sum()) for c in ["Calls","TalkHrs","AvailHrs","LoggedHrs","UniqueCustomers","Connects","RPC","Sales"]}
        T["ConnectPct"]  = safe_div(T["Connects"], T["Calls"])        * 100
        T["RPCPct"]      = safe_div(T["RPC"],       T["Connects"])     * 100
        T["SalesPerRPC"] = safe_div(T["Sales"],     T["RPC"])          * 100
        T["SalesHr"]     = safe_div(T["Sales"],     T["AvailHrs"])
        T["Adherence"]   = safe_div(T["AvailHrs"],  T["LoggedHrs"]) * 100
        T["Agents"]      = len(df)
        return T

    T  = _team_totals(curr_ag)
    CT = _team_totals(comp_ag) if not comp_ag.empty else {k: 0 for k in ["Sales","SalesHr","ConnectPct","RPCPct","SalesPerRPC","Agents","Calls","Connects","RPC","TalkHrs","AvailHrs","LoggedHrs","Adherence","UniqueCustomers"]}

    # ── KPI strip ────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    kpi(k1, "Team Sales",      T["Sales"],      CT["Sales"],      lambda v: f"{int(v):,}")
    kpi(k2, "Sales per Hour",  T["SalesHr"],    CT["SalesHr"],    lambda v: f"{v:.2f}")
    kpi(k3, "Avail Hrs",       T["AvailHrs"],   CT["AvailHrs"],   lambda v: f"{v:.1f}")
    kpi(k4, "Adherence %",     T["Adherence"],  CT["Adherence"],  lambda v: f"{v:.1f}%", is_pct=True)
    kpi(k5, "Connect %",       T["ConnectPct"], CT["ConnectPct"], lambda v: f"{v:.1f}%", is_pct=True)
    kpi(k6, "RPC %",           T["RPCPct"],     CT["RPCPct"],     lambda v: f"{v:.1f}%", is_pct=True)
    kpi(k7, "Active Agents",   T["Agents"],     CT["Agents"],     lambda v: f"{int(v)}")

    st.divider()

    # ── Performance flags ────────────────────────────────────────────────────
    def _flag(row, avg_connect, avg_rpc_conv, avg_sales_hr):
        if row["AvailHrs"] < 1:
            return "🔴", "Low hours on dialler"
        if row["Sales"] == 0 and row["RPC"] >= 10:
            return "🔴", "0 sales despite RPCs — skill gap"
        if avg_connect > 0 and row["ConnectPct"] < avg_connect * 0.5:
            return "🟡", "Low connect rate — list or data issue"
        if avg_rpc_conv > 0 and row["SalesPerRPC"] < avg_rpc_conv * 0.5 and row["RPC"] >= 5:
            return "🟡", "Low RPC conversion — coaching needed"
        if avg_sales_hr > 0 and row["SalesHr"] < avg_sales_hr * 0.5 and row["Calls"] >= 20:
            return "🟡", "Below team average productivity"
        return "✅", ""

    curr_ag[["Flag","FlagReason"]] = curr_ag.apply(
        lambda r: pd.Series(_flag(r, T["ConnectPct"], T["SalesPerRPC"], T["SalesHr"])), axis=1
    )

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_board, tab_diag, tab_team = st.tabs(["🏆 Leaderboard", "🔍 Diagnostics", "👥 Team Summary"])

    # ── Leaderboard ──────────────────────────────────────────────────────────
    with tab_board:
        all_teams = sorted(curr_ag["Team"].dropna().unique().tolist())
        team_filter = st.selectbox(
            "Team", options=["All"] + all_teams, index=0, label_visibility="collapsed",
            key="lb_team_filter",
        )
        filtered_ag = curr_ag if team_filter == "All" else curr_ag[curr_ag["Team"] == team_filter]

        ranked = filtered_ag.sort_values("SalesHr", ascending=False).reset_index(drop=True)
        ranked.index = ranked.index + 1  # 1-based rank

        display_cols = {
            "Agent":        ranked["AgentName"],
            "Logged Hrs":   ranked["LoggedHrs"].round(1),
            "Avail Hrs":    ranked["AvailHrs"].round(1),
            "Adherence %":  ranked["Adherence"].round(1).astype(str) + "%",
            "Talk Hrs":     ranked["TalkHrs"].round(1),
            "Calls":       ranked["Calls"].astype(int),
            "Connects":    ranked["Connects"].astype(int),
            "Connect %":   ranked["ConnectPct"].round(1).astype(str) + "%",
            "RPC":         ranked["RPC"].astype(int),
            "RPC %":       ranked["RPCPct"].round(1).astype(str) + "%",
            "Sales":       ranked["Sales"].astype(int),
            "Sales/RPC":   ranked["SalesPerRPC"].round(1).astype(str) + "%",
            "Sales/Avail Hr": ranked["SalesHr"].round(2),
            "Status":      ranked["Flag"],
        }
        board_df = pd.DataFrame(display_cols)
        board_df.index.name = "Rank"
        st.dataframe(board_df, use_container_width=True, height=500)

        with st.expander("Download CSV"):
            st.download_button(
                "Download agent data CSV",
                data=curr_ag.drop(columns=["Flag","FlagReason"]).to_csv(index=False),
                file_name=f"agent_analytics_{date_from}_{date_to}.csv",
                mime="text/csv",
            )

    # ── Diagnostics ──────────────────────────────────────────────────────────
    with tab_diag:
        red   = curr_ag[curr_ag["Flag"] == "🔴"].sort_values("SalesHr")
        amber = curr_ag[curr_ag["Flag"] == "🟡"].sort_values("SalesHr")

        if not red.empty:
            st.markdown("#### 🔴 Immediate Attention")
            for _, r in red.iterrows():
                st.error(
                    f"**{r['AgentName']}** — {r['FlagReason']}  |  "
                    f"Sales: {int(r['Sales'])}  |  RPCs: {int(r['RPC'])}  |  "
                    f"Avail Hrs: {r['AvailHrs']:.1f}h  |  Sales/Avail Hr: {r['SalesHr']:.2f}"
                )

        if not amber.empty:
            st.markdown("#### 🟡 Coaching Recommended")
            for _, r in amber.iterrows():
                st.warning(
                    f"**{r['AgentName']}** — {r['FlagReason']}  |  "
                    f"Connect: {r['ConnectPct']:.1f}%  |  "
                    f"RPC conv: {r['SalesPerRPC']:.1f}%  |  "
                    f"Sales/Hr: {r['SalesHr']:.2f}"
                )

        if red.empty and amber.empty:
            st.success("No performance flags for the selected period.")

        st.divider()
        st.markdown("#### Performance Distribution")

        p75 = curr_ag["SalesHr"].quantile(0.75)
        p25 = curr_ag["SalesHr"].quantile(0.25)
        top = curr_ag[curr_ag["SalesHr"] >= p75]
        mid = curr_ag[(curr_ag["SalesHr"] >= p25) & (curr_ag["SalesHr"] < p75)]
        bot = curr_ag[curr_ag["SalesHr"] < p25]

        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric("🏆 Top performers", len(top))
            if not top.empty:
                st.caption(f"Avg Sales/Hr: {top['SalesHr'].mean():.2f}  |  Avg Sales: {top['Sales'].mean():.0f}")
        with d2:
            st.metric("📊 Consistent", len(mid))
            if not mid.empty:
                st.caption(f"Avg Sales/Hr: {mid['SalesHr'].mean():.2f}  |  Avg Sales: {mid['Sales'].mean():.0f}")
        with d3:
            st.metric("⚠️ At-risk", len(bot))
            if not bot.empty:
                st.caption(f"Avg Sales/Hr: {bot['SalesHr'].mean():.2f}  |  Avg Sales: {bot['Sales'].mean():.0f}")

        st.divider()
        chart_df = curr_ag[["AgentName", "SalesHr"]].sort_values("SalesHr", ascending=False)
        avg_line = T["SalesHr"]
        fig = px.bar(
            chart_df, x="AgentName", y="SalesHr",
            title="Sales per Available Hour by Agent",
            labels={"SalesHr": "Sales / Avail Hr", "AgentName": ""},
            color="SalesHr", color_continuous_scale="RdYlGn",
        )
        fig.add_hline(
            y=avg_line, line_dash="dash", line_color="grey",
            annotation_text=f"Team avg {avg_line:.2f}", annotation_position="top right"
        )
        fig.update_layout(xaxis_tickangle=-45, coloraxis_showscale=False, margin=dict(t=50, b=80))
        st.plotly_chart(fig, use_container_width=True)

    # ── Team Summary ─────────────────────────────────────────────────────────
    with tab_team:
        st.subheader("Team Summary")

        def _team_rollup(df):
            grp = df.groupby("Team", sort=False)
            t = pd.DataFrame({
                "Agents":      grp["Agent"].nunique(),
                "Logged Hrs":  grp["LoggedHrs"].sum(),
                "Avail Hrs":   grp["AvailHrs"].sum(),
                "Talk Hrs":    grp["TalkHrs"].sum(),
                "Calls":       grp["Calls"].sum(),
                "Connects":    grp["Connects"].sum(),
                "RPC":         grp["RPC"].sum(),
                "Sales":       grp["Sales"].sum(),
            }).reset_index()
            t["Adherence %"]  = (t["Avail Hrs"]  / t["Logged Hrs"].where(t["Logged Hrs"] > 0, 1) * 100).round(1)
            t["Connect %"]    = (t["Connects"]    / t["Calls"].where(t["Calls"] > 0, 1)          * 100).round(1)
            t["RPC %"]        = (t["RPC"]         / t["Connects"].where(t["Connects"] > 0, 1)    * 100).round(1)
            t["Sales/RPC %"]  = (t["Sales"]       / t["RPC"].where(t["RPC"] > 0, 1)              * 100).round(1)
            t["Sales/Avail Hr"] = (t["Sales"]     / t["Avail Hrs"].where(t["Avail Hrs"] > 0, 1)).round(2)
            return t.sort_values("Sales", ascending=False).reset_index(drop=True)

        team_curr = _team_rollup(curr_ag_all)
        team_comp = _team_rollup(comp_ag_all) if not comp_ag_all.empty else None

        # Format for display
        disp = team_curr.copy().astype(object)
        int_cols = {"Agents", "Calls", "Connects", "RPC", "Sales"}
        pct_cols = {"Adherence %", "Connect %", "RPC %", "Sales/RPC %"}
        hr_cols  = {"Logged Hrs", "Avail Hrs", "Talk Hrs"}
        for idx in team_curr.index:
            for col in team_curr.columns:
                v = team_curr.at[idx, col]
                if col == "Team":
                    pass
                elif col in int_cols:
                    disp.at[idx, col] = f"{int(v):,}"
                elif col in pct_cols:
                    disp.at[idx, col] = f"{v:.1f}%"
                elif col in hr_cols:
                    disp.at[idx, col] = f"{v:.1f}"
                elif col == "Sales/Avail Hr":
                    disp.at[idx, col] = f"{v:.2f}"

        st.dataframe(disp.set_index("Team"), use_container_width=True)

        st.divider()
        st.markdown("#### Sales by Team")
        fig = px.bar(
            team_curr.head(20), x="Team", y="Sales",
            color="Sales", color_continuous_scale="Blues",
        )
        fig.update_layout(xaxis_tickangle=-35, coloraxis_showscale=False, margin=dict(t=20, b=80))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                team_curr.head(20).sort_values("Sales/Avail Hr", ascending=False),
                x="Team", y="Sales/Avail Hr", title="Sales per Avail Hour by Team",
            )
            fig.update_layout(xaxis_tickangle=-35, margin=dict(t=40, b=80))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(
                team_curr.head(20).sort_values("Adherence %", ascending=False),
                x="Team", y="Adherence %", title="Dialler Adherence % by Team",
            )
            fig.update_layout(xaxis_tickangle=-35, margin=dict(t=40, b=80))
            st.plotly_chart(fig, use_container_width=True)
