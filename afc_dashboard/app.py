import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from scipy import stats

st.set_page_config(page_title="AFC Fuhrpark-Dashboard", page_icon="🚗",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #f8fafc; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }
section[data-testid="stSidebar"] { background: #1a2332; }
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
section[data-testid="stSidebar"] .stSelectbox label {
    color: #64748b !important; font-size: 0.7rem;
    letter-spacing: 0.08em; text-transform: uppercase; }
.kpi-card { background: white; border-radius: 10px; padding: 1.1rem 1.3rem;
    border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
.kpi-label { font-size: 0.68rem; font-weight: 600; letter-spacing: 0.09em;
    text-transform: uppercase; color: #94a3b8; margin-bottom: 0.35rem; }
.kpi-value { font-size: 1.75rem; font-weight: 700; color: #1a2332; line-height: 1.1; }
.kpi-sub   { font-size: 0.75rem; color: #94a3b8; margin-top: 0.25rem; }
.kpi-red   { color: #dc2626; }
.section-label { font-size: 0.68rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #94a3b8;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 0.4rem; margin: 1.4rem 0 1rem 0; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    import os as _os
    _here = _os.path.dirname(_os.path.abspath(__file__))
    df = pd.read_csv(_os.path.join(_here, "claims_data.csv"), parse_dates=["claim_date"])
    gmc = df.groupby(["damage_category","vehicle_type"])["estimated_cost_eur"].transform("median")
    cmc = df.groupby("damage_category")["estimated_cost_eur"].transform("median")
    df["estimated_cost_eur"] = df["estimated_cost_eur"].fillna(gmc).fillna(cmc)
    gmd = df.groupby(["damage_category","status"])["repair_duration_days"].transform("median")
    cmd = df.groupby("damage_category")["repair_duration_days"].transform("median")
    df["repair_duration_days"] = df["repair_duration_days"].fillna(gmd).fillna(cmd)
    df["preventable"] = df["fault_type"].isin(["Eigenverschulden","Teilschuld"])
    df["month"] = df["claim_date"].dt.to_period("M").dt.to_timestamp()
    df["year"]  = df["claim_date"].dt.year
    active = df[df["status"] != "storniert"].copy()
    return df, active

@st.cache_data
def build_forecast(active):
    all_months    = pd.date_range(
        active["claim_date"].min().to_period("M").to_timestamp(),
        active["claim_date"].max().to_period("M").to_timestamp(), freq="MS")
    all_customers = active["customer_id"].unique()
    grid = pd.MultiIndex.from_product(
        [all_customers, all_months], names=["customer_id","month"]
    ).to_frame(index=False)
    grid["month"] = pd.to_datetime(grid["month"])
    counts = (active
        .assign(month=active["claim_date"].dt.to_period("M").dt.to_timestamp())
        .groupby(["customer_id","month"])["claim_id"].count()
        .reset_index(name="claim_count"))
    panel = grid.merge(counts, on=["customer_id","month"], how="left").fillna({"claim_count":0})
    panel["claim_count"]   = panel["claim_count"].astype(int)
    panel["month_of_year"] = panel["month"].dt.month
    panel["year"]          = panel["month"].dt.year
    panel = panel.sort_values(["customer_id","month"]).reset_index(drop=True)
    cust_mean = panel.groupby("customer_id")["claim_count"].transform("mean")
    panel["rolling_12m_rate"] = (panel.groupby("customer_id")["claim_count"]
        .transform(lambda x: x.shift(1).rolling(12, min_periods=3).mean()))
    panel["rolling_12m_rate"] = panel["rolling_12m_rate"].fillna(cust_mean)
    le = LabelEncoder()
    panel["customer_enc"] = le.fit_transform(panel["customer_id"])
    FEATURES = ["customer_enc","month_of_year","year","rolling_12m_rate"]
    train = panel[panel["month"] <= pd.Timestamp("2025-09-30")]
    def make_X(d):
        X = d[FEATURES].copy().astype(float)
        X.insert(0, "const", 1.0)
        return X
    result = sm.GLM(train["claim_count"], make_X(train),
                    family=sm.families.Poisson()).fit()
    fc_months = pd.date_range("2026-04-01","2026-12-01", freq="MS")
    fc_grid   = pd.DataFrame(
        [{"customer_id": c, "month": m} for c in all_customers for m in fc_months])
    fc_grid["month_of_year"] = fc_grid["month"].dt.month
    fc_grid["year"]          = fc_grid["month"].dt.year
    fc_grid["customer_enc"]  = le.transform(fc_grid["customer_id"])
    last_rate = panel.groupby("customer_id")["rolling_12m_rate"].last()
    fc_grid["rolling_12m_rate"] = fc_grid["customer_id"].map(last_rate)
    mu = result.predict(make_X(fc_grid)).values
    fc_grid["predicted_claims"] = np.round(mu).astype(int)
    fc_grid["ci_low"]  = stats.poisson.ppf(0.10, np.maximum(mu, 0.01)).clip(min=0).astype(int)
    fc_grid["ci_high"] = stats.poisson.ppf(0.90, np.maximum(mu, 0.01)).astype(int)
    global_avg  = active["actual_cost_eur"].mean()
    cust_stats  = active.groupby("customer_id").agg(n=("actual_cost_eur","count"), avg=("actual_cost_eur","mean"))
    k = 5
    cust_stats["blended"] = (cust_stats["n"]*cust_stats["avg"] + k*global_avg) / (cust_stats["n"] + k)
    fc_grid["blended_cost"]       = fc_grid["customer_id"].map(cust_stats["blended"]).fillna(global_avg)
    fc_grid["predicted_cost_eur"] = fc_grid["predicted_claims"] * fc_grid["blended_cost"]
    return panel, fc_grid

df, active = load_data()
panel, fc_grid = build_forecast(active)

with st.sidebar:
    st.markdown("### 🚗 AFC Fuhrpark-Dashboard")
    st.markdown("---")
    customers = sorted(active["customer_id"].unique())
    selected  = st.selectbox("Kunde auswählen", customers, index=customers.index("CUST-002"))
    st.markdown("---")
    st.markdown("""<div style='font-size:0.75rem; color:#64748b; line-height:1.6;'>
    Schadensanalyse & Vorhersage<br>Datenbasis: Jan 2023 – Feb 2026<br><br>
    <b style='color:#94a3b8;'>Auto Fleet Control</b></div>""", unsafe_allow_html=True)

cust_df  = active[active["customer_id"] == selected].copy()
port_df  = active[active["customer_id"] != selected].copy()
resolved = cust_df[cust_df["status"] == "abgeschlossen"]
open_cls = cust_df[cust_df["status"] != "abgeschlossen"]
cust_fc  = fc_grid[fc_grid["customer_id"] == selected].sort_values("month")
cust_pan = panel[panel["customer_id"] == selected].copy()

BLUE = "#2563eb"; RED = "#dc2626"; GREY = "#cbd5e1"; ORANGE = "#f97316"
CL = dict(plot_bgcolor="white", paper_bgcolor="white",
          margin=dict(l=0, r=10, t=40, b=0),
          font=dict(family="Inter", size=11, color="#1e293b"),
          title_font=dict(size=13, color="#1a2332"),
          hovermode="x unified",
          legend=dict(orientation="h", y=-0.22, font=dict(size=10)))

total_cost   = cust_df["actual_cost_eur"].sum()
total_claims = len(cust_df)
prev_rate    = cust_df["preventable"].mean() * 100
prev_cost    = cust_df[cust_df["preventable"]]["actual_cost_eur"].sum()
open_count   = len(open_cls)
avg_dur      = resolved["repair_duration_days"].mean() if len(resolved) > 0 else 0
fc_total     = cust_fc["predicted_claims"].sum()
fc_cost      = cust_fc["predicted_cost_eur"].sum()
fc_lo        = cust_fc["ci_low"].sum()
fc_hi        = cust_fc["ci_high"].sum()

st.markdown(f"## {selected} · Schadensanalyse")
st.markdown(
    f"<span style='color:#64748b; font-size:0.9rem;'>Datenbasis: Jan 2023 – Feb 2026 &nbsp;·&nbsp; "
    f"{total_claims} Schäden gesamt &nbsp;·&nbsp; "
    f"Gesamtkosten: <b style='color:#1a2332;'>€{total_cost:,.0f}</b></span>",
    unsafe_allow_html=True)

st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Gesamtkosten</div>
        <div class="kpi-value">€{total_cost/1000:.0f}k</div>
        <div class="kpi-sub">Gesamter Zeitraum</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Schadensfälle</div>
        <div class="kpi-value">{total_claims}</div>
        <div class="kpi-sub">Aktive Schäden</div></div>""", unsafe_allow_html=True)
with k3:
    c = "kpi-red" if prev_rate >= 40 else ""
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Vermeidbare Schäden</div>
        <div class="kpi-value {c}">{prev_rate:.0f}%</div>
        <div class="kpi-sub">€{prev_cost/1000:.0f}k vermeidbare Kosten</div></div>""", unsafe_allow_html=True)
with k4:
    c = "kpi-red" if open_count > 10 else ""
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Offene Schäden</div>
        <div class="kpi-value {c}">{open_count}</div>
        <div class="kpi-sub">Ø {avg_dur:.1f} Tage Reparatur</div></div>""", unsafe_allow_html=True)
with k5:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Prognose Apr–Dez 2026</div>
        <div class="kpi-value">{fc_total:.0f}</div>
        <div class="kpi-sub">Schäden · KI: {fc_lo:.0f}–{fc_hi:.0f} · €{fc_cost/1000:.0f}k</div></div>""",
        unsafe_allow_html=True)

# ── Q1 ────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Q1 · Wie entwickeln sich meine Gesamtkosten?</div>',
            unsafe_allow_html=True)
monthly = cust_df.groupby("month")["actual_cost_eur"].sum().reset_index()
monthly["rolling"] = monthly["actual_cost_eur"].rolling(3, min_periods=1).mean()
fig = go.Figure()
fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["actual_cost_eur"],
    fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
    line=dict(color=BLUE, width=1.5), name="Monatliche Kosten",
    hovertemplate="€%{y:,.0f}<extra>Monatlich</extra>"))
fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["rolling"],
    line=dict(color=BLUE, width=2.5), name="3-Monats-Durchschnitt",
    hovertemplate="€%{y:,.0f}<extra>3M-Schnitt</extra>"))
fig.update_layout(**CL, title="Monatliche Schadenskosten", height=300,
    yaxis=dict(tickprefix="€", tickformat=",", gridcolor="#f1f5f9"),
    xaxis=dict(gridcolor="#f1f5f9"))
st.plotly_chart(fig, use_container_width=True)

# ── Q2 ────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Q2 · Welche Schadensarten treiben meine Kosten?</div>',
            unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    cat = cust_df.groupby("damage_category")["actual_cost_eur"].sum().reset_index()
    cat = cat.sort_values("actual_cost_eur", ascending=False)
    fig = go.Figure(go.Pie(labels=cat["damage_category"], values=cat["actual_cost_eur"],
        hole=0.52, marker=dict(colors=[BLUE,"#3b82f6","#60a5fa","#93c5fd","#bfdbfe",GREY]),
        textinfo="label+percent", textfont=dict(size=11),
        hovertemplate="%{label}<br>€%{value:,.0f}<extra></extra>"))
    fig.update_layout(**{k:v for k,v in CL.items() if k not in ["xaxis","yaxis","hovermode"]},
        title="Kostenanteil nach Schadensart", height=300, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
with c2:
    pa = port_df.groupby("damage_category")["actual_cost_eur"].mean()
    ca = cust_df.groupby("damage_category")["actual_cost_eur"].mean()
    comp = pd.DataFrame({"Ihr Ø":ca, "Portfolio Ø":pa}).dropna().sort_values("Ihr Ø")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=comp.index, x=comp["Portfolio Ø"], name="Portfolio Ø",
        orientation="h", marker_color=GREY,
        hovertemplate="%{x:,.0f} €<extra>Portfolio Ø</extra>"))
    fig.add_trace(go.Bar(y=comp.index, x=comp["Ihr Ø"], name="Ihr Ø",
        orientation="h", marker_color=BLUE,
        hovertemplate="%{x:,.0f} €<extra>Ihr Ø</extra>"))
    fig.update_layout(**CL, title="Ø Kosten je Schaden vs. Portfolio",
        barmode="overlay", height=300,
        xaxis=dict(title="€ je Schaden", gridcolor="#f1f5f9"),
        yaxis=dict(gridcolor="white"))
    st.plotly_chart(fig, use_container_width=True)

# ── Q3 ────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Q3 · Wie lange stehen meine Fahrzeuge still?</div>',
            unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    dc = resolved.groupby("damage_category")["repair_duration_days"].mean()
    dp = port_df[port_df["status"]=="abgeschlossen"].groupby("damage_category")["repair_duration_days"].mean()
    comp3 = pd.DataFrame({"Ihr Ø":dc, "Portfolio Ø":dp}).dropna().sort_values("Ihr Ø")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=comp3.index, x=comp3["Portfolio Ø"], name="Portfolio Ø",
        orientation="h", marker_color=GREY,
        hovertemplate="%{x:.1f} Tage<extra>Portfolio Ø</extra>"))
    fig.add_trace(go.Bar(y=comp3.index, x=comp3["Ihr Ø"], name="Ihr Ø",
        orientation="h", marker_color=BLUE,
        hovertemplate="%{x:.1f} Tage<extra>Ihr Ø</extra>"))
    fig.update_layout(**CL, title="Ø Reparaturdauer nach Schadensart",
        barmode="overlay", height=300,
        xaxis=dict(title="Tage", gridcolor="#f1f5f9"),
        yaxis=dict(gridcolor="white"))
    st.plotly_chart(fig, use_container_width=True)
with c2:
    slbl = {"in_bearbeitung":"In Bearbeitung","wartend_auf_teile":"Wartet auf Teile"}
    if len(open_cls) > 0:
        oc = open_cls["status"].map(slbl).value_counts().reset_index()
        oc.columns = ["status","count"]
        fig = go.Figure(go.Bar(x=oc["status"], y=oc["count"],
            marker_color=[ORANGE,RED][:len(oc)],
            text=oc["count"], textposition="outside",
            hovertemplate="%{x}: %{y}<extra></extra>"))
        fig.update_layout(**CL, title=f"Offene Schäden heute · {open_count} gesamt",
            height=300, showlegend=False,
            yaxis=dict(gridcolor="#f1f5f9"), xaxis=dict(gridcolor="white"))
    else:
        fig = go.Figure()
        fig.add_annotation(text="Keine offenen Schäden ✓", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="#16a34a"))
        fig.update_layout(height=300, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

# ── Q4 ────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Q4 · Welche Schäden sind vermeidbar?</div>',
            unsafe_allow_html=True)
yearly = (cust_df.groupby(["year","preventable"])["actual_cost_eur"]
          .sum().unstack(fill_value=0))
yearly.columns = ["Nicht vermeidbar","Vermeidbar (Eigen-/Teilschuld)"]
yearly = yearly[yearly.index < 2026].reset_index()
fig = go.Figure()
fig.add_trace(go.Bar(x=yearly["year"].astype(str), y=yearly["Nicht vermeidbar"],
    name="Nicht vermeidbar", marker_color=GREY,
    hovertemplate="€%{y:,.0f}<extra>Nicht vermeidbar</extra>"))
fig.add_trace(go.Bar(x=yearly["year"].astype(str), y=yearly["Vermeidbar (Eigen-/Teilschuld)"],
    name="Vermeidbar (Eigen-/Teilschuld)", marker_color=RED,
    hovertemplate="€%{y:,.0f}<extra>Vermeidbar</extra>"))
fig.update_layout(**CL,
    title=f"Vermeidbare vs. nicht vermeidbare Kosten · €{prev_cost/1000:.0f}k reduzierbar",
    barmode="stack", height=300,
    yaxis=dict(tickprefix="€", tickformat=",", gridcolor="#f1f5f9"),
    xaxis=dict(gridcolor="white"))
st.plotly_chart(fig, use_container_width=True)

# ── Prognose ──────────────────────────────────────────────────────
st.markdown('<div class="section-label">Prognose · Erwartete Schäden Apr–Dez 2026</div>',
            unsafe_allow_html=True)
hist_all    = cust_pan[cust_pan["month"] <= pd.Timestamp("2026-02-28")].tail(20)
hist_train  = hist_all[hist_all["month"] <= pd.Timestamp("2025-09-30")]
hist_recent = hist_all[hist_all["month"] >  pd.Timestamp("2025-09-30")]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=pd.concat([cust_fc["month"], cust_fc["month"].iloc[::-1]]),
    y=pd.concat([cust_fc["ci_high"], cust_fc["ci_low"].iloc[::-1]]),
    fill="toself", fillcolor="rgba(37,99,235,0.10)",
    line=dict(color="rgba(0,0,0,0)"), name="80% Konfidenzband", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=hist_train["month"], y=hist_train["claim_count"],
    line=dict(color=BLUE, width=2), mode="lines+markers", marker=dict(size=4),
    name="Tatsächliche Schäden",
    hovertemplate="%{y} Schäden<extra>Actual</extra>"))
if len(hist_recent) > 0:
    join = pd.concat([hist_train.tail(1), hist_recent])
    fig.add_trace(go.Scatter(x=join["month"], y=join["claim_count"],
        line=dict(color=BLUE, width=2), mode="lines+markers", marker=dict(size=4),
        opacity=0.45, showlegend=False,
        hovertemplate="%{y} Schäden<extra>Testperiode</extra>"))
    fig.add_trace(go.Scatter(
        x=[hist_recent["month"].iloc[-1], cust_fc["month"].iloc[0]],
        y=[hist_recent["claim_count"].iloc[-1], cust_fc["predicted_claims"].iloc[0]],
        line=dict(color=BLUE, width=2, dash="dot"),
        opacity=0.35, showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=cust_fc["month"], y=cust_fc["predicted_claims"],
    line=dict(color=BLUE, width=2, dash="dash"), mode="lines+markers", marker=dict(size=4),
    name=f"Prognose · {int(fc_total)} Schäden erwartet",
    hovertemplate="%{y:.0f} Schäden<extra>Prognose</extra>"))
fig.add_vline(x="2025-10-01", line_color="#e2e8f0", line_width=1.5, line_dash="dot")
fig.add_vline(x="2026-04-01", line_color="#94a3b8", line_width=1.5, line_dash="dot")
fig.add_annotation(x="2025-10-15", y=0, yref="paper", yanchor="bottom",
    text="Testperiode", showarrow=False, font=dict(size=10, color="#94a3b8"))
fig.add_annotation(x="2026-04-15", y=0, yref="paper", yanchor="bottom",
    text="Prognose →", showarrow=False, font=dict(size=10, color="#64748b"))
fig.update_layout(**CL,
    title=f"Schadenshäufigkeit & Prognose · 80% KI: {int(fc_lo)}–{int(fc_hi)} Schäden (Apr–Dez 2026)",
    height=340,
    yaxis=dict(title="Schäden / Monat", gridcolor="#f1f5f9"),
    xaxis=dict(gridcolor="#f1f5f9"))
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("""<div style='text-align:center; color:#94a3b8; font-size:0.72rem;'>
Auto Fleet Control · Data & AI Team · Fallstudie-Dashboard · 2026</div>""",
unsafe_allow_html=True)
