import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = 'AFC Case Study Data Scientist_claims_data.csv'
TODAY     = pd.Timestamp('2026-03-09')
TRAIN_END = pd.Timestamp('2025-09-30')
FEATURES  = ['customer_enc', 'month_of_year', 'year', 'rolling_12m_rate']
BLUE      = '#2980b9'


# ── Pipeline (runs once, cached) ──────────────────────────────────────────────

@st.cache_data
def run_pipeline():
    df = pd.read_csv(DATA_PATH, parse_dates=['claim_date'])

    # Impute estimated_cost_eur by damage_category × vehicle_type group median
    gmc = df.groupby(['damage_category', 'vehicle_type'])['estimated_cost_eur'].transform('median')
    df['estimated_cost_eur'] = df['estimated_cost_eur'].fillna(gmc).fillna(
        df.groupby('damage_category')['estimated_cost_eur'].transform('median')
    )
    # Impute repair_duration_days by damage_category × status group median
    gmd = df.groupby(['damage_category', 'status'])['repair_duration_days'].transform('median')
    df['repair_duration_days'] = df['repair_duration_days'].fillna(gmd).fillna(
        df.groupby('damage_category')['repair_duration_days'].transform('median')
    )

    df['year']           = df['claim_date'].dt.year
    df['month']          = df['claim_date'].dt.month
    df['cost_delta']     = df['actual_cost_eur'] - df['estimated_cost_eur']
    df['cost_delta_pct'] = df['cost_delta'] / df['estimated_cost_eur']

    active   = df[df['status'] != 'storniert'].copy()
    resolved = active[active['status'] == 'abgeschlossen'].copy()

    # ── KPIs per customer ─────────────────────────────────────────────────────
    cust_span = df.groupby('customer_id')['claim_date'].agg(['min', 'max'])
    cust_span['months_active'] = (
        (cust_span['max'] - cust_span['min']).dt.days / 30.44
    ).clip(lower=1)

    cust_vol = df.groupby('customer_id').agg(
        total_claims=('claim_id', 'count'),
        cancelled   =('status', lambda x: (x == 'storniert').sum()),
    )
    cust_vol['cancellation_rate']    = cust_vol['cancelled'] / cust_vol['total_claims']
    cust_vol['claim_frequency_rate'] = cust_vol['total_claims'] / cust_span['months_active'] * 12

    kpis = cust_vol[['claim_frequency_rate', 'cancellation_rate']].join(
        resolved.groupby('customer_id').agg(
            avg_cost_per_claim  =('actual_cost_eur',     'mean'),
            avg_repair_duration =('repair_duration_days', 'mean'),
            estimation_accuracy =('cost_delta_pct',      'mean'),
        )
    )

    # ── Customer × month panel ────────────────────────────────────────────────
    hist = df[df['claim_date'] <= TODAY].copy()
    all_months = pd.date_range(
        hist['claim_date'].min().to_period('M').to_timestamp(),
        hist['claim_date'].max().to_period('M').to_timestamp(),
        freq='MS',
    )
    all_customers = hist['customer_id'].unique()

    grid = pd.MultiIndex.from_product(
        [all_customers, all_months], names=['customer_id', 'month']
    ).to_frame(index=False)
    grid['month'] = pd.to_datetime(grid['month'])

    mc = (
        hist.assign(month=hist['claim_date'].dt.to_period('M').dt.to_timestamp())
        .groupby(['customer_id', 'month'])['claim_id']
        .count()
        .reset_index(name='claim_count')
    )
    panel = grid.merge(mc, on=['customer_id', 'month'], how='left').fillna({'claim_count': 0})
    panel['claim_count']   = panel['claim_count'].astype(int)
    panel['month_of_year'] = panel['month'].dt.month
    panel['year']          = panel['month'].dt.year
    panel = panel.sort_values(['customer_id', 'month']).reset_index(drop=True)

    cust_mean = panel.groupby('customer_id')['claim_count'].transform('mean')
    panel['rolling_12m_rate'] = (
        panel.groupby('customer_id')['claim_count']
        .transform(lambda x: x.shift(1).rolling(12, min_periods=3).mean())
    ).fillna(cust_mean)

    le = LabelEncoder()
    panel['customer_enc'] = le.fit_transform(panel['customer_id'])

    # ── Poisson GLM ───────────────────────────────────────────────────────────
    def make_X(d):
        X = d[FEATURES].copy().astype(float)
        X.insert(0, 'const', 1.0)
        return X

    train  = panel[panel['month'] <= TRAIN_END]
    result = sm.GLM(
        train['claim_count'], make_X(train), family=sm.families.Poisson()
    ).fit()
    if result.deviance / result.df_resid > 1.5:
        result = sm.GLM(
            train['claim_count'], make_X(train), family=sm.families.NegativeBinomial()
        ).fit()

    # ── Forecast Apr–Dec 2026 ─────────────────────────────────────────────────
    fc_months = pd.date_range('2026-04-01', '2026-12-01', freq='MS')
    fg = pd.DataFrame([
        {'customer_id': c, 'month': m}
        for c in all_customers for m in fc_months
    ])
    fg['month_of_year'] = fg['month'].dt.month
    fg['year']          = fg['month'].dt.year
    fg['customer_enc']  = le.transform(fg['customer_id'])
    fg['rolling_12m_rate'] = fg['customer_id'].map(
        panel.groupby('customer_id')['rolling_12m_rate'].last()
    )

    mu = result.predict(make_X(fg)).values
    fg['predicted_claims'] = mu
    fg['ci_low']  = stats.poisson.ppf(0.10, np.maximum(mu, 0.01)).clip(min=0)
    fg['ci_high'] = stats.poisson.ppf(0.90, np.maximum(mu, 0.01))

    rc  = resolved.groupby('customer_id')['actual_cost_eur'].agg(['mean', 'count'])
    gac = resolved['actual_cost_eur'].mean()
    rc['blended_cost']       = (rc['mean'] * rc['count'] + gac * 5) / (rc['count'] + 5)
    fg['avg_cost']           = fg['customer_id'].map(rc['blended_cost']).fillna(gac)
    fg['predicted_cost_eur'] = fg['predicted_claims'] * fg['avg_cost']

    annual = fg.groupby('customer_id').agg(
        predicted_claims  =('predicted_claims',   'sum'),
        predicted_cost_eur=('predicted_cost_eur', 'sum'),
        ci_low            =('ci_low',             'sum'),
        ci_high           =('ci_high',            'sum'),
    ).round(1)

    med_c = annual['predicted_claims'].median()
    med_e = annual['predicted_cost_eur'].median()
    annual['risk_segment'] = annual.apply(
        lambda r: (
            ('high-freq' if r['predicted_claims']   >= med_c else 'low-freq') + '/' +
            ('high-cost' if r['predicted_cost_eur'] >= med_e else 'low-cost')
        ),
        axis=1,
    )

    return df, active, resolved, kpis, panel, fg, annual


# ── Layout ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title='AFC Claims Dashboard', layout='wide')

df, active, resolved, kpis, panel, fg, annual = run_pipeline()

st.sidebar.title('Filter')
customer_options = ['All customers'] + sorted(df['customer_id'].unique().tolist())
selected     = st.sidebar.selectbox('Customer', customer_options)
is_filtered  = selected != 'All customers'

tab1, tab2, tab3 = st.tabs(['Portfolio', 'Customer', 'Forecast'])


# ── Tab 1: Portfolio ──────────────────────────────────────────────────────────

with tab1:
    res_all = active[active['status'] == 'abgeschlossen']
    n_cancelled = (df['status'] == 'storniert').sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('Customers',          df['customer_id'].nunique())
    c2.metric('Active Claims',      f'{len(active):,}')
    c3.metric('Avg Cost / Claim',   f'€{res_all["actual_cost_eur"].mean():,.0f}')
    c4.metric('Avg Repair Duration',f'{res_all["repair_duration_days"].mean():.1f} days')
    c5.metric('Cancellation Rate',  f'{n_cancelled / len(df):.1%}')

    st.markdown('---')

    col_l, col_r = st.columns(2)

    with col_l:
        monthly = (
            active.set_index('claim_date')
            .resample('MS')['claim_id'].count()
            .reset_index()
        )
        monthly.columns = ['month', 'claims']
        monthly['rolling_avg'] = monthly['claims'].rolling(3, min_periods=1).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly['month'], y=monthly['claims'],
            fill='tozeroy', fillcolor='rgba(41,128,185,0.12)',
            line=dict(color=BLUE, width=1.5), name='Active claims',
        ))
        fig.add_trace(go.Scatter(
            x=monthly['month'], y=monthly['rolling_avg'],
            line=dict(color=BLUE, width=2.5, dash='dot'), name='3-month avg',
        ))
        fig.update_layout(
            title='Monthly claim volume', height=300,
            margin=dict(l=0, r=0, t=40, b=0), yaxis_title='Claims', xaxis_title=None,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        cat_order = (
            active.groupby('damage_category')['actual_cost_eur']
            .median().sort_values(ascending=False).index.tolist()
        )
        fig = px.box(
            active, y='damage_category', x='actual_cost_eur',
            category_orders={'damage_category': cat_order},
            labels={'actual_cost_eur': 'Actual cost (EUR)', 'damage_category': ''},
            title='Cost by damage category',
        )
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Customer risk scatter
    total_cost_map = active.groupby('customer_id')['actual_cost_eur'].sum()
    cust_plot = kpis.reset_index()
    cust_plot['total_cost']    = cust_plot['customer_id'].map(total_cost_map)
    cust_plot['risk_segment']  = cust_plot['customer_id'].map(annual['risk_segment'])

    fig = px.scatter(
        cust_plot,
        x='claim_frequency_rate', y='avg_cost_per_claim',
        size='total_cost', color='risk_segment', hover_name='customer_id',
        hover_data={
            'claim_frequency_rate': ':.1f',
            'avg_cost_per_claim':   ':,.0f',
            'avg_repair_duration':  ':.1f',
            'total_cost':           ':,.0f',
        },
        labels={
            'claim_frequency_rate': 'Claim frequency (per year)',
            'avg_cost_per_claim':   'Avg cost per claim (EUR)',
        },
        title='Customer risk — bubble size = total cost exposure',
    )
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 2: Customer ───────────────────────────────────────────────────────────

with tab2:
    if not is_filtered:
        st.info('Select a customer from the sidebar to view their profile.')
    else:
        cust_active = active[active['customer_id'] == selected]

        if selected in kpis.index:
            ck       = kpis.loc[selected]
            port_med = kpis.median()

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric(
                'Claim Frequency',
                f'{ck["claim_frequency_rate"]:.1f} /yr',
                delta=f'{ck["claim_frequency_rate"] - port_med["claim_frequency_rate"]:+.1f} vs median',
            )
            c2.metric(
                'Avg Cost / Claim',
                f'€{ck["avg_cost_per_claim"]:,.0f}',
                delta=f'€{ck["avg_cost_per_claim"] - port_med["avg_cost_per_claim"]:+,.0f} vs median',
            )
            c3.metric('Estimation Accuracy', f'{ck["estimation_accuracy"]:+.1%}')
            c4.metric('Cancellation Rate',   f'{ck["cancellation_rate"]:.1%}')
            c5.metric(
                'Avg Repair Duration',
                f'{ck["avg_repair_duration"]:.1f} days',
                delta=f'{ck["avg_repair_duration"] - port_med["avg_repair_duration"]:+.1f}d vs median',
            )

        st.markdown('---')

        col_l, col_r = st.columns(2)

        with col_l:
            fault_counts = cust_active['fault_type'].value_counts().reset_index()
            fault_counts.columns = ['fault_type', 'count']
            fig = px.pie(
                fault_counts, names='fault_type', values='count',
                hole=0.45, title='Fault type breakdown',
            )
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            cust_monthly = (
                cust_active[cust_active['status'] == 'abgeschlossen']
                .set_index('claim_date')
                .resample('MS')['actual_cost_eur'].sum()
                .reset_index()
            )
            cust_monthly.columns = ['month', 'total_cost']
            fig = px.bar(
                cust_monthly, x='month', y='total_cost',
                labels={'month': '', 'total_cost': 'Total resolved cost (EUR)'},
                title='Monthly resolved cost',
            )
            fig.update_traces(marker_color=BLUE)
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)


# ── Tab 3: Forecast ───────────────────────────────────────────────────────────

with tab3:
    if not is_filtered:
        annual_plot = annual.reset_index().sort_values('predicted_cost_eur', ascending=True)

        col_l, col_r = st.columns(2)

        with col_l:
            fig = px.bar(
                annual_plot, y='customer_id', x='predicted_claims',
                orientation='h', color='risk_segment',
                labels={'predicted_claims': 'Predicted claims (Apr–Dec 2026)', 'customer_id': ''},
                title='Predicted claim volume — all customers',
                category_orders={'customer_id': annual_plot['customer_id'].tolist()},
            )
            fig.update_layout(height=550, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            fig = px.bar(
                annual_plot, y='customer_id', x='predicted_cost_eur',
                orientation='h', color='risk_segment',
                labels={'predicted_cost_eur': 'Predicted cost (EUR, Apr–Dec 2026)', 'customer_id': ''},
                title='Predicted cost exposure — all customers',
                category_orders={'customer_id': annual_plot['customer_id'].tolist()},
            )
            fig.update_layout(height=550, margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('---')
        st.dataframe(
            annual.reset_index()
            .sort_values('predicted_cost_eur', ascending=False)
            .style.format({
                'predicted_claims'  : '{:.0f}',
                'predicted_cost_eur': '€{:,.0f}',
                'ci_low'            : '{:.0f}',
                'ci_high'           : '{:.0f}',
            }),
            use_container_width=True,
        )

    else:
        cust_hist = panel[panel['customer_id'] == selected][['month', 'claim_count']].copy()
        cust_fc   = fg[fg['customer_id'] == selected][
            ['month', 'predicted_claims', 'ci_low', 'ci_high']
        ].copy()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=cust_hist['month'], y=cust_hist['claim_count'],
            name='Historical', marker_color='#bdc3c7',
        ))
        fig.add_trace(go.Bar(
            x=cust_fc['month'], y=cust_fc['predicted_claims'],
            name='Forecast (80% CI)', marker_color=BLUE,
            error_y=dict(
                type='data', symmetric=False,
                array=(cust_fc['ci_high'] - cust_fc['predicted_claims']).values,
                arrayminus=(cust_fc['predicted_claims'] - cust_fc['ci_low']).values,
                color='#7f8c8d', thickness=1.5,
            ),
        ))
        fig.add_vline(
            x=TODAY.timestamp() * 1000,
            line_dash='dash', line_color='#7f8c8d',
            annotation_text='Today', annotation_position='top right',
        )
        fig.update_layout(
            title=f'{selected} — historical claims + 2026 forecast',
            height=440, margin=dict(l=0, r=0, t=50, b=0),
            yaxis_title='Claims', xaxis_title=None,
        )
        st.plotly_chart(fig, use_container_width=True)

        if selected in annual.index:
            row = annual.loc[selected]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Predicted Claims (Apr–Dec)', f'{row["predicted_claims"]:.0f}')
            c2.metric('Predicted Cost',              f'€{row["predicted_cost_eur"]:,.0f}')
            c3.metric('80% CI',                      f'{row["ci_low"]:.0f} – {row["ci_high"]:.0f}')
            c4.metric('Risk Segment',                row['risk_segment'])
