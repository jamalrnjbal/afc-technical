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

# Friendly labels for risk segments shown to sales users
RISK_LABELS = {
    'high-freq/high-cost': 'High Risk',
    'high-freq/low-cost':  'Watch List',
    'low-freq/high-cost':  'Monitor',
    'low-freq/low-cost':   'Low Risk',
}
RISK_COLORS = {
    'High Risk':  '#e74c3c',
    'Watch List': '#e67e22',
    'Monitor':    '#f1c40f',
    'Low Risk':   '#27ae60',
}


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

    # Translate German status values to English
    status_map = {'storniert': 'cancelled', 'abgeschlossen': 'settled'}
    df['status_en'] = df['status'].map(status_map).fillna(df['status'])

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
    annual['risk_segment_raw'] = annual.apply(
        lambda r: (
            ('high-freq' if r['predicted_claims']   >= med_c else 'low-freq') + '/' +
            ('high-cost' if r['predicted_cost_eur'] >= med_e else 'low-cost')
        ),
        axis=1,
    )
    annual['risk_segment'] = annual['risk_segment_raw'].map(RISK_LABELS)

    return df, active, resolved, kpis, panel, fg, annual


# ── Layout ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title='AFC Claims Dashboard', layout='wide')

df, active, resolved, kpis, panel, fg, annual = run_pipeline()

st.sidebar.title('Filter')
customer_options = ['All clients'] + sorted(df['customer_id'].unique().tolist())
selected     = st.sidebar.selectbox('Client', customer_options)
is_filtered  = selected != 'All clients'

st.sidebar.markdown('---')
st.sidebar.markdown(
    '**About this dashboard**\n\n'
    'Shows claims history and a cost forecast for Apr–Dec 2026. '
    'Select a client to see their individual profile and outlook.'
)

tab1, tab2, tab3 = st.tabs(['Business Health', 'Client Profile', '2026 Outlook'])


# ── Tab 1: Business Health ────────────────────────────────────────────────────

with tab1:
    res_all     = active[active['status'] == 'abgeschlossen']
    n_cancelled = (df['status'] == 'storniert').sum()

    st.subheader('Portfolio at a Glance')
    st.caption('Key numbers across all clients, based on claims submitted to date.')

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        'Total Clients', df['customer_id'].nunique(),
        help='Number of unique clients with at least one claim on record.',
    )
    c2.metric(
        'Open Claims', f'{len(active):,}',
        help='Claims that are currently active (not cancelled).',
    )
    c3.metric(
        'Average Claim Cost', f'€{res_all["actual_cost_eur"].mean():,.0f}',
        help='Mean cost of fully settled claims across all clients.',
    )
    c4.metric(
        'Average Repair Time', f'{res_all["repair_duration_days"].mean():.1f} days',
        help='How long, on average, it takes to settle and close a claim.',
    )
    c5.metric(
        'Cancelled Claims', f'{n_cancelled / len(df):.1%}',
        help='Share of all submitted claims that were cancelled before settlement.',
    )

    st.markdown('---')

    # ── Clients needing attention ─────────────────────────────────────────────
    attention = annual[annual['risk_segment'].isin(['High Risk', 'Watch List'])].reset_index()
    if len(attention):
        st.subheader(f'⚠ {len(attention)} Client(s) Needing Attention')
        st.caption(
            'These clients are forecast to generate the highest number of claims '
            'and/or the highest costs in the rest of 2026. Consider proactive '
            'outreach or pricing review before renewal.'
        )
        cols = st.columns(min(len(attention), 4))
        for i, (_, row) in enumerate(attention.iterrows()):
            color = RISK_COLORS.get(row['risk_segment'], BLUE)
            cols[i % 4].markdown(
                f"**{row['customer_id']}**  \n"
                f"<span style='color:{color};font-weight:bold'>{row['risk_segment']}</span>  \n"
                f"~{row['predicted_claims']:.0f} claims · €{row['predicted_cost_eur']:,.0f} forecast",
                unsafe_allow_html=True,
            )
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
            line=dict(color=BLUE, width=1.5), name='Claims',
        ))
        fig.add_trace(go.Scatter(
            x=monthly['month'], y=monthly['rolling_avg'],
            line=dict(color=BLUE, width=2.5, dash='dot'), name='3-month trend',
        ))
        fig.update_layout(
            title='Claims Per Month (all clients)',
            height=300,
            margin=dict(l=0, r=0, t=40, b=0),
            yaxis_title='Number of Claims',
            xaxis_title=None,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption('The dotted line shows the 3-month rolling average — useful for spotting trends.')

    with col_r:
        cat_order = (
            active.groupby('damage_category')['actual_cost_eur']
            .median().sort_values(ascending=False).index.tolist()
        )
        fig = px.box(
            active, y='damage_category', x='actual_cost_eur',
            category_orders={'damage_category': cat_order},
            labels={
                'actual_cost_eur':   'Settled Cost (EUR)',
                'damage_category':   '',
            },
            title='Repair Cost by Damage Type',
        )
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption('Each bar shows the typical cost range for that damage type. The centre line is the median.')

    # ── Customer risk scatter ─────────────────────────────────────────────────
    st.subheader('Client Risk Overview')
    st.caption(
        'Each bubble is a client. Position shows how often they claim (left–right) '
        'and how expensive those claims are (up–down). Bigger bubble = higher total cost. '
        'Colour shows their risk category.'
    )

    total_cost_map = active.groupby('customer_id')['actual_cost_eur'].sum()
    cust_plot = kpis.reset_index()
    cust_plot['total_cost']   = cust_plot['customer_id'].map(total_cost_map)
    cust_plot['risk_segment'] = cust_plot['customer_id'].map(annual['risk_segment'])

    fig = px.scatter(
        cust_plot,
        x='claim_frequency_rate', y='avg_cost_per_claim',
        size='total_cost', color='risk_segment',
        color_discrete_map=RISK_COLORS,
        hover_name='customer_id',
        hover_data={
            'claim_frequency_rate': ':.1f',
            'avg_cost_per_claim':   ':,.0f',
            'avg_repair_duration':  ':.1f',
            'total_cost':           ':,.0f',
        },
        labels={
            'claim_frequency_rate': 'Claims Per Year',
            'avg_cost_per_claim':   'Average Cost Per Claim (EUR)',
            'risk_segment':         'Risk Level',
            'total_cost':           'Total Cost (EUR)',
            'avg_repair_duration':  'Avg Repair Days',
        },
        title='Client Risk Map — bubble size = total cost to date',
        category_orders={'risk_segment': ['High Risk', 'Watch List', 'Monitor', 'Low Risk']},
    )
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 2: Client Profile ─────────────────────────────────────────────────────

with tab2:
    if not is_filtered:
        st.info('Select a client from the sidebar to view their individual profile.')
    else:
        cust_active = active[active['customer_id'] == selected]

        st.subheader(f'Client: {selected}')

        if selected in kpis.index:
            ck       = kpis.loc[selected]
            port_med = kpis.median()

            st.caption(
                'KPIs for this client compared to the portfolio median. '
                'Arrows show whether this client is above or below average.'
            )

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric(
                'Claims Per Year',
                f'{ck["claim_frequency_rate"]:.1f}',
                delta=f'{ck["claim_frequency_rate"] - port_med["claim_frequency_rate"]:+.1f} vs avg',
                help='How many claims this client submits per year on average.',
            )
            c2.metric(
                'Average Claim Cost',
                f'€{ck["avg_cost_per_claim"]:,.0f}',
                delta=f'€{ck["avg_cost_per_claim"] - port_med["avg_cost_per_claim"]:+,.0f} vs avg',
                help='Average cost of a settled claim for this client.',
            )
            c3.metric(
                'Quote Accuracy',
                f'{ck["estimation_accuracy"]:+.1%}',
                help=(
                    'How close our initial cost estimate was to the final settled amount. '
                    'A positive number means final costs ran over the estimate; '
                    'negative means we over-estimated.'
                ),
            )
            c4.metric(
                'Cancelled Claims',
                f'{ck["cancellation_rate"]:.1%}',
                help='Share of this client\'s claims that were cancelled before settlement.',
            )
            c5.metric(
                'Average Repair Time',
                f'{ck["avg_repair_duration"]:.1f} days',
                delta=f'{ck["avg_repair_duration"] - port_med["avg_repair_duration"]:+.1f}d vs avg',
                help='Average number of days from claim submission to settlement.',
            )

        st.markdown('---')

        col_l, col_r = st.columns(2)

        with col_l:
            fault_counts = cust_active['fault_type'].value_counts().reset_index()
            fault_counts.columns = ['fault_type', 'count']
            fig = px.pie(
                fault_counts, names='fault_type', values='count',
                hole=0.45,
                title='What Is Causing This Client\'s Claims?',
            )
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.caption('Breakdown of claim causes — useful for identifying patterns or coaching opportunities.')

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
                labels={'month': '', 'total_cost': 'Total Settled Cost (EUR)'},
                title='Monthly Settled Claim Cost',
            )
            fig.update_traces(marker_color=BLUE)
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.caption('Total cost of claims fully settled each month. Spikes may indicate seasonal patterns or incidents.')


# ── Tab 3: 2026 Outlook ───────────────────────────────────────────────────────

with tab3:
    if not is_filtered:
        st.subheader('Forecast: Apr – Dec 2026')
        st.caption(
            'This forecast uses historical claim patterns to predict how many claims each client '
            'is likely to submit over the rest of 2026, and what those claims are likely to cost. '
            'Use this to prioritise renewals, set reserves, or flag accounts for review.'
        )

        annual_plot = annual.reset_index().sort_values('predicted_cost_eur', ascending=True)

        col_l, col_r = st.columns(2)

        with col_l:
            fig = px.bar(
                annual_plot, y='customer_id', x='predicted_claims',
                orientation='h', color='risk_segment',
                color_discrete_map=RISK_COLORS,
                labels={
                    'predicted_claims': 'Expected Claims (Apr–Dec 2026)',
                    'customer_id':      '',
                    'risk_segment':     'Risk Level',
                },
                title='Expected Number of Claims per Client',
                category_orders={
                    'customer_id':  annual_plot['customer_id'].tolist(),
                    'risk_segment': ['High Risk', 'Watch List', 'Monitor', 'Low Risk'],
                },
            )
            fig.update_layout(height=550, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            fig = px.bar(
                annual_plot, y='customer_id', x='predicted_cost_eur',
                orientation='h', color='risk_segment',
                color_discrete_map=RISK_COLORS,
                labels={
                    'predicted_cost_eur': 'Expected Cost (EUR, Apr–Dec 2026)',
                    'customer_id':        '',
                    'risk_segment':       'Risk Level',
                },
                title='Expected Cost Exposure per Client',
                category_orders={
                    'customer_id':  annual_plot['customer_id'].tolist(),
                    'risk_segment': ['High Risk', 'Watch List', 'Monitor', 'Low Risk'],
                },
            )
            fig.update_layout(height=550, margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('---')
        st.subheader('Full Forecast Table')
        st.caption(
            '"Likely Range" shows the lower and upper bound of expected claims — '
            'there is an 80% chance the actual figure will fall within this range.'
        )

        display_df = (
            annual.reset_index()
            .sort_values('predicted_cost_eur', ascending=False)
            .rename(columns={
                'customer_id':        'Client',
                'predicted_claims':   'Expected Claims',
                'predicted_cost_eur': 'Expected Cost (EUR)',
                'ci_low':             'Low Estimate (Claims)',
                'ci_high':            'High Estimate (Claims)',
                'risk_segment':       'Risk Level',
            })
        )
        st.dataframe(
            display_df.style.format({
                'Expected Claims':      '{:.0f}',
                'Expected Cost (EUR)':  '€{:,.0f}',
                'Low Estimate (Claims)': '{:.0f}',
                'High Estimate (Claims)': '{:.0f}',
            }),
            use_container_width=True,
        )

    else:
        st.subheader(f'2026 Outlook: {selected}')
        st.caption(
            'Grey bars show actual claims from the past. Blue bars show what we expect '
            'for the rest of 2026, with error bars indicating the likely range.'
        )

        cust_hist = panel[panel['customer_id'] == selected][['month', 'claim_count']].copy()
        cust_fc   = fg[fg['customer_id'] == selected][
            ['month', 'predicted_claims', 'ci_low', 'ci_high']
        ].copy()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=cust_hist['month'], y=cust_hist['claim_count'],
            name='Historical claims', marker_color='#bdc3c7',
        ))
        fig.add_trace(go.Bar(
            x=cust_fc['month'], y=cust_fc['predicted_claims'],
            name='Forecast (with likely range)', marker_color=BLUE,
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
            title=f'{selected} — Past Claims & 2026 Forecast',
            height=440, margin=dict(l=0, r=0, t=50, b=0),
            yaxis_title='Number of Claims', xaxis_title=None,
        )
        st.plotly_chart(fig, use_container_width=True)

        if selected in annual.index:
            row = annual.loc[selected]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                'Expected Claims (Apr–Dec)',
                f'{row["predicted_claims"]:.0f}',
                help='Total number of claims we expect this client to submit over the rest of 2026.',
            )
            c2.metric(
                'Expected Cost',
                f'€{row["predicted_cost_eur"]:,.0f}',
                help='Forecast total cost of those claims based on this client\'s historical average.',
            )
            c3.metric(
                'Likely Range',
                f'{row["ci_low"]:.0f} – {row["ci_high"]:.0f} claims',
                help='There is an 80% chance the actual claim count will fall within this range.',
            )
            c4.metric(
                'Risk Level',
                row['risk_segment'],
                help=(
                    'High Risk: high claim frequency AND high cost. '
                    'Watch List: high frequency, lower cost. '
                    'Monitor: lower frequency, higher cost. '
                    'Low Risk: low frequency and low cost.'
                ),
            )
