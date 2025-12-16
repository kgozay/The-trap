import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
import xlsxwriter

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SA DCF Pro Terminal",
    page_icon="🇿🇦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Light Financial Terminal Aesthetic
st.markdown("""
    <style>
    /* Global Font & Background */
    .reportview-container {
        background: #f8f9fa;
    }
    .main {
        background-color: #ffffff;
    }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 6px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    
    /* Custom Card Class for HTML injection */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 20px;
        border-radius: 8px;
        text-align: left;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .metric-label {
        font-size: 0.7rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        font-family: 'Courier New', monospace;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 5px;
    }
    
    /* Colors */
    .text-emerald { color: #059669; }
    .text-red { color: #dc2626; }
    .text-blue { color: #2563eb; }
    .text-purple { color: #7c3aed; }
    
    /* Buttons */
    .stButton>button {
        border-radius: 4px;
        border: 1px solid #cbd5e1;
        background-color: white;
        color: #334155;
    }
    .stButton>button:hover {
        border-color: #94a3b8;
        color: #0f172a;
    }
    
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. SIDEBAR & INPUTS
# -----------------------------------------------------------------------------
st.sidebar.title("🇿🇦 SA DCF Pro")
st.sidebar.caption("Financial Terminal v2.1")

# Helper function for "Toggle Buttons" / Precision Control
def input_widget(label, val, min_v, max_v, step_v, key, help_text=None):
    if precision_mode:
        return st.number_input(label, value=float(val), min_value=float(min_v), max_value=float(max_v), step=float(step_v), key=f"num_{key}", help=help_text)
    else:
        return st.slider(label, value=float(val), min_value=float(min_v), max_value=float(max_v), step=float(step_v), key=f"slid_{key}", help=help_text)

# --- Scenario Engine ---
st.sidebar.header("1. Scenario Selection")
scenario = st.sidebar.radio("Select Case", ["Base", "Bull", "Bear"], horizontal=True, label_visibility="collapsed")

# Precision Mode Toggle
st.sidebar.markdown("---")
precision_mode = st.sidebar.toggle("Precision Mode (+/- Buttons)", value=False, help="Switch to Number Inputs for fine-tuning")

# Defaults based on Scenario
if scenario == "Base":
    def_rev = 6.0; def_marg = 12.0; def_beta = 1.1; def_g = 4.0
elif scenario == "Bull":
    def_rev = 9.0; def_marg = 14.5; def_beta = 1.0; def_g = 5.0
else: # Bear
    def_rev = 2.0; def_marg = 8.0; def_beta = 1.4; def_g = 2.5

# --- Macro Inputs ---
with st.sidebar.expander("Macro & Rates", expanded=True):
    rf = input_widget("Risk-free Rate (SARB) %", 10.5, 7.0, 14.0, 0.1, "rf")
    erp = input_widget("Equity Risk Premium %", 6.5, 4.0, 10.0, 0.1, "erp")
    beta = input_widget("Beta", def_beta, 0.5, 2.0, 0.05, "beta")
    inflation = input_widget("Inflation (CPI) %", 4.5, 3.0, 8.0, 0.1, "inf")
    zar_impact = input_widget("ZAR Margin Impact (pts)", 0.0, -5.0, 5.0, 0.1, "zar")

# --- Operating Inputs ---
with st.sidebar.expander("Operating Drivers", expanded=True):
    # Number input always used for large currency values for ease of typing
    revenue_base = st.number_input("Base Revenue (Rm)", value=25000, step=100, help="Enter in millions (e.g., 25000 for R25bn)")
    rev_growth = input_widget("Revenue Growth %", def_rev, -5.0, 25.0, 0.5, "growth")
    ebit_margin = input_widget("EBIT Margin %", def_marg, 0.0, 40.0, 0.5, "margin")
    depr_pct = input_widget("Depreciation % Rev", 3.0, 1.0, 10.0, 0.1, "depr")
    tax_rate = input_widget("Tax Rate %", 27.0, 20.0, 35.0, 1.0, "tax")

# --- Reinvestment ---
with st.sidebar.expander("Reinvestment Strategy", expanded=False):
    input_mode = st.toggle("Auto-calc via ROIC?", value=False)

    if input_mode:
        st.caption("Reinvestment derived from Target ROIC")
        target_roic = input_widget("Target ROIC %", 15.0, 5.0, 40.0, 0.5, "roic")
        capex_pct, wc_pct = 0, 0
    else:
        capex_pct = input_widget("Capex % Revenue", 4.0, 1.0, 15.0, 0.1, "capex")
        wc_pct = input_widget("WC % Revenue", 10.0, 0.0, 25.0, 0.5, "wc")
        target_roic = 0

    g = input_widget("Terminal Growth (g) %", def_g, 0.0, 8.0, 0.1, "g")

# --- Capital Structure ---
with st.sidebar.expander("Capital Structure", expanded=False):
    net_debt = st.number_input("Net Debt (Rm)", value=5000, step=100)
    shares = st.number_input("Shares (m)", value=600, step=10)
    cost_of_debt = input_widget("Cost of Debt %", 11.0, 5.0, 18.0, 0.1, "kd")
    debt_weight = input_widget("Debt Weight %", 25.0, 0.0, 90.0, 5.0, "wd")

# --- M&A ---
with st.sidebar.expander("M&A / Synergies", expanded=False):
    enable_ma = st.checkbox("Enable Synergies")
    if enable_ma:
        rev_syn = input_widget("Revenue Synergies %", 0.0, 0.0, 20.0, 1.0, "rsyn")
        cost_syn = input_widget("Margin Uplift %", 0.0, 0.0, 10.0, 0.1, "csyn")
        integ_costs = st.number_input("Integration Costs (Rm)", 0, step=10)
    else:
        rev_syn, cost_syn, integ_costs = 0, 0, 0

# -----------------------------------------------------------------------------
# 3. CALCULATION ENGINE
# -----------------------------------------------------------------------------

def calculate_model():
    # WACC
    equity_weight = 1 - (debt_weight / 100)
    ke = (rf / 100) + beta * (erp / 100)
    kd_at = (cost_of_debt / 100) * (1 - (tax_rate / 100))
    wacc = (equity_weight * ke) + ((debt_weight / 100) * kd_at)

    # Forecast Logic
    forecast_data = []
    
    # Synergies
    base_rev_adjusted = revenue_base * (1 + rev_syn/100) if enable_ma else revenue_base
    previous_wc = base_rev_adjusted * (wc_pct / 100)
    invested_capital = 15000 # dummy starting IC
    
    for year in range(1, 6):
        # Revenue
        revenue = base_rev_adjusted * (1 + rev_growth/100)**year
        
        # Margin
        actual_margin = ebit_margin + zar_impact + (cost_syn if enable_ma else 0)
        ebit = revenue * (actual_margin / 100)
        nopat = ebit * (1 - tax_rate/100)
        depreciation = revenue * (depr_pct / 100)
        ebitda = ebit + depreciation
        
        # Reinvestment Logic
        if input_mode:
            # Reinvestment Rate = g / ROIC. Using revenue growth as proxy for g
            rr = (rev_growth / 100) / (target_roic / 100) if target_roic > 0 else 0
            total_reinv = nopat * rr
            capex = total_reinv * 0.8
            delta_wc = total_reinv * 0.2
        else:
            capex = revenue * (capex_pct / 100)
            target_wc = revenue * (wc_pct / 100)
            delta_wc = target_wc - previous_wc
            previous_wc = target_wc
            
        fcf = nopat - capex - delta_wc
        
        # Discounting
        df = 1 / (1 + wacc)**year
        pv_fcf = fcf * df
        
        # Metrics
        invested_capital += (capex - depreciation + delta_wc) # approx net new capital
        roic = nopat / invested_capital if invested_capital > 0 else 0
        
        # Debt Metrics
        interest_expense = net_debt * (cost_of_debt / 100)
        interest_cover = ebit / interest_expense if interest_expense > 0 else 0
        net_debt_ebitda = net_debt / ebitda if ebitda > 0 else 0

        forecast_data.append({
            "Year": year,
            "Revenue": revenue,
            "EBIT": ebit,
            "NOPAT": nopat,
            "FCF": fcf,
            "PV_FCF": pv_fcf,
            "ROIC": roic,
            "Interest_Cover": interest_cover,
            "NetDebt_EBITDA": net_debt_ebitda
        })
        
    df_forecast = pd.DataFrame(forecast_data)
    
    # Terminal Value
    fcf_n = df_forecast.iloc[-1]["FCF"]
    tv = (fcf_n * (1 + g/100)) / (wacc - g/100)
    pv_tv = tv / (1 + wacc)**5
    
    # Enterprise Value
    ev = df_forecast["PV_FCF"].sum() + pv_tv
    if enable_ma:
        ev -= integ_costs
        
    equity_val = ev - net_debt
    share_price = equity_val / shares
    
    # Implied Multiples
    y1 = df_forecast.iloc[0]
    y1_ebitda = y1["EBIT"] + (y1["Revenue"] * depr_pct/100)
    implied_ev_ebitda = ev / y1_ebitda if y1_ebitda > 0 else 0
    
    return {
        "wacc": wacc,
        "ev": ev,
        "equity_val": equity_val,
        "share_price": share_price,
        "df": df_forecast,
        "pv_tv": pv_tv,
        "roic_y1": df_forecast.iloc[0]["ROIC"],
        "implied_ev_ebitda": implied_ev_ebitda,
        "y1_metrics": df_forecast.iloc[0]
    }

# Run Calculation
res = calculate_model()

# -----------------------------------------------------------------------------
# 4. DASHBOARD UI
# -----------------------------------------------------------------------------

# --- Header ---
st.title("🇿🇦 SA DCF Pro Terminal")
st.markdown(f"**Scenario:** {scenario} | **G:** {g}% | **WACC:** {res['wacc']*100:.1f}%")

# --- Top Metrics Grid ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Implied Share Price</div>
        <div class="metric-value text-emerald">R {res['share_price']:,.2f}</div>
        <div class="metric-sub">Equity Value: R {res['equity_val']/1000:,.1f}bn</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Enterprise Value</div>
        <div class="metric-value text-blue">R {res['ev']/1000:,.1f}bn</div>
        <div class="metric-sub">PV TV: {(res['pv_tv']/res['ev'])*100:.0f}% of EV</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    spread = res['roic_y1'] - res['wacc']
    spread_color = "text-emerald" if spread > 0 else "text-red"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Return Profile</div>
        <div class="metric-value {spread_color}">{res['roic_y1']*100:.1f}%</div>
        <div class="metric-sub">Spread to WACC: {spread*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Valuation Multiple</div>
        <div class="metric-value text-purple">{res['implied_ev_ebitda']:.1f}x</div>
        <div class="metric-sub">Implied EV/EBITDA (fwd)</div>
    </div>
    """, unsafe_allow_html=True)

st.write("") # Spacer

# --- Main Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Forecast & Visuals", "🌡️ Sensitivity", "🎲 Monte Carlo", "📉 Debt & Covenants"])

with tab1:
    # 1. Forecast Chart
    col_chart, col_data = st.columns([2, 1])
    
    with col_chart:
        st.subheader("Free Cash Flow Forecast")
        
        # Plotly Area Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=res['df']['Year'], y=res['df']['FCF'], 
            fill='tozeroy', name='FCF', 
            line=dict(color='#10b981', width=3),
            fillcolor='rgba(16, 185, 129, 0.1)'
        ))
        fig.add_trace(go.Scatter(
            x=res['df']['Year'], y=res['df']['NOPAT'], 
            name='NOPAT', 
            line=dict(color='#6366f1', dash='dash', width=2)
        ))
        
        fig.update_layout(
            xaxis_title="Forecast Year",
            yaxis_title="Rand (millions)",
            template="plotly_white",
            height=350,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col_data:
        st.subheader("Forecast Data")
        # Format dataframe with commas
        display_df = res['df'].set_index('Year')[['Revenue', 'EBIT', 'FCF']].style.format("{:,.0f}")
        st.dataframe(display_df, use_container_width=True)
        
        # Flags
        st.markdown("### Diagnostics")
        if res['roic_y1'] < res['wacc']:
            st.error("⚠️ Value Destroying (ROIC < WACC)")
        else:
            st.success("✅ Value Creating (ROIC > WACC)")
            
        if (res['pv_tv'] / res['ev']) > 0.75:
            st.warning(f"⚠️ High TV Dependency ({(res['pv_tv']/res['ev'])*100:.0f}%)")

with tab2:
    st.subheader("Sensitivity Analysis: WACC vs Terminal Growth")
    st.caption("Matrix shows Implied Share Price (R)")
    
    # Calculate Matrix
    wacc_steps = np.array([-1.0, -0.5, 0.0, 0.5, 1.0]) / 100
    g_steps = np.array([-1.0, -0.5, 0.0, 0.5, 1.0]) / 100
    
    matrix_data = []
    current_wacc = res['wacc']
    current_g = g / 100
    
    for g_adj in g_steps:
        row = []
        for w_adj in wacc_steps:
            sim_wacc = current_wacc + w_adj
            sim_g = current_g + g_adj
            
            # Recalc Valuation roughly
            # Adjust forecast PV for WACC change
            sim_pv_fcf = sum([row['FCF'] / (1 + sim_wacc)**row['Year'] for _, row in res['df'].iterrows()])
            
            sim_tv = (res['df'].iloc[-1]['FCF'] * (1 + sim_g)) / (sim_wacc - sim_g)
            sim_pv_tv = sim_tv / (1 + sim_wacc)**5
            
            sim_ev = sim_pv_fcf + sim_pv_tv - integ_costs
            sim_price = (sim_ev - net_debt) / shares
            row.append(sim_price)
        matrix_data.append(row)
        
    # Display Heatmap
    cols = [f"{(current_wacc + w)*100:.1f}%" for w in wacc_steps]
    idx = [f"{(current_g + x)*100:.1f}%" for x in g_steps]
    
    sens_df = pd.DataFrame(matrix_data, columns=cols, index=idx)
    
    fig_heat = px.imshow(
        sens_df, 
        labels=dict(x="WACC", y="Terminal Growth (g)", color="Price"),
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdYlGn"
    )
    fig_heat.update_xaxes(side="top")
    st.plotly_chart(fig_heat, use_container_width=True)

with tab3:
    st.subheader("Monte Carlo Simulation")
    st.markdown("Probabilistic valuation based on 1,000 iterations of randomizing WACC (+/- 1%), Growth (+/- 0.5%), and Margins (+/- 2%).")
    
    if st.button("Run Simulation", type="primary"):
        iterations = 1000
        sim_results = []
        
        base_wacc = res['wacc']
        base_g = g / 100
        base_rev = rev_growth / 100
        base_marg = ebit_margin / 100
        
        with st.spinner("Crunching scenarios..."):
            for _ in range(iterations):
                # Randomize
                s_wacc = base_wacc + np.random.normal(0, 0.01)
                s_g = base_g + np.random.normal(0, 0.005)
                s_marg = base_marg + np.random.normal(0, 0.02)
                
                # Simplified Calc for speed
                # 1. Year 1 FCF proxy
                rev_y1 = revenue_base * (1 + base_rev)
                ebit_y1 = rev_y1 * s_marg
                nopat_y1 = ebit_y1 * (1 - tax_rate/100)
                reinv = nopat_y1 * 0.3 # approx 30% reinvestment
                fcf_y1 = nopat_y1 - reinv
                
                # Safety
                if s_g >= s_wacc: s_g = s_wacc - 0.01
                
                # PV FCF (Annuity approx)
                pv_fcf = 0
                for y in range(1, 6):
                    pv_fcf += fcf_y1 / (1 + s_wacc)**y
                
                tv = (fcf_y1 * (1 + s_g)) / (s_wacc - s_g)
                pv_tv = tv / (1 + s_wacc)**5
                
                val = (pv_fcf + pv_tv - net_debt) / shares
                if val > 0: sim_results.append(val)
        
        sim_results = np.array(sim_results)
        
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            st.metric("Bear Case (P10)", f"R {np.percentile(sim_results, 10):,.2f}")
            st.metric("Base Case (P50)", f"R {np.percentile(sim_results, 50):,.2f}")
            st.metric("Bull Case (P90)", f"R {np.percentile(sim_results, 90):,.2f}")
            
        with col_res2:
            fig_hist = px.histogram(sim_results, nbins=30, title="Valuation Distribution", labels={'value': 'Share Price'})
            fig_hist.add_vline(x=res['share_price'], line_dash="dash", line_color="red", annotation_text="Current")
            fig_hist.update_layout(showlegend=False, template="plotly_white")
            st.plotly_chart(fig_hist, use_container_width=True)

with tab4:
    st.subheader("Debt & Covenant Analysis")
    
    cd1, cd2 = st.columns(2)
    
    metrics = res['y1_metrics']
    nd_ebitda = metrics['NetDebt_EBITDA']
    int_cov = metrics['Interest_Cover']
    
    with cd1:
        st.markdown("#### Credit Metrics (Year 1)")
        st.metric("Net Debt / EBITDA", f"{nd_ebitda:.2f}x", delta_color="inverse" if nd_ebitda > 3.0 else "normal")
        st.metric("Interest Cover", f"{int_cov:.2f}x", delta_color="normal" if int_cov > 2.5 else "inverse")
        st.metric("Debt Ratio", f"{debt_weight}%")
        
    with cd2:
        st.markdown("#### Covenant Status")
        
        if nd_ebitda < 3.0:
            st.success("✅ Leverage Covenant OK (< 3.0x)")
        else:
            st.error("❌ Leverage Covenant BREACHED (> 3.0x)")
            
        if int_cov > 2.5:
            st.success("✅ Interest Cover OK (> 2.5x)")
        else:
            st.error("❌ Interest Cover BREACHED (< 2.5x)")

# -----------------------------------------------------------------------------
# 5. FOOTER & EXPORT
# -----------------------------------------------------------------------------
st.markdown("---")
st.write("### Export Results")

# Excel Export Logic
def to_excel(df, metrics):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    
    # Sheet 1: Forecast
    df.to_excel(writer, index=False, sheet_name='Forecast')
    
    # Sheet 2: Summary Metrics
    summary_data = {
        'Metric': ['Share Price', 'Enterprise Value', 'Equity Value', 'WACC', 'ROIC', 'Terminal Growth'],
        'Value': [metrics['share_price'], metrics['ev'], metrics['equity_val'], metrics['wacc'], metrics['roic_y1'], g]
    }
    pd.DataFrame(summary_data).to_excel(writer, index=False, sheet_name='Summary')
    
    writer.close()
    processed_data = output.getvalue()
    return processed_data

excel_file = to_excel(res['df'], res)

st.download_button(
    label="📥 Download Model to Excel",
    data=excel_file,
    file_name='dcf_valuation_model.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)

st.caption("Built with Streamlit & Python. Model logic based on standard DCF methodology.")
    