import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# --- CONSTANTS & DEFAULTS ---
DEFAULT_SCENARIOS = {
    "Base": {
        "rev_growth": 6.5, "ebit_margin": 8.5, "beta": 0.85, "g": 4.5,
        "exit_multiple": 7.5, "terminal_wacc_spread": 0.0
    },
    "Bull": {
        "rev_growth": 8.5, "ebit_margin": 10.0, "beta": 0.80, "g": 5.5,
        "exit_multiple": 9.0, "terminal_wacc_spread": -0.5
    },
    "Bear": {
        "rev_growth": 3.0, "ebit_margin": 6.0, "beta": 1.10, "g": 3.0,
        "exit_multiple": 6.0, "terminal_wacc_spread": 1.0
    }
}

# --- PAGE CONFIG ---
st.set_page_config(page_title="SA DCF Pro", layout="wide", page_icon="🇿🇦")

# --- CALCULATION ENGINE ---

def calculate_model(inputs):
    # Unpack inputs
    rf = inputs['rf']
    erp = inputs['erp']
    beta = inputs['beta']
    tax_rate = inputs['tax_rate']
    cost_of_debt = inputs['cost_of_debt']
    debt_weight = inputs['debt_weight']
    
    revenue_base = inputs['revenue_base']
    rev_growth = inputs['rev_growth']
    ebit_margin = inputs['ebit_margin']
    depr_pct = inputs['depr_pct'] # depreciation % revenue
    
    # Reinvestment
    capex_pct = inputs['capex_pct']
    wc_pct = inputs['wc_pct']
    
    # Terminal
    g = inputs['g']
    terminal_method = inputs['terminal_method'] # 'gordon', 'exit', 'blend'
    exit_multiple = inputs['exit_multiple']
    mid_year_convention = inputs['mid_year_convention']
    terminal_wacc_spread = inputs['terminal_wacc_spread']
    
    net_debt = inputs['net_debt']
    shares = inputs['shares']

    # 1. Rates
    equity_weight = 1 - (debt_weight / 100.0)
    ke = (rf / 100.0) + beta * (erp / 100.0)
    kd_at = (cost_of_debt / 100.0) * (1 - (tax_rate / 100.0))
    wacc = (equity_weight * ke) + ((debt_weight / 100.0) * kd_at)
    terminal_wacc = wacc + (terminal_wacc_spread / 100.0)

    # 2. Forecast Loop
    forecast_data = []
    
    # Initial State
    previous_wc = revenue_base * (wc_pct / 100.0)
    cumulative_pv_fcf = 0
    invested_capital = 15000 + previous_wc # Proxy starting IC

    for year in range(1, 6):
        revenue = revenue_base * ((1 + rev_growth/100.0) ** year)
        ebit = revenue * (ebit_margin / 100.0)
        nopat = ebit * (1 - tax_rate/100.0)
        depreciation = revenue * (depr_pct / 100.0)
        ebitda = ebit + depreciation
        
        # Reinvestment
        capex = revenue * (capex_pct / 100.0)
        target_wc = revenue * (wc_pct / 100.0)
        delta_wc = target_wc - previous_wc
        previous_wc = target_wc
        
        # FCF
        fcf = nopat + depreciation - capex - delta_wc
        
        # Discounting
        time_period = year - 0.5 if mid_year_convention else year
        df = 1 / ((1 + wacc) ** time_period)
        pv_fcf = fcf * df
        cumulative_pv_fcf += pv_fcf
        
        # ROIC & Credit
        invested_capital += (capex - depreciation + delta_wc)
        roic = nopat / invested_capital if invested_capital > 0 else 0
        
        interest_expense = net_debt * (cost_of_debt / 100.0)
        interest_cover = ebit / interest_expense if interest_expense > 0 else 0
        nd_ebitda = net_debt / ebitda if ebitda > 0 else 0
        
        forecast_data.append({
            "Year": year,
            "Revenue": revenue,
            "EBITDA": ebitda,
            "EBIT": ebit,
            "NOPAT": nopat,
            "FCF": fcf,
            "PV FCF": pv_fcf,
            "ROIC": roic,
            "Interest Cover": interest_cover,
            "Net Debt/EBITDA": nd_ebitda
        })

    forecast_df = pd.DataFrame(forecast_data)
    last_metrics = forecast_df.iloc[-1]

    # 3. Terminal Value
    tv = 0
    # Method A: Gordon Growth
    gordon_denom = (terminal_wacc - g/100.0)
    tv_gordon = (last_metrics["FCF"] * (1 + g/100.0)) / gordon_denom if gordon_denom > 0 else 0
    
    # Method B: Exit Multiple
    tv_multiple = last_metrics["EBITDA"] * exit_multiple
    
    if terminal_method == 'gordon':
        tv = tv_gordon
    elif terminal_method == 'exit':
        tv = tv_multiple
    else: # blend
        tv = (tv_gordon + tv_multiple) / 2.0
        
    pv_tv = tv / ((1 + terminal_wacc) ** 5)
    
    # 4. Valuation
    enterprise_value = cumulative_pv_fcf + pv_tv
    equity_value = enterprise_value - net_debt
    share_price = equity_value / shares 
    
    # Implied Multiple (Forward Y1)
    fwd_ebitda = forecast_df.iloc[0]["EBITDA"]
    implied_ev_ebitda = enterprise_value / fwd_ebitda if fwd_ebitda > 0 else 0

    return {
        "wacc": wacc,
        "terminal_wacc": terminal_wacc,
        "ev": enterprise_value,
        "equity_val": equity_value,
        "share_price": share_price,
        "forecast_df": forecast_df,
        "pv_tv": pv_tv,
        "roic_y1": forecast_df.iloc[0]["ROIC"],
        "implied_ev_ebitda": implied_ev_ebitda,
        "y1_metrics": forecast_df.iloc[0]
    }

# --- SIDEBAR ---
with st.sidebar:
    st.header("🇿🇦 SA DCF Pro")
    
    # Scenario Selection
    st.subheader("Scenario")
    selected_scenario = st.selectbox("Select Case", ["Base", "Bull", "Bear", "Custom"])
    
    # Initialize session state for inputs if not exists
    # We load defaults based on scenario selection
    defaults = DEFAULT_SCENARIOS.get(selected_scenario, DEFAULT_SCENARIOS["Base"])
    
    # Helper to clean code
    def inp(key, default, min_v, max_v, step=None, format=None):
        # If scenario changes, we might want to update these, but Streamlit widgets hold state.
        # Simple approach: let the user adjust manually, or reset if they want.
        # For this version, we set defaults but allow overrides.
        val = default
        if selected_scenario != "Custom" and key in defaults:
            val = defaults[key]
        return val

    with st.expander("Macro & Rates", expanded=True):
        rf = st.slider("Risk-free Rate (%)", 5.0, 15.0, 10.5, step=0.1)
        erp = st.slider("Equity Risk Premium (%)", 3.0, 12.0, 6.0, step=0.1)
        beta = st.columns([1])[0].number_input("Beta", 0.5, 2.5, inp("beta", 0.85, 0.5, 2.5), step=0.05)
        mid_year = st.checkbox("Mid-Year Convention", value=False)
        
    with st.expander("Operating Drivers", expanded=True):
        rev_base = st.number_input("Base Revenue (Rm)", value=87000, step=1000)
        rev_growth = st.slider("Revenue Growth (%)", -10.0, 30.0, inp("rev_growth", 6.5, -10.0, 30.0), step=0.5)
        ebit_margin = st.slider("EBIT Margin (%)", 0.0, 50.0, inp("ebit_margin", 8.5, 0.0, 50.0), step=0.5)
        tax_rate = st.slider("Tax Rate (%)", 15.0, 35.0, 27.0, step=1.0)
        depr_pct = st.slider("Depreciation % Rev", 0.0, 10.0, 3.5, step=0.5)
        
    with st.expander("Working Capital & Capex"):
        wc_pct = st.slider("WC % Revenue", 0.0, 30.0, 4.0, step=0.5)
        capex_pct = st.slider("Capex % Revenue", 0.0, 20.0, 3.5, step=0.5)
        
    with st.expander("Terminal Value"):
        term_method = st.selectbox("Methodology", ["blend", "gordon", "exit"])
        g = st.slider("Terminal Growth (g) %", 0.0, 10.0, inp("g", 4.5, 0.0, 8.0), step=0.1)
        exit_mult = st.slider("Exit EV/EBITDA x", 2.0, 25.0, inp("exit_multiple", 7.5, 2.0, 20.0), step=0.5)
        
    with st.expander("Capital Structure"):
        net_debt = st.number_input("Net Debt (Rm)", value=8500, step=100)
        shares = st.number_input("Shares (m)", value=1050, step=10)
        cost_debt = st.slider("Cost of Debt (%)", 5.0, 18.0, 11.5, step=0.5)
        debt_weight = st.slider("Debt Weight (%)", 0.0, 90.0, 15.0, step=5.0)
        term_wacc_spread = st.slider("Term. WACC Spread (%)", -2.0, 2.0, inp("terminal_wacc_spread", 0.0, -2.0, 2.0), step=0.1)

# Compile Inputs
inputs = {
    "rf": rf, "erp": erp, "beta": beta,
    "revenue_base": rev_base, "rev_growth": rev_growth, "ebit_margin": ebit_margin,
    "tax_rate": tax_rate, "depr_pct": depr_pct,
    "wc_pct": wc_pct, "capex_pct": capex_pct,
    "g": g, "exit_multiple": exit_mult, "terminal_method": term_method,
    "mid_year_convention": mid_year,
    "net_debt": net_debt, "shares": shares,
    "cost_of_debt": cost_debt, "debt_weight": debt_weight, "terminal_wacc_spread": term_wacc_spread
}

# --- RUN MODEL ---
model = calculate_model(inputs)

# --- HEADER ---
st.title("Valuation Dashboard")
cols = st.columns(4)
cols[0].metric("Implied Share Price", f"R {model['share_price']:,.2f}", f"Equity: R {model['equity_val']/1000:.1f}bn")
cols[1].metric("Enterprise Value", f"R {model['ev']/1000:.1f}bn", f"Use: {term_method.title()}")
cols[2].metric("WACC", f"{model['wacc']*100:.1f}%", f"Term WACC: {model['terminal_wacc']*100:.1f}%")
cols[3].metric("ROIC (Y1)", f"{model['roic_y1']*100:.1f}%", f"Spread: {(model['roic_y1']-model['wacc'])*100:.1f}%")


# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Forecast", "Scenarios", "Sensitivity", "Monte Carlo", "Debt"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Free Cash Flow Forecast")
        df_chart = model['forecast_df']
        
        # Altair Composed Chart
        base = alt.Chart(df_chart).encode(x=alt.X('Year:O', axis=alt.Axis(labelAngle=0)))
        
        bar = base.mark_bar(color='#cbd5e1').encode(y='EBITDA', tooltip=['Year', 'EBITDA'])
        area = base.mark_area(color='#10b981', opacity=0.3).encode(y='FCF', tooltip=['Year', 'FCF'])
        line = base.mark_line(color='#10b981').encode(y='FCF')
        
        c = (bar + area + line).interactive()
        st.altair_chart(c, use_container_width=True)
        
        st.dataframe(df_chart[["Year", "Revenue", "EBITDA", "NOPAT", "FCF", "ROIC"]].style.format("{:,.0f}"))
        
    with col2:
        st.subheader("Valuation Bridge")
        # Simple waterfall data
        waterfall_data = [
            {"Category": "Enterprise Value", "Value": model['ev'], "Color": "#3b82f6"},
            {"Category": "Net Debt", "Value": -model['net_debt'], "Color": "#ef4444"},
            {"Category": "Equity Value", "Value": model['equity_val'], "Color": "#10b981"}
        ]
        wf_df = pd.DataFrame(waterfall_data)
        
        # Simple Bar for waterfall (Altair waterfall is complex, using Bar for simplicity)
        c_wf = alt.Chart(wf_df).mark_bar().encode(
            x=alt.X('Category', sort=None),
            y='Value',
            color=alt.Color('Color', scale=None),
            tooltip=['Category', 'Value']
        )
        st.altair_chart(c_wf, use_container_width=True)

with tab2:
    st.subheader("Scenario Comparison")
    
    # Calculate all scenarios
    scen_results = []
    current_inputs = inputs.copy()
    
    for case, params in DEFAULT_SCENARIOS.items():
        # Merge params
        temp_inputs = current_inputs.copy()
        temp_inputs.update(params)
        res = calculate_model(temp_inputs)
        scen_results.append({
            "Scenario": case,
            "Share Price": res['share_price'],
            "EV": res['ev'],
            "ROIC": res['roic_y1']
        })
        
    scen_df = pd.DataFrame(scen_results)
    
    # Cards
    scols = st.columns(3)
    for i, row in scen_df.iterrows():
        scols[i].metric(row["Scenario"], f"R {row['Share Price']:.2f}", f"EV: R {row['EV']/1000:.1f}bn")
        
    # Football Field (Range Chart)
    st.subheader("Valuation Range")
    
    min_p = scen_df["Share Price"].min() * 0.9
    max_p = scen_df["Share Price"].max() * 1.1
    
    # Simple dot plot for football field
    chart_range = alt.Chart(scen_df).mark_circle(size=200).encode(
        x=alt.X('Share Price', scale=alt.Scale(domain=[min_p, max_p])),
        y='Scenario',
        color='Scenario',
        tooltip=['Scenario', 'Share Price', 'ROIC']
    ).properties(height=200)
    
    st.altair_chart(chart_range, use_container_width=True)

with tab3:
    st.subheader("Sensitivity Analysis")
    st.caption("Share Price (R) vs WACC and Terminal Growth/Multiple")
    
    # Axis Ranges
    wacc_steps = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    y_steps = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    # Base Vals
    base_wacc = model['terminal_wacc'] * 100
    base_y = inputs['exit_multiple'] if inputs['terminal_method'] == 'exit' else inputs['g']
    y_label = "Exit Multiple" if inputs['terminal_method'] == 'exit' else "Growth (g)"
    step_size = 1.0 if inputs['terminal_method'] == 'exit' else 0.5
    
    # Generate Table
    data = []
    columns = [f"{base_wacc + w:.1f}%" for w in wacc_steps]
    indices = [f"{base_y + (y * step_size):.1f}" for y in y_steps]
    
    sens_matrix = []
    
    for y_adj in y_steps:
        row = []
        for w_adj in wacc_steps:
            # Modify inputs
            # Need strict control, re-calculating whole model is cleaner
            temp_in = inputs.copy()
            # Approx WACC mod: adjust RF or Debt weight? No, just force terminal WACC in logic? 
            # In calculate_model we derive wacc. Let's adjust risk_free to shift wacc roughly.
            # Actually easier: Hack calculate_model to accept override? 
            # Better: Just sensitivity logic approximation as per React code for speed?
            # React code used approximation. Let's do full calc for accuracy in Python.
            
            # Adjust WACC by adjusting Terminal WACC Spread
            temp_in['terminal_wacc_spread'] = inputs['terminal_wacc_spread'] + w_adj
            
            # Adjust g or multiple
            if inputs['terminal_method'] == 'exit':
                temp_in['exit_multiple'] = base_y + (y_adj * step_size)
            else:
                temp_in['g'] = base_y + (y_adj * step_size)
            
            res = calculate_model(temp_in)
            row.append(res['share_price'])
        sens_matrix.append(row)
        
    sens_df = pd.DataFrame(sens_matrix, index=indices, columns=columns)
    
    # Display with gradient
    st.dataframe(sens_df.style.format("R {:.2f}").background_gradient(cmap="RdYlGn", axis=None))


with tab4:
    st.subheader("Monte Carlo Simulation")
    
    if st.button("Run Simulation (1,000 Iterations)"):
        results = []
        bar = st.progress(0)
        
        for i in range(1000):
            # Randomize
            s_margin = inputs['ebit_margin'] + np.random.normal(0, 1.5)
            s_rev = inputs['rev_growth'] + np.random.normal(0, 1.0)
            
            temp = inputs.copy()
            temp['ebit_margin'] = s_margin
            temp['rev_growth'] = s_rev
            
            res = calculate_model(temp)
            results.append(res['share_price'])
            
            if i % 100 == 0:
                bar.progress(i / 1000)
                
        bar.progress(100)
        
        res_series = pd.Series(results)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Bear (P10)", f"R {res_series.quantile(0.1):.2f}")
        c2.metric("Base (P50)", f"R {res_series.median():.2f}")
        c3.metric("Bull (P90)", f"R {res_series.quantile(0.9):.2f}")
        
        # Hist
        hist_chart = alt.Chart(pd.DataFrame({'Price': results})).mark_bar().encode(
            x=alt.X('Price', bin=alt.Bin(maxbins=30)),
            y='count()'
        )
        st.altair_chart(hist_chart, use_container_width=True)

with tab5:
    st.subheader("Credit Metrics (Year 1)")
    
    m = model['y1_metrics']
    col1, col2 = st.columns(2)
    
    col1.metric("Net Debt / EBITDA", f"{m['Net Debt/EBITDA']:.2f}x")
    col2.metric("Interest Cover", f"{m['Interest Cover']:.2f}x")
    
    if m['Net Debt/EBITDA'] > 3.0:
        st.error("⚠️ Leverage Breach: ND/EBITDA > 3.0x")
    else:
        st.success("✅ Leverage Compliant")
