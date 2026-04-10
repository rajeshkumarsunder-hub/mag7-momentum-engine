import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import gspread
import json
from datetime import datetime

warnings.filterwarnings('ignore')

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Mag 7 Momentum Engine", layout="wide")
st.title("Mag 7 Quantitative Momentum Engine (V4.4)")

# Put the button first so it renders at the top
run_pressed = st.sidebar.button("Run Simulation", type="primary", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.caption("👨‍💻 Developed by Rajesh")

# Then render all your inputs below it
st.sidebar.header("Simulation Parameters")
# start_year = st.sidebar.selectbox("Start Year", [str(y) for y in range(2010, 2025)], index=2)
start_year = str(st.sidebar.number_input("Start Year", min_value=1990, max_value=2026, value=2012, step=1))
start_month = st.sidebar.selectbox("Start Month", [f"{m:02d}" for m in range(1, 13)], index=0)
starting_lump_sum = st.sidebar.number_input("Starting Lump Sum ($)", min_value=1, value=1, step=1000)
monthly_sip = st.sidebar.number_input("Monthly SIP ($)", min_value=0, value=100, step=100)

with open("Mag7_Strategy_HowTo.pdf", "rb") as pdf_file:
    PDFbyte = pdf_file.read()

import os

pdf_path = "Mag7_Strategy_HowTo.pdf"

if os.path.exists(pdf_path):
    with open(pdf_path, "rb") as pdf_file:
        PDFbyte = pdf_file.read()

    st.sidebar.download_button(
        label="📄 Download Strategy Playbook",
        data=PDFbyte,
        file_name="Mag7_Strategy_HowTo.pdf",
        mime="application/pdf"
    )
else:
    st.sidebar.warning("⚠️ Strategy PDF is currently unavailable.")


st.sidebar.markdown("---")
st.sidebar.header("Engine Settings (Locked)")
st.sidebar.text("Execution Day: 05th\nRegime Filter: 200D SMA (97% Buffer)\nMomentum Hurdle: +1.25%\nLookback: 63 Days\n")
st.sidebar.text("tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA']")



EXECUTION_DAY = 5
REGIME_BUFFER = 0.97
MIN_ABS_MOMENTUM = 0.0125
ANNUAL_RISK_FREE = 0.0
MOMENTUM_WINDOW = 63

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA']
benchmark = 'QQQ'
broad_market = 'SPY'
# end_date = '2026-03-28'
end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

user_start_date = f"{start_year}-{start_month}-01"
fetch_start_date = (pd.to_datetime(user_start_date) - pd.DateOffset(years=1)).strftime('%Y-%m-%d')

@st.cache_data(show_spinner="Downloading Market Data...")
def load_data(start, end):
    data = yf.download(tickers + [benchmark, broad_market], start=start, end=end)['Close']
    data = data.ffill()
    data[benchmark] = data[benchmark].bfill()
    data[broad_market] = data[broad_market].bfill()
    return data

if run_pressed:
  
    data = load_data(fetch_start_date, end_date)
    exec_prices_df = data[tickers].shift(-1)

    qqq_sma_200 = data[benchmark].rolling(window=200).mean()
    regime_threshold = qqq_sma_200 * REGIME_BUFFER
    risk_on = data[benchmark] > regime_threshold
    momentum = data[tickers].pct_change(periods=MOMENTUM_WINDOW)

    DAILY_RF = (1 + ANNUAL_RISK_FREE) ** (1 / 252) - 1

    slots = [
        {'id': 1, 'ticker': None, 'cash': starting_lump_sum / 3, 'shares': 0, 'cost_basis': 0},
        {'id': 2, 'ticker': None, 'cash': starting_lump_sum / 3, 'shares': 0, 'cost_basis': 0},
        {'id': 3, 'ticker': None, 'cash': starting_lump_sum / 3, 'shares': 0, 'cost_basis': 0},
    ]

    total_invested_principal = starting_lump_sum
    history = []
    trade_log = []
    # --- NEW: Master Ledger for XIRR (Inflows are negative) ---
    base_cash_flows = [(pd.to_datetime(user_start_date), -float(starting_lump_sum))]
    
    sim_start_mask = data.index >= pd.to_datetime(user_start_date)
    if not sim_start_mask.any():
        st.error("Start date is out of range.")
        st.stop()
        
    first_valid_row = data[sim_start_mask].iloc[0]
    qqq_shares = starting_lump_sum / float(first_valid_row[benchmark]) if starting_lump_sum > 0 else 0
    spy_shares = starting_lump_sum / float(first_valid_row[broad_market]) if starting_lump_sum > 0 else 0

    mag7_shares = {ticker: 0 for ticker in tickers}
    if starting_lump_sum > 0:
        valid_initial = [t for t in tickers if not pd.isna(first_valid_row[t])]
        if valid_initial:
            for t in valid_initial:
                mag7_shares[t] = (starting_lump_sum / len(valid_initial)) / float(first_valid_row[t])

    current_month = -1

    def format_pnl(val):
        if val == 0: return "$0"
        return f"+${val:,.0f}" if val > 0 else f"-${abs(val):,.0f}"

    progress_bar = st.progress(0)
    total_days = len(data[sim_start_mask])
    day_count = 0

    for date, row in data.iterrows():
        if date < pd.to_datetime(user_start_date): continue
        if pd.isna(qqq_sma_200.loc[date]) or pd.isna(momentum.loc[date]).all(): continue

        day_count += 1
        if day_count % 100 == 0: progress_bar.progress(day_count / total_days)

        current_prices = row[tickers]
        exec_row = exec_prices_df.loc[date]
        exec_prices = exec_row.where(exec_row.notna(), other=current_prices)
        is_risk_on = risk_on.loc[date]
        is_execution_day = False

        if date.day >= EXECUTION_DAY and date.month != current_month:
            is_execution_day = True
            current_month = date.month
            # --- NEW: Record the exact date of the SIP deposit ---
            if monthly_sip > 0:
                base_cash_flows.append((date, -float(monthly_sip)))
            total_invested_principal += monthly_sip
            qqq_shares += monthly_sip / float(row[benchmark])
            spy_shares += monthly_sip / float(row[broad_market])
            
            valid_sip_tickers = [t for t in tickers if not pd.isna(row[t])]
            if valid_sip_tickers:
                for ticker in valid_sip_tickers:
                    mag7_shares[ticker] += (monthly_sip / len(valid_sip_tickers)) / float(row[ticker])

        current_val = sum(s['cash'] + (s['shares'] * float(current_prices[s['ticker']]) if s['ticker'] else 0) for s in slots)
        val_qqq = qqq_shares * float(row[benchmark])
        val_spy = spy_shares * float(row[broad_market])
        
        val_mag7 = 0
        for ticker in tickers:
            price = float(current_prices[ticker])
            if not pd.isna(price): val_mag7 += mag7_shares[ticker] * price

        history.append({
            'Date': date, 'Strategy_Value': current_val, 'QQQ_Hold': val_qqq, 
            'SPY_Hold': val_spy, 'Mag7_Hold': val_mag7, 'Total_Principal': total_invested_principal, 'Risk_On': is_risk_on
        })

        if not is_risk_on:
            sold_anything = False
            daily_realized_pnl = 0
            action_details = []

            for s in slots:
                if s['ticker'] is not None:
                    sold_anything = True
                    sell_price = float(exec_prices[s['ticker']])
                    sold_val = s['shares'] * sell_price
                    trade_pnl = sold_val - s['cost_basis']
                    daily_realized_pnl += trade_pnl
                    action_details.append(f"Sold {s['ticker']} ({format_pnl(trade_pnl)})")
                    s['cash'] += sold_val
                    s['shares'] = 0
                    s['ticker'] = None
                    s['cost_basis'] = 0

            if is_execution_day:
                for s in slots: s['cash'] += monthly_sip / 3

            for s in slots: s['cash'] *= (1 + DAILY_RF)
            total_cash = sum(s['cash'] for s in slots)

            if sold_anything or is_execution_day:
                trade_log.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'Event': 'RISK OFF — EVACUATION' if sold_anything else 'SAFE HARBOR',
                    'Strategy Action': f"200D trend broken. {', '.join(action_details)}" if sold_anything else f"Market below 200D MA. Added ${monthly_sip:,} SIP to cash.",
                    'Realized PnL': format_pnl(daily_realized_pnl) if sold_anything else '$0',
                    'Active Holdings': "None",
                    'Total Cash': f"${total_cash:,.0f}", 'Strategy Value': f"${current_val:,.0f}", 'Mag7 Benchmark': f"${val_mag7:,.0f}", 'QQQ Benchmark': f"${val_qqq:,.0f}", 'SPY Benchmark': f"${val_spy:,.0f}"
                })
            continue 

        if is_risk_on and is_execution_day:
            action_log, monthly_realized_pnl = [], 0
            current_mom = momentum.loc[date].dropna()
            positive_mom = current_mom[current_mom > MIN_ABS_MOMENTUM]
            current_top3 = positive_mom.sort_values(ascending=False).index.tolist()[:3]

            if current_mom[current_mom <= MIN_ABS_MOMENTUM].index.tolist():
                action_log.append(f"Abs. mom filter blocked: {current_mom[current_mom <= MIN_ABS_MOMENTUM].index.tolist()}")

            for s in slots:
                if s['ticker'] is not None and s['ticker'] not in current_top3:
                    sell_price = float(exec_prices[s['ticker']])
                    sold_val = s['shares'] * sell_price
                    trade_pnl = sold_val - s['cost_basis']
                    monthly_realized_pnl += trade_pnl
                    s['cash'] += sold_val
                    action_log.append(f"Slot {s['id']}: Sold {s['ticker']} (PnL: {format_pnl(trade_pnl)})")
                    s['shares'], s['ticker'], s['cost_basis'] = 0, None, 0

            for s in slots:
                if s['ticker'] is not None:
                    buy_price = float(exec_prices[s['ticker']])
                    s['shares'] += (monthly_sip / 3) / buy_price
                    s['cost_basis'] += monthly_sip / 3
                    action_log.append(f"Slot {s['id']}: SIP topped up {s['ticker']} (${monthly_sip/3:,.0f})")
                else:
                    s['cash'] += monthly_sip / 3

            empty_slots = [s for s in slots if s['ticker'] is None]
            if empty_slots:
                alloc_per_slot = sum(s['cash'] for s in empty_slots) / len(empty_slots)
                for s in empty_slots: s['cash'] = alloc_per_slot
                rookies = [t for t in current_top3 if t not in [s['ticker'] for s in slots if s['ticker'] is not None]]

                for s in empty_slots:
                    if rookies:
                        new_ticker = rookies.pop(0)
                        buy_price = float(exec_prices[new_ticker])  
                        s['ticker'], s['shares'], s['cost_basis'] = new_ticker, s['cash'] / buy_price, s['cash']
                        action_log.append(f"Slot {s['id']}: Bought {new_ticker} @ ${buy_price:.2f} (${s['cash']:,.0f})")
                        s['cash'] = 0
                    else:
                        action_log.append(f"Slot {s['id']}: No rookie available — hoarding ${s['cash']:,.0f}")

            active_str = " | ".join(f"Slot {s['id']} ({s['ticker']}): ${s['shares'] * float(current_prices[s['ticker']]):,.0f}" for s in slots if s['ticker'] is not None)
            
            trade_log.append({
                'Date': date.strftime('%Y-%m-%d'), 'Event': 'RISK ON — MONTHLY UPDATE',
                'Strategy Action': " | ".join(action_log) if action_log else "No rotation. SIP distributed.",
                'Realized PnL': format_pnl(monthly_realized_pnl),
                'Active Holdings': active_str or "None", 'Total Cash': f"${sum(s['cash'] for s in slots):,.0f}",
                'Strategy Value': f"${current_val:,.0f}", 'Mag7 Benchmark': f"${val_mag7:,.0f}", 'QQQ Benchmark': f"${val_qqq:,.0f}", 'SPY Benchmark': f"${val_spy:,.0f}"
            })

    progress_bar.empty()

    results_df = pd.DataFrame(history).set_index('Date')
    final_principal = results_df['Total_Principal'].iloc[-1]
# ---  SHEET LOGGER ---
    try:
        # Pull the locked keys from Streamlit Secrets
        credentials_dict = json.loads(st.secrets["google_credentials"])
        gc = gspread.service_account_from_dict(credentials_dict)
        
        # Connect to your spreadsheet
        sheet = gc.open("Mag7_TrafficLogger").sheet1
        
        # Extract the final values and force them into standard Python floats
        final_principal = float(results_df['Total_Principal'].iloc[-1])
        final_strategy = float(round(results_df['Strategy_Value'].iloc[-1], 2))
        final_mag7 = float(round(results_df['Mag7_Hold'].iloc[-1], 2))
        final_qqq = float(round(results_df['QQQ_Hold'].iloc[-1], 2))
        final_spy = float(round(results_df['SPY_Hold'].iloc[-1], 2))

        # Log the 9-item data row
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        payload = [
            timestamp, 
            start_year, 
            starting_lump_sum, 
            monthly_sip, 
            final_principal, 
            final_strategy, 
            final_mag7, 
            final_qqq, 
            final_spy
        ]
        
        sheet.append_row(payload)
        
    except Exception as e:
        if "200" in str(e):
            pass 
        else:
            st.error(f"Logger Error: {e}") # Let's see the error!
    # ---------------------------
    
    
    def get_drawdown(series): return (series - series.cummax()) / series.cummax()
    for col in ['Strategy', 'Mag7', 'QQQ', 'SPY']:
        results_df[f'{col}_DD'] = get_drawdown(results_df[f'{col}_Value' if col == 'Strategy' else f'{col}_Hold'])

    years = (results_df.index[-1] - results_df.index[0]).days / 365.25
    # --- TRUE XIRR CALCULATION (Bisection Method) ---
    def calc_xirr(final_value, base_cfs):
        final_date = results_df.index[-1]
        # Append the final portfolio value as a positive withdrawal
        cfs = base_cfs + [(final_date, float(final_value))]
        t0 = cfs[0][0]
        
        def xnpv(rate):
            if rate <= -1.0: return float('inf')
            return sum([cf / (1.0 + rate)**((d - t0).days / 365.25) for d, cf in cfs])
        
        # Bisection solver to find the rate where NPV equals 0
        low, high = -0.99, 100.0  # Range: -99% to +10,000%
        for _ in range(100):
            mid = (low + high) / 2.0
            val = xnpv(mid)
            if abs(val) < 0.01: break
            if val > 0:
                low = mid  # Rate is too low
            else:
                high = mid # Rate is too high
        return mid * 100

    st.markdown("### Performance Metrics")
    cols = st.columns(4)
    
    metrics = [
        ("Top 3 Strategy", results_df['Strategy_Value'], results_df['Strategy_DD']),
        ("Mag 7 Equal Weight", results_df['Mag7_Hold'], results_df['Mag7_DD']),
        ("QQQ Benchmark", results_df['QQQ_Hold'], results_df['QQQ_DD']),
        ("Total Cash Invested", results_df['Total_Principal'], None)
    ]

    for i, (name, series, dd_series) in enumerate(metrics):
        with cols[i]:
            final_val = series.iloc[-1]
            st.metric(label=name, value=f"${final_val:,.0f}")
            if dd_series is not None:
                # Calculate the true XIRR dynamically for this specific strategy
                xirr_pct = calc_xirr(final_val, base_cash_flows)
                st.caption(f"XIRR: **{xirr_pct:.1f}%** |  Max DD: **{dd_series.min()*100:.1f}%**")
                
                # Calmar Ratio adapted to use XIRR instead of standard CAGR
                calmar = (xirr_pct / 100) / abs(dd_series.min()) if abs(dd_series.min()) > 0 else 0
                st.caption(f"Calmar Ratio: **{calmar:.2f}**")

    st.markdown("### Equity Curve & Drawdowns")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    ax1.fill_between(results_df.index, 0, 1, where=(results_df['Risk_On'] == False), color='red', alpha=0.1, transform=ax1.get_xaxis_transform())
    ax2.fill_between(results_df.index, 0, 1, where=(results_df['Risk_On'] == False), color='red', alpha=0.1, transform=ax2.get_xaxis_transform())

    ax1.plot(results_df.index, results_df['Strategy_Value'], label='Mag 7 Top 3 Strategy', color='#185FA5', linewidth=2)
    ax1.plot(results_df.index, results_df['Mag7_Hold'], label='Mag 7 Equal Weight', color='#D9534F', linewidth=1.5)
    ax1.plot(results_df.index, results_df['QQQ_Hold'], label='Nasdaq 100', color='#1D9E75', alpha=0.5)
    ax1.plot(results_df.index, results_df['Total_Principal'], label='Cash Out of Pocket', color='black', linestyle=':', alpha=0.5)
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.25)
    ax1.set_yscale('log')

    ax2.fill_between(results_df.index, results_df['Strategy_DD'] * 100, 0, color='#185FA5', alpha=0.3)
    ax2.plot(results_df.index, results_df['Mag7_DD'] * 100, color='#D9534F', linewidth=1.5)
    ax2.plot(results_df.index, results_df['QQQ_DD'] * 100, color='#1D9E75', linewidth=1, alpha=0.5)
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### Execution Ledger")
    log_df = pd.DataFrame(trade_log)
    csv_cols = ['Date', 'Event', 'Strategy Action', 'Realized PnL', 'Active Holdings', 'Total Cash', 'Strategy Value', 'Mag7 Benchmark', 'QQQ Benchmark', 'SPY Benchmark']
    log_df = log_df[csv_cols]
    st.dataframe(log_df, use_container_width=True)

    csv = log_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Execution Log (CSV)", data=csv, file_name='Mag7_Execution_Log.csv', mime='text/csv')
else:
    st.info("👈 Set your parameters in the sidebar and click **Run Simulation** to generate the dashboard.")
