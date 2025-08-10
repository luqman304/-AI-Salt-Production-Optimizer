import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

# ---------- Streamlit page config ----------
st.set_page_config(page_title="AI-Salt-Production-Optimizer", page_icon=":factory:", layout="wide")

# ---------- Compatibility helpers ----------
def show_image(img, caption=None):
    """Display image across Streamlit versions (use_container_width vs use_column_width)."""
    try:
        st.image(img, caption=caption, use_container_width=True)
    except TypeError:
        st.image(img, caption=caption, use_column_width=True)

def card(title, body=""):
    st.markdown(
        """
<div style="padding:16px;border:1px solid #1f2937;border-radius:14px;background:#111827">
  <div style="font-weight:600;margin-bottom:6px">""" + title + """</div>
  <div>""" + body + """</div>
</div>
""",
        unsafe_allow_html=True,
    )

# ---------- Header ----------
st.markdown("# AI-Salt-Production-Optimizer")
st.caption("Core Four: Predictive Maintenance • Energy • CV Quality • Logistics • ROI")

# ---------- Sidebar ----------
with st.sidebar:
    if os.path.exists("process_flow.png"):
        show_image("process_flow.png", caption="Core Four pipeline")

    st.markdown("""
**NASCON FY2024 (public filings):**
- Revenue: ~NGN 120.4B
- Gross margin: ~46%
- EBITDA margin: ~23%

Sources: NGX audited results & FY2024 investor materials.
""")

    section = st.radio(
        "Navigate",
        [
            "Predictive Maintenance",
            "Energy Optimization",
            "Quality Control (CV)",
            "Logistics Optimization",
            "ROI Calculator",
        ],
    )
    use_bench = st.checkbox("Use NASCON FY2024 benchmarks in ROI tab", True)

# ======================================================================
# Predictive Maintenance
# ======================================================================
if section == "Predictive Maintenance":
    st.subheader("Predictive Maintenance — Failure Risk (Demo)")

    df = pd.read_csv("sample_data/maintenance.csv", parse_dates=["date"])
    c1, c2 = st.columns([1, 1])

    with c1:
        st.dataframe(df.tail(), use_container_width=True)

    with c2:
        fig, ax = plt.subplots()
        ax.plot(df["date"], df["temp_c"], label="Temp (C)")
        ax2 = ax.twinx()
        ax2.plot(df["date"], df["vibration_g"], color="gray", alpha=0.6, label="Vibration (g)")
        ax.set_title("Temperature & Vibration — last 120 days")
        st.pyplot(fig)

    X = df[["temp_c", "vibration_g", "run_hours"]]
    y = df["failure_next_7d"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    model = RandomForestClassifier(n_estimators=140, random_state=42).fit(Xtr, ytr)

    acc = accuracy_score(yte, model.predict(Xte))
    latest = df.iloc[-1][["temp_c", "vibration_g", "run_hours"]].values.reshape(1, -1)
    prob = model.predict_proba(latest)[0][1]

    m1, m2, m3 = st.columns(3)
    m1.metric("Holdout accuracy", f"{acc:.2f}")
    m2.metric("Latest failure risk (7d)", f"{prob*100:.1f}%")
    m3.metric("Last run hours", f"{df.iloc[-1]['run_hours']:.0f}h")

    risk_series = ((df["temp_c"] - 70) / 10 + (df["vibration_g"] - 0.9) / 0.3).rolling(7).mean()
    fig2, ax = plt.subplots()
    ax.plot(df["date"], (1 / (1 + np.exp(-risk_series))).fillna(0), label="Risk trend")
    ax.set_ylim(0, 1)
    ax.set_title("Failure risk trend (7-day) — demo")
    st.pyplot(fig2)

    with st.expander("How it works"):
        st.write("RandomForest on synthetic features (temperature, vibration, run hours) predicts 7-day failure risk. Replace with historian/PLC data in production.")

# ======================================================================
# Energy Optimization
# ======================================================================
elif section == "Energy Optimization":
    st.subheader("Energy Optimization — Usage & Baseline Forecast")

    edf = pd.read_csv("sample_data/energy.csv", parse_dates=["date"])
    c1, c2 = st.columns([1, 1])

    with c1:
        st.dataframe(edf.tail(), use_container_width=True)

    with c2:
        fig, ax = plt.subplots()
        ax.plot(edf["date"], edf["energy_mwh"])
        ax.set_title("Daily energy (365 days)")
        st.pyplot(fig)

    window = st.slider("Moving average window (days)", 7, 30, 14)
    ma = pd.Series(edf["energy_mwh"]).rolling(window=window).mean()

    fig2, ax = plt.subplots()
    ax.plot(edf["date"], edf["energy_mwh"], label="Actual")
    ax.plot(edf["date"], ma, label=f"{window}-day MA")
    ax.fill_between(edf["date"], ma * 0.97, ma * 1.03, alpha=0.2, label="+/-3%")
    ax.legend()
    ax.set_title("Baseline forecast band")
    st.pyplot(fig2)

    with st.expander("How it works"):
        st.write("Lightweight moving average forecast as a baseline. Replace with Prophet/SARIMAX/LSTM for production scheduling.")

# ======================================================================
# Quality Control (CV)
# ======================================================================
elif section == "Quality Control (CV)":
    st.subheader("Computer Vision QC — Heuristic Defect Score (Demo)")
    st.caption("Upload an image or use the samples below. Heuristic uses dark specks + yellow tint.")

    up = st.file_uploader("Upload image (jpg/png)", type=["jpg", "jpeg", "png"])

    if up is None:
        cols = st.columns(4)
        samples = [
            "sample_data/quality/good1.jpg",
            "sample_data/quality/good2.jpg",
            "sample_data/quality/defect1.jpg",
            "sample_data/quality/defect2.jpg",
        ]
        for i, p in enumerate(samples):
            with cols[i]:
                if os.path.exists(p):
                    show_image(p, caption=os.path.basename(p))
    else:
        img = Image.open(up).convert("RGB")
        arr = np.asarray(img).astype(np.float32)
        dark = (arr.mean(axis=2) < 120).mean()
        yellow = ((arr[:, :, 0] + arr[:, :, 1]) / 2 - arr[:, :, 2] > 25).mean()
        score = 0.6 * dark + 0.4 * yellow
        label = "DEFECT" if score > 0.08 else "GOOD"
        show_image(img, caption=f"{label} (score={score:.3f})")

    fig, ax = plt.subplots()
    ax.bar(["GOOD", "DEFECT"], [2, 2])
    ax.set_title("Recent QC outcomes (sample)")
    st.pyplot(fig)

    with st.expander("How it works"):
        st.write("This demo uses a simple heuristic. Swap for a small CNN/YOLO model trained on labeled images for real QC.")

# ======================================================================
# Logistics Optimization
# ======================================================================
elif section == "Logistics Optimization":
    st.subheader("Logistics — Simple Packing Heuristic (Demo)")

    ldf = pd.read_csv("sample_data/logistics.csv")
    st.dataframe(ldf.head(), use_container_width=True)

    trucks = st.slider("Available trucks", 3, 12, 6)

    ldf_sorted = ldf.sort_values("distance_km", ascending=False).reset_index(drop=True)
    routes = [[] for _ in range(trucks)]
    loads = [0.0] * trucks

    for _, r in ldf_sorted.iterrows():
        idx = int(np.argmin(loads))
        if loads[idx] + r["weight_tons"] > 40:
            placed = False
            for j in range(trucks):
                if loads[j] + r["weight_tons"] <= 40:
                    routes[j].append(r["order_id"])
                    loads[j] += r["weight_tons"]
                    placed = True
                    break
            if not placed:
                routes[idx].append(r["order_id"])
                loads[idx] += r["weight_tons"]
        else:
            routes[idx].append(r["order_id"])
            loads[idx] += r["weight_tons"]

    c1, c2 = st.columns([1, 1])
    with c1:
        for i, (rt, ld) in enumerate(zip(routes, loads), start=1):
            card(f"Truck {i}", f"Load: <b>{ld:.1f} tons</b> • Orders: {', '.join(rt) if rt else '-'}")
    with c2:
        fig, ax = plt.subplots()
        ax.scatter(ldf["distance_km"], ldf["cost_ngn"])
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Cost (NGN)")
        ax.set_title("Distance vs Cost")
        st.pyplot(fig)

    with st.expander("How it works"):
        st.write("Greedy packing by available capacity. Replace with Google OR-Tools VRP for optimal routing in production.")

# ======================================================================
# ROI Calculator
# ======================================================================
else:
    st.subheader("ROI Calculator — Core Four Savings (Demo)")
    st.caption("Toggle NASCON FY2024 benchmarks in the sidebar.")

    colA, colB = st.columns(2)

    if use_bench:
        rev = colA.number_input("Annual Revenue (NGN)", value=120.4e9, step=1e9, format="%.0f")
        gm = colA.number_input("Gross Margin (%)", value=46.0, step=0.5)
        ebitda_m = colA.number_input("EBITDA Margin (%)", value=23.0, step=0.5)
    else:
        rev = colA.number_input("Annual Revenue (NGN)", value=50e9, step=1e9, format="%.0f")
        gm = colA.number_input("Gross Margin (%)", value=35.0, step=0.5)
        ebitda_m = colA.number_input("EBITDA Margin (%)", value=15.0, step=0.5)

    maint_cost = colA.number_input("Annual Maintenance Cost (NGN)", value=4e9, step=1e8, format="%.0f")
    energy_cost = colA.number_input("Annual Energy Cost (NGN)", value=10e9, step=1e8, format="%.0f")
    logistics_cost = colA.number_input("Annual Logistics Cost (NGN)", value=6e9, step=1e8, format="%.0f")
    waste_pct = colA.slider("Quality Waste (% of COGS)", 0.0, 10.0, 4.0, 0.1)

    colA.markdown("---")
    capex = colA.number_input("Program CAPEX (NGN)", value=1.2e9, step=1e8, format="%.0f")

    colB.write("Core Four Savings Ranges")
    dt_reduc = colB.slider("Downtime reduction (%)", 30, 50, 40)
    maint_reduc = colB.slider("Maintenance cost reduction (%)", 10, 25, 15)
    energy_reduc = colB.slider("Energy reduction (%)", 10, 15, 12)
    waste_reduc = colB.slider("Waste reduction (%)", 5, 8, 6)
    trans_reduc = colB.slider("Transport cost reduction (%)", 10, 20, 12)

    gross_profit = rev * (gm / 100.0)
    ebitda = rev * (ebitda_m / 100.0)
    cogs = rev - gross_profit
    waste_cost = cogs * (waste_pct / 100.0)

    downtime_savings = ebitda * (dt_reduc / 100.0) * 0.2   # demo assumption
    maint_savings = maint_cost * (maint_reduc / 100.0)
    energy_savings = energy_cost * (energy_reduc / 100.0)
    logistics_savings = logistics_cost * (trans_reduc / 100.0)
    waste_savings = waste_cost * (waste_reduc / 100.0)

    total_savings = downtime_savings + maint_savings + energy_savings + logistics_savings + waste_savings
    ebitda_uplift_pp = (total_savings / rev) * 100.0
    payback_months = (capex / (total_savings / 12.0)) if total_savings > 0 else float("inf")

    m1, m2, m3 = st.columns(3)
    m1.metric("Annual savings (NGN)", f"{total_savings:,.0f}")
    m2.metric("EBITDA uplift (pp)", f"{ebitda_uplift_pp:.2f} pp")
    m3.metric("Payback (months)", f"{payback_months:.1f}" if np.isfinite(payback_months) else "N/A")

    st.markdown("### Before vs After (Annual Costs)")
    labels = ["Maintenance", "Energy", "Logistics", "Quality Waste"]
    before = [maint_cost, energy_cost, logistics_cost, waste_cost]
    after = [
        maint_cost * (1 - maint_reduc / 100.0),
        energy_cost * (1 - energy_reduc / 100.0),
        logistics_cost * (1 - trans_reduc / 100.0),
        waste_cost * (1 - waste_reduc / 100.0),
    ]
    fig, ax = plt.subplots()
    x = np.arange(len(labels))
    ax.bar(x - 0.2, before, width=0.4, label="Before")
    ax.bar(x + 0.2, after, width=0.4, label="After")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("NGN (annual)")
    ax.legend()
    st.pyplot(fig)

    st.markdown("### Savings Breakdown")
    parts = [
        ("Downtime", downtime_savings),
        ("Maintenance", maint_savings),
        ("Energy", energy_savings),
        ("Logistics", logistics_savings),
        ("Waste", waste_savings),
    ]
    labels2, sizes2 = zip(*parts)
    fig2, ax2 = plt.subplots()
    ax2.pie(sizes2, labels=labels2, autopct="%1.1f%%", startangle=140)
    ax2.axis("equal")
    st.pyplot(fig2)

    st.markdown("---")
    if st.button("Export 1-page PDF"):
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        w, h = A4
        y = h - 2 * cm

        c.setFont("Helvetica-Bold", 16)
        c.drawString(2 * cm, y, "AI-Salt-Production-Optimizer — ROI Summary")
        y -= 1.2 * cm

        c.setFont("Helvetica", 10)
        c.drawString(2 * cm, y, f"Revenue: NGN {rev:,.0f} | GM: {gm:.1f}% | EBITDA: {ebitda_m:.1f}%")
        y -= 0.6 * cm
        c.drawString(2 * cm, y, f"Savings: NGN {total_savings:,.0f} | EBITDA uplift: {ebitda_uplift_pp:.2f} pp | Payback: {payback_months:.1f} mo")
        y -= 0.8 * cm

        c.setFont("Helvetica-Bold", 12)
        c.drawString(2 * cm, y, "Savings by Area")
        y -= 0.6 * cm
        for name, val in parts:
            c.drawString(2.2 * cm, y, f"- {name}: NGN {val:,.0f}")
            y -= 0.5 * cm

        y -= 0.4 * cm
        c.setFont("Helvetica-Oblique", 9)
        c.drawString(2 * cm, y, "Sources: NASCON FY2024 audited results & investor materials (public filings).")
        y -= 0.5 * cm
        c.drawString(2 * cm, y, "Demo assumptions; replace with plant data & validated finance inputs.")

        c.showPage()
        c.save()
        buf.seek(0)
        st.download_button("Download PDF", buf, file_name="ROI_Summary.pdf", mime="application/pdf")