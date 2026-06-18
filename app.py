"""
공정능력분석 + 통계적공정관리(SPC) 웹앱
데이터를 업로드/수정하면 자동으로 재계산되는 대시보드
"""
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from spc_core import (
    make_subgroups, xbar_r_chart, xbar_s_chart, imr_chart,
    p_chart, np_chart, nelson_rules, RULE_DESC,
    capability_indices, capability_status, normality_test,
    rolling_capability, generate_diagnosis,
)

st.set_page_config(page_title="공정능력분석 + SPC 대시보드", layout="wide", page_icon="📈")

# ---------------------------------------------------------------
# 커스텀 스타일 (기본 Streamlit 테마보다 카드형 UI로 개선)
# ---------------------------------------------------------------
st.markdown("""
<style>
div[data-testid="stMetric"] {
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 12px 14px 6px 14px;
}
div[data-testid="stMetricValue"] { font-size: 1.6rem; }
.status-banner {
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 14px;
    font-size: 1.05rem;
    font-weight: 600;
}
.diagnosis-box {
    background-color: #F1F5F9;
    border-left: 5px solid #2563EB;
    border-radius: 8px;
    padding: 14px 18px;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# 샘플 데이터 생성 (반도체 박막 증착 공정을 모사)
# ---------------------------------------------------------------
def generate_sample_data():
    rng = np.random.default_rng(42)
    rows = []
    for h in range(1, 25):
        mean = 100.0
        if 11 <= h <= 20:
            mean += (h - 10) * 0.3       # 점진적 drift
        if h == 9:
            mean += 6                     # 이상점
        if h == 21:
            mean -= 6                     # 이상점
        vals = rng.normal(mean, 1.0, 5)
        for v in vals:
            rows.append({"Hour": h, "Thickness": round(float(v), 3)})
    return pd.DataFrame(rows)


def generate_sample_attribute_data():
    rng = np.random.default_rng(7)
    rows = []
    for lot in range(1, 21):
        n = 200
        p = 0.03 if lot not in (15,) else 0.10
        defects = rng.binomial(n, p)
        rows.append({"Lot": lot, "Defects": defects, "SampleSize": n})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# 사이드바 - 데이터 입력
# ---------------------------------------------------------------
st.sidebar.header("⚙️ 데이터 설정")
data_source = st.sidebar.radio("계량형(연속) 데이터 소스", ["샘플 데이터 사용", "CSV 업로드"])

if data_source == "샘플 데이터 사용":
    df = generate_sample_data()
else:
    up = st.sidebar.file_uploader("CSV 파일 업로드", type=["csv"], key="main_csv")
    if up is None:
        st.sidebar.info("CSV를 업로드하면 분석이 시작됩니다. 그 전까지는 샘플 데이터로 미리보기를 표시합니다.")
        df = generate_sample_data()
    else:
        df = pd.read_csv(up)

st.sidebar.markdown("---")
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
all_cols = list(df.columns)

value_col = st.sidebar.selectbox("측정값(품질특성) 컬럼", numeric_cols,
                                  index=len(numeric_cols)-1 if numeric_cols else 0)

group_mode_label = st.sidebar.selectbox(
    "서브그룹 구성 방식",
    ["컬럼으로 지정 (예: 시간/Lot)", "고정 크기로 묶기", "개별값 (I-MR 차트)"],
)
group_mode_map = {
    "컬럼으로 지정 (예: 시간/Lot)": "column",
    "고정 크기로 묶기": "fixed",
    "개별값 (I-MR 차트)": "individual",
}
group_mode = group_mode_map[group_mode_label]

group_col, fixed_size = None, 5
if group_mode == "column":
    default_idx = all_cols.index("Hour") if "Hour" in all_cols else 0
    group_col = st.sidebar.selectbox("서브그룹 ID 컬럼", all_cols, index=default_idx)
elif group_mode == "fixed":
    fixed_size = st.sidebar.number_input("서브그룹 크기", min_value=2, max_value=15, value=5)

st.sidebar.markdown("---")
st.sidebar.subheader("규격 한계 (Spec Limits)")
default_mean = float(df[value_col].mean()) if value_col else 100.0
default_std = float(df[value_col].std()) if value_col else 1.0
usl = st.sidebar.number_input("USL (상한)", value=round(default_mean + 4*default_std, 2))
lsl = st.sidebar.number_input("LSL (하한)", value=round(default_mean - 4*default_std, 2))
target = st.sidebar.number_input("Target (목표값, 선택)", value=round(default_mean, 2))

st.sidebar.markdown("---")
with st.sidebar.expander("📋 계수형(P/NP) 데이터 — 선택"):
    use_attr = st.checkbox("계수형 관리도 사용", value=False)
    if use_attr:
        attr_up = st.file_uploader("계수형 CSV 업로드 (불량수, 표본크기)", type=["csv"], key="attr_csv")
        if attr_up is None:
            attr_df = generate_sample_attribute_data()
            st.caption("샘플 계수형 데이터를 사용 중입니다.")
        else:
            attr_df = pd.read_csv(attr_up)
        attr_cols = list(attr_df.columns)
        defect_col = st.selectbox("불량(결점) 수 컬럼", attr_cols,
                                   index=attr_cols.index("Defects") if "Defects" in attr_cols else 0)
        size_mode = st.radio("표본 크기", ["컬럼으로 지정 (P chart)", "고정값 (NP chart)"])
        if size_mode == "컬럼으로 지정 (P chart)":
            size_col = st.selectbox("표본 크기 컬럼", attr_cols,
                                     index=attr_cols.index("SampleSize") if "SampleSize" in attr_cols else 0)
        else:
            fixed_n = st.number_input("고정 표본 크기 n", min_value=1, value=200)

st.sidebar.markdown("---")
phase1_fix = st.sidebar.checkbox("🛠 Phase I 수정 (Rule 1 위반점 제외 후 재계산)", value=False)

# ---------------------------------------------------------------
# 계산
# ---------------------------------------------------------------
long_df, n_sub = make_subgroups(df, value_col, group_mode, group_col, fixed_size)

if n_sub == 1:
    chart_res = imr_chart(long_df, value_col)
    main_series, center_key, ucl_key, lcl_key = chart_res["x"], "center_x", "ucl_x", "lcl_x"
elif n_sub <= 8:
    chart_res = xbar_r_chart(long_df, value_col)
    main_series, center_key, ucl_key, lcl_key = chart_res["x"], "center_x", "ucl_x", "lcl_x"
else:
    chart_res = xbar_s_chart(long_df, value_col)
    main_series, center_key, ucl_key, lcl_key = chart_res["x"], "center_x", "ucl_x", "lcl_x"

sigma_main = (chart_res[ucl_key] - chart_res[center_key]) / 3
violations = nelson_rules(main_series, chart_res[center_key], sigma_main)
violated_points = sorted(set(i for v in violations.values() for i in v))

# Phase I 수정: Rule1 위반 서브그룹 제외 후 재계산
chart_res_p2 = chart_res
cap_before = capability_indices(df[value_col], usl, lsl, chart_res["sigma_within"])
if phase1_fix and violations.get(1):
    keep_groups = [g for g in main_series.index if g not in violations[1]]
    long_df_p2 = long_df[long_df["__subgroup__"].isin(keep_groups)]
    if n_sub == 1:
        chart_res_p2 = imr_chart(long_df_p2, value_col)
    elif n_sub <= 8:
        chart_res_p2 = xbar_r_chart(long_df_p2, value_col)
    else:
        chart_res_p2 = xbar_s_chart(long_df_p2, value_col)
    cap_after = capability_indices(long_df_p2[value_col], usl, lsl, chart_res_p2["sigma_within"])
else:
    cap_after = cap_before

cap = cap_after if phase1_fix else cap_before
status_label, status_color = capability_status(cap["cpk"])

# ---------------------------------------------------------------
# 헤더
# ---------------------------------------------------------------
st.title("📈 공정능력분석 + 통계적공정관리(SPC) 대시보드")
st.caption("데이터를 바꾸면 모든 분석(관리도·공정능력·Nelson's Rules)이 자동으로 다시 계산됩니다.")

banner_bg = {"green": "#DCFCE7", "orange": "#FEF9C3", "red": "#FEE2E2", "gray": "#E2E8F0"}[status_color]
banner_fg = {"green": "#15803D", "orange": "#A16207", "red": "#B91C1C", "gray": "#475569"}[status_color]
banner_icon = {"green": "🟢", "orange": "🟡", "red": "🔴", "gray": "⚪"}[status_color]
st.markdown(
    f'<div class="status-banner" style="background-color:{banner_bg};color:{banner_fg};">'
    f'{banner_icon} 종합 공정 판정: {status_label}  (Cpk = {cap["cpk"]:.2f})</div>',
    unsafe_allow_html=True,
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Cp", f"{cap['cp']:.2f}")
k2.metric("Cpk", f"{cap['cpk']:.2f}")
k3.metric("Pp", f"{cap['pp']:.2f}")
k4.metric("Ppk", f"{cap['ppk']:.2f}")
k5.metric("Nelson's Rules 위반", f"{len(violated_points)}건")

tabs = st.tabs(["📊 데이터 개요", "🎯 공정능력분석", "📈 관리도(계량형)",
                 "🔢 관리도(계수형)", "🔍 Nelson's Rules", "🛠 Phase I 수정",
                 "📐 능력 추이", "🧾 종합 진단 리포트"])

# ---------------- Tab 1: 데이터 개요 ----------------
with tabs[0]:
    st.subheader("데이터 미리보기")
    st.dataframe(df, width='stretch', height=300)
    c1, c2 = st.columns(2)
    with c1:
        st.write("기술통계")
        st.dataframe(df[value_col].describe().to_frame())
    with c2:
        fig = go.Figure(go.Histogram(x=df[value_col], nbinsx=30, marker_color="#4C78A8"))
        fig.update_layout(title=f"{value_col} 분포", height=350)
        st.plotly_chart(fig, width='stretch')

# ---------------- Tab 2: 공정능력분석 ----------------
with tabs[1]:
    st.subheader("공정능력 지수")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cp (잠재능력)", f"{cap['cp']:.3f}")
    c2.metric("Cpk (실제능력)", f"{cap['cpk']:.3f}")
    c3.metric("Pp (전체변동 기준)", f"{cap['pp']:.3f}")
    c4.metric("Ppk (전체변동 기준)", f"{cap['ppk']:.3f}")
    st.info(f"σ(within)={cap['sigma_within']:.4f}, σ(overall)={cap['sigma_overall']:.4f}, "
            f"평균={cap['mu']:.3f} → 상태: **{status_label}**")

    x = np.linspace(df[value_col].min()-3*cap['sigma_overall'], df[value_col].max()+3*cap['sigma_overall'], 300)
    from scipy.stats import norm
    y = norm.pdf(x, cap['mu'], cap['sigma_overall'])
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df[value_col], histnorm="probability density",
                                nbinsx=30, name="데이터", marker_color="#A8C8E8"))
    fig.add_trace(go.Scatter(x=x, y=y, name="정규분포 적합", line=dict(color="#1f77b4")))
    fig.add_vline(x=usl, line_dash="dash", line_color="red", annotation_text="USL")
    fig.add_vline(x=lsl, line_dash="dash", line_color="red", annotation_text="LSL")
    fig.add_vline(x=target, line_dash="dot", line_color="green", annotation_text="Target")
    fig.update_layout(title="공정능력 히스토그램 (규격한계 포함)", height=420)
    st.plotly_chart(fig, width='stretch')

    nt = normality_test(df[value_col])
    st.write(f"**정규성 검정 (Shapiro-Wilk)**: statistic={nt['statistic']:.4f}, "
             f"p-value={nt['p_value']:.4g} → "
             f"{'정규분포를 따른다고 볼 수 있음 (p>0.05)' if nt['is_normal'] else '정규분포 가정에 위배될 수 있음 (p≤0.05)'}")

# ---------------- Tab 3: 계량형 관리도 ----------------
with tabs[2]:
    chart_name = {"xbar_r": "Xbar-R 관리도", "xbar_s": "Xbar-S 관리도", "imr": "I-MR 관리도"}[chart_res["kind"]]
    st.subheader(chart_name + f"  (서브그룹 크기 n={chart_res['n']})")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                         subplot_titles=("중심/평균 차트", "산포(R/S/MR) 차트"))
    idx = list(main_series.index)
    colors = ["red" if i in violated_points else "#1f77b4" for i in idx]
    fig.add_trace(go.Scatter(x=idx, y=main_series.values, mode="lines+markers",
                              marker=dict(color=colors, size=8), name="값"), row=1, col=1)
    fig.add_hline(y=chart_res[center_key], line_color="green", row=1, col=1)
    fig.add_hline(y=chart_res[ucl_key], line_color="red", line_dash="dash", row=1, col=1)
    fig.add_hline(y=chart_res[lcl_key], line_color="red", line_dash="dash", row=1, col=1)

    if chart_res["kind"] == "xbar_r":
        spread, c_key2, u_key2, l_key2 = chart_res["r"], "center_r", "ucl_r", "lcl_r"
    elif chart_res["kind"] == "xbar_s":
        spread, c_key2, u_key2, l_key2 = chart_res["s"], "center_s", "ucl_s", "lcl_s"
    else:
        spread, c_key2, u_key2, l_key2 = chart_res["mr"], "center_mr", "ucl_mr", "lcl_mr"

    fig.add_trace(go.Scatter(x=list(spread.index), y=spread.values, mode="lines+markers",
                              line=dict(color="#888")), row=2, col=1)
    fig.add_hline(y=chart_res[c_key2], line_color="green", row=2, col=1)
    fig.add_hline(y=chart_res[u_key2], line_color="red", line_dash="dash", row=2, col=1)
    fig.add_hline(y=chart_res[l_key2], line_color="red", line_dash="dash", row=2, col=1)
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, width='stretch')

    if violated_points:
        st.warning(f"⚠️ Nelson's Rules 위반 서브그룹: {violated_points}")
    else:
        st.success("✅ 검출된 Nelson's Rules 위반이 없습니다. 공정이 통계적으로 안정 상태입니다.")

# ---------------- Tab 4: 계수형 관리도 ----------------
with tabs[3]:
    if not use_attr:
        st.info("좌측 사이드바에서 '계수형 관리도 사용'을 체크하면 P/NP 관리도를 볼 수 있습니다.")
    else:
        if size_mode == "컬럼으로 지정 (P chart)":
            res_a = p_chart(attr_df[defect_col], attr_df[size_col])
            title, yvals, center, ucl, lcl = "P 관리도 (불량률)", res_a["p"], res_a["center"], res_a["ucl"], res_a["lcl"]
        else:
            res_a = np_chart(attr_df[defect_col], fixed_n)
            title, yvals, center, ucl, lcl = "NP 관리도 (불량개수)", res_a["np_"], res_a["center"], res_a["ucl"], res_a["lcl"]

        st.subheader(title)
        fig = go.Figure()
        xidx = list(range(len(yvals)))
        out_of_control = [(i, v) for i, v in zip(xidx, yvals.values if hasattr(yvals, "values") else yvals)
                           if (isinstance(ucl, pd.Series) and (v > ucl.iloc[i] or v < lcl.iloc[i]))
                           or (not isinstance(ucl, pd.Series) and (v > ucl or v < lcl))]
        fig.add_trace(go.Scatter(x=xidx, y=yvals.values if hasattr(yvals, "values") else yvals,
                                  mode="lines+markers", name=title))
        fig.add_hline(y=center, line_color="green")
        if isinstance(ucl, pd.Series):
            fig.add_trace(go.Scatter(x=xidx, y=ucl.values, line=dict(color="red", dash="dash"), name="UCL"))
            fig.add_trace(go.Scatter(x=xidx, y=lcl.values, line=dict(color="red", dash="dash"), name="LCL"))
        else:
            fig.add_hline(y=ucl, line_color="red", line_dash="dash")
            fig.add_hline(y=lcl, line_color="red", line_dash="dash")
        if out_of_control:
            ox, oy = zip(*out_of_control)
            fig.add_trace(go.Scatter(x=list(ox), y=list(oy), mode="markers",
                                      marker=dict(color="red", size=12, symbol="x"), name="관리이탈"))
        fig.update_layout(height=450)
        st.plotly_chart(fig, width='stretch')
        if out_of_control:
            st.warning(f"⚠️ 관리한계 이탈 지점 {len(out_of_control)}건 발견")
        else:
            st.success("✅ 모든 점이 관리한계 내에 있습니다.")

# ---------------- Tab 5: Nelson's Rules ----------------
with tabs[4]:
    st.subheader("Nelson's Rules 위반 분석 (Xbar/I 차트 기준)")
    if not violations:
        st.success("✅ 8가지 Nelson's Rules 모두 위반 없음 — 공정이 통계적 안정 상태입니다.")
    else:
        rows = []
        for rule, points in sorted(violations.items()):
            rows.append({"Rule": rule, "설명": RULE_DESC[rule], "위반 서브그룹": ", ".join(map(str, points)),
                         "건수": len(points)})
        st.dataframe(pd.DataFrame(rows), width='stretch')
    st.caption("Rule 1(3σ 초과)은 가장 심각한 특별원인 신호로, Phase I 수정 시 제외 대상이 됩니다.")

# ---------------- Tab 6: Phase I 수정 ----------------
with tabs[5]:
    st.subheader("Phase I 관리도 수정 (특별원인 제거 후 재계산)")
    st.write("좌측 사이드바의 'Phase I 수정' 체크박스를 켜면 Rule 1 위반 서브그룹을 제외하고 "
             "관리한계 및 공정능력을 다시 계산합니다.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**수정 전 (Phase 0)**")
        st.write(f"Cpk = {cap_before['cpk']:.3f} / Ppk = {cap_before['ppk']:.3f}")
        st.write(f"σ(within) = {cap_before['sigma_within']:.4f}")
    with c2:
        st.markdown("**수정 후 (Phase I, Rule1 제외)**" if phase1_fix else "**수정 후 (체크박스를 켜세요)**")
        if phase1_fix:
            st.write(f"Cpk = {cap_after['cpk']:.3f} / Ppk = {cap_after['ppk']:.3f}")
            st.write(f"σ(within) = {cap_after['sigma_within']:.4f}")
            delta = cap_after['cpk'] - cap_before['cpk']
            st.metric("Cpk 개선", f"{cap_after['cpk']:.3f}", delta=f"{delta:+.3f}")

# ---------------- Tab 7: [차별화] 공정능력 추이(트렌드) ----------------
with tabs[6]:
    st.subheader("📐 공정능력 추이(트렌드) 분석")
    st.caption("전체 구간을 시간순으로 나누어 구간별 Cp/Cpk를 추적합니다. 공정능력이 개선되는지, 악화되는지 한눈에 확인할 수 있습니다.")

    max_window = max(2, (long_df["__subgroup__"].nunique()) // 2)
    window_groups = st.slider("추이 분석 구간 크기 (서브그룹 개수)", min_value=2,
                               max_value=max(2, max_window), value=min(4, max(2, max_window)))
    trend_df = rolling_capability(long_df, value_col, chart_res["kind"], usl, lsl, window_groups)

    if len(trend_df) < 2:
        st.info("구간 수가 너무 적어 추이를 표시할 수 없습니다. 구간 크기를 줄여보세요.")
    else:
        fig = go.Figure()
        fig.add_hrect(y0=0, y1=1.0, fillcolor="#FEE2E2", opacity=0.5, line_width=0)
        fig.add_hrect(y0=1.0, y1=1.33, fillcolor="#FEF9C3", opacity=0.5, line_width=0)
        fig.add_hrect(y0=1.33, y1=max(2.0, trend_df["Cpk"].max()+0.2), fillcolor="#DCFCE7", opacity=0.5, line_width=0)
        fig.add_trace(go.Scatter(x=trend_df["구간"], y=trend_df["Cpk"], mode="lines+markers",
                                  name="Cpk", line=dict(color="#2563EB", width=3), marker=dict(size=9)))
        fig.add_trace(go.Scatter(x=trend_df["구간"], y=trend_df["Cp"], mode="lines+markers",
                                  name="Cp", line=dict(color="#94A3B8", dash="dot")))
        fig.update_layout(title="구간별 Cp / Cpk 추이 (배경: 빨강<1.0 / 노랑 1.0~1.33 / 초록≥1.33)",
                           height=450, yaxis_title="지수 값")
        st.plotly_chart(fig, width='stretch')
        st.dataframe(trend_df, width='stretch')

        slope = trend_df["Cpk"].iloc[-1] - trend_df["Cpk"].iloc[0]
        if slope > 0.1:
            st.success(f"📈 Cpk가 구간 초반({trend_df['Cpk'].iloc[0]:.2f}) 대비 후반({trend_df['Cpk'].iloc[-1]:.2f})에 개선되는 추세입니다.")
        elif slope < -0.1:
            st.warning(f"📉 Cpk가 구간 초반({trend_df['Cpk'].iloc[0]:.2f}) 대비 후반({trend_df['Cpk'].iloc[-1]:.2f})에 악화되는 추세입니다.")
        else:
            st.info("Cpk가 구간별로 큰 변화 없이 유지되고 있습니다.")

# ---------------- Tab 8: [차별화] 종합 진단 리포트 ----------------
with tabs[7]:
    st.subheader("🧾 종합 진단 리포트")

    c1, c2 = st.columns([1, 1.4])
    with c1:
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(cap["cpk"]) if not np.isnan(cap["cpk"]) else 0,
            number={"valueformat": ".2f"},
            title={"text": "Cpk"},
            gauge={
                "axis": {"range": [0, max(2.0, cap["cpk"]*1.2 if not np.isnan(cap["cpk"]) else 2.0)]},
                "bar": {"color": banner_fg},
                "steps": [
                    {"range": [0, 1.0], "color": "#FEE2E2"},
                    {"range": [1.0, 1.33], "color": "#FEF9C3"},
                    {"range": [1.33, max(2.0, cap["cpk"]*1.2 if not np.isnan(cap["cpk"]) else 2.0)], "color": "#DCFCE7"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.8, "value": 1.33},
            },
        ))
        gauge_fig.update_layout(height=320, margin=dict(t=40, b=10, l=20, r=20))
        st.plotly_chart(gauge_fig, width='stretch')

    with c2:
        st.markdown("**🩺 자동 진단 코멘트**")
        diagnosis = generate_diagnosis(cap, status_label, violations, len(main_series))
        st.markdown(f'<div class="diagnosis-box">{diagnosis}</div>', unsafe_allow_html=True)
        st.write("")
        m1, m2, m3 = st.columns(3)
        m1.metric("Nelson's Rules 위반", f"{len(violated_points)}건")
        m2.metric("정규성 p-value", f"{normality_test(df[value_col])['p_value']:.4g}")
        m3.metric("Phase I 적용", "예" if phase1_fix else "아니오")

    st.markdown("---")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(main_series.index), y=main_series.values,
                              mode="lines+markers",
                              marker=dict(color=["red" if i in violated_points else "#1f77b4"
                                                  for i in main_series.index])))
    fig.add_hrect(y0=lsl, y1=usl, fillcolor="green", opacity=0.08, line_width=0)
    fig.add_hline(y=usl, line_color="red", line_dash="dot")
    fig.add_hline(y=lsl, line_color="red", line_dash="dot")
    fig.update_layout(title="공정 추이 vs 규격한계", height=380)
    st.plotly_chart(fig, width='stretch')

    st.markdown("### 📥 분석 리포트 다운로드")
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="원본데이터", index=False)
        pd.DataFrame([{
            "Cp": cap["cp"], "Cpk": cap["cpk"], "Pp": cap["pp"], "Ppk": cap["ppk"],
            "평균": cap["mu"], "sigma_within": cap["sigma_within"], "sigma_overall": cap["sigma_overall"],
            "USL": usl, "LSL": lsl, "공정상태": status_label,
        }]).to_excel(writer, sheet_name="공정능력지수", index=False)
        if violations:
            vrows = [{"Rule": k, "설명": RULE_DESC[k], "위반지점": ", ".join(map(str, v)), "건수": len(v)}
                     for k, v in sorted(violations.items())]
            pd.DataFrame(vrows).to_excel(writer, sheet_name="NelsonRules위반", index=False)
        pd.DataFrame([{"진단코멘트": diagnosis}]).to_excel(writer, sheet_name="자동진단", index=False)
        if len(trend_df) >= 2:
            trend_df.to_excel(writer, sheet_name="능력추이", index=False)
    excel_buf.seek(0)
    st.download_button("📊 Excel 리포트 다운로드 (.xlsx)", data=excel_buf,
                        file_name="SPC_공정능력분석_리포트.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.caption("이 리포트는 데이터(CSV)를 교체하거나 사이드바 설정을 바꿀 때마다 전체 계산이 자동으로 갱신됩니다.")
