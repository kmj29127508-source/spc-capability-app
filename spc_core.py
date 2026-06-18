"""
SPC(통계적공정관리) + 공정능력분석 핵심 계산 모듈
강의 내용 기준: Xbar-R/Xbar-S/I-MR 관리도, P/NP 관리도,
Nelson's Rules, Cp/Cpk/Pp/Ppk, Phase I 수정
"""
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------
# 관리도 상수표 (subgroup size n = 2~10)
# A2, D3, D4 : Xbar-R 차트용 / A3, B3, B4, c4 : Xbar-S 차트용 / d2 : 공정능력 sigma_within 계산용
# ---------------------------------------------------------------
CONTROL_CONSTANTS = {
    2:  dict(A2=1.880, D3=0.000, D4=3.267, A3=2.659, B3=0.000, B4=3.267, c4=0.7979, d2=1.128),
    3:  dict(A2=1.023, D3=0.000, D4=2.574, A3=1.954, B3=0.000, B4=2.568, c4=0.8862, d2=1.693),
    4:  dict(A2=0.729, D3=0.000, D4=2.282, A3=1.628, B3=0.000, B4=2.266, c4=0.9213, d2=2.059),
    5:  dict(A2=0.577, D3=0.000, D4=2.114, A3=1.427, B3=0.000, B4=2.089, c4=0.9400, d2=2.326),
    6:  dict(A2=0.483, D3=0.000, D4=2.004, A3=1.287, B3=0.030, B4=1.970, c4=0.9515, d2=2.534),
    7:  dict(A2=0.419, D3=0.076, D4=1.924, A3=1.182, B3=0.118, B4=1.882, c4=0.9594, d2=2.704),
    8:  dict(A2=0.373, D3=0.136, D4=1.864, A3=1.099, B3=0.185, B4=1.815, c4=0.9650, d2=2.847),
    9:  dict(A2=0.337, D3=0.184, D4=1.816, A3=1.032, B3=0.239, B4=1.761, c4=0.9693, d2=2.970),
    10: dict(A2=0.308, D3=0.223, D4=1.777, A3=0.975, B3=0.284, B4=1.716, c4=0.9727, d2=3.078),
}
D2_FOR_MR = 1.128   # n=2 (I-MR 차트의 moving range는 항상 두 점 차이)
D4_FOR_MR = 3.267


def make_subgroups(df: pd.DataFrame, value_col: str, group_mode: str,
                    group_col: str = None, fixed_size: int = 5):
    """원본 데이터프레임을 (subgroup_id, value) 형태의 long-format으로 정리.
    group_mode: 'individual' | 'column' | 'fixed'
    반환: long_df(원본+subgroup 컬럼), subgroup_size(고정 크기일 때 int, 가변이면 None)
    """
    out = df.copy()
    if group_mode == "individual":
        out["__subgroup__"] = range(len(out))
        return out, 1
    if group_mode == "column":
        out["__subgroup__"] = out[group_col]
        sizes = out.groupby("__subgroup__")[value_col].count()
        size = int(sizes.mode().iloc[0]) if len(sizes) else 1
        return out, size
    if group_mode == "fixed":
        out["__subgroup__"] = np.arange(len(out)) // fixed_size
        return out, fixed_size
    raise ValueError("unknown group_mode")


# ---------------------------------------------------------------
# 계량형 관리도
# ---------------------------------------------------------------
def xbar_r_chart(long_df: pd.DataFrame, value_col: str):
    g = long_df.groupby("__subgroup__")[value_col]
    xbar = g.mean()
    rng = g.apply(lambda s: s.max() - s.min())
    n = int(g.count().mode().iloc[0])
    n = min(max(n, 2), 10)
    c = CONTROL_CONSTANTS[n]
    xbarbar = xbar.mean()
    rbar = rng.mean()
    ucl_x, lcl_x = xbarbar + c["A2"] * rbar, xbarbar - c["A2"] * rbar
    ucl_r, lcl_r = c["D4"] * rbar, c["D3"] * rbar
    sigma_within = rbar / c["d2"]
    return dict(kind="xbar_r", x=xbar, r=rng, center_x=xbarbar, ucl_x=ucl_x, lcl_x=lcl_x,
                center_r=rbar, ucl_r=ucl_r, lcl_r=lcl_r, sigma_within=sigma_within, n=n)


def xbar_s_chart(long_df: pd.DataFrame, value_col: str):
    g = long_df.groupby("__subgroup__")[value_col]
    xbar = g.mean()
    s = g.std(ddof=1)
    n = int(g.count().mode().iloc[0])
    n = min(max(n, 2), 10)
    c = CONTROL_CONSTANTS[n]
    xbarbar = xbar.mean()
    sbar = s.mean()
    ucl_x, lcl_x = xbarbar + c["A3"] * sbar, xbarbar - c["A3"] * sbar
    ucl_s, lcl_s = c["B4"] * sbar, c["B3"] * sbar
    sigma_within = sbar / c["c4"]
    return dict(kind="xbar_s", x=xbar, s=s, center_x=xbarbar, ucl_x=ucl_x, lcl_x=lcl_x,
                center_s=sbar, ucl_s=ucl_s, lcl_s=lcl_s, sigma_within=sigma_within, n=n)


def imr_chart(long_df: pd.DataFrame, value_col: str):
    x = long_df.set_index("__subgroup__")[value_col]
    mr = x.diff().abs().dropna()
    xbar = x.mean()
    mrbar = mr.mean()
    ucl_x, lcl_x = xbar + 3 * mrbar / D2_FOR_MR, xbar - 3 * mrbar / D2_FOR_MR
    ucl_mr, lcl_mr = D4_FOR_MR * mrbar, 0.0
    sigma_within = mrbar / D2_FOR_MR
    return dict(kind="imr", x=x, mr=mr, center_x=xbar, ucl_x=ucl_x, lcl_x=lcl_x,
                center_mr=mrbar, ucl_mr=ucl_mr, lcl_mr=lcl_mr, sigma_within=sigma_within, n=1)


# ---------------------------------------------------------------
# 계수형 관리도 (P / NP)
# ---------------------------------------------------------------
def p_chart(defects: pd.Series, sample_sizes: pd.Series):
    p = defects / sample_sizes
    pbar = defects.sum() / sample_sizes.sum()
    ucl = pbar + 3 * np.sqrt(pbar * (1 - pbar) / sample_sizes)
    lcl = (pbar - 3 * np.sqrt(pbar * (1 - pbar) / sample_sizes)).clip(lower=0)
    return dict(kind="p", p=p, center=pbar, ucl=ucl, lcl=lcl, n=sample_sizes)


def np_chart(defects: pd.Series, n: float):
    pbar = defects.mean() / n
    npbar = defects.mean()
    ucl = npbar + 3 * np.sqrt(npbar * (1 - pbar))
    lcl = max(0.0, npbar - 3 * np.sqrt(npbar * (1 - pbar)))
    return dict(kind="np", np_=defects, center=npbar, ucl=ucl, lcl=lcl, n=n)


# ---------------------------------------------------------------
# Nelson's Rules (8 rules) - 시리즈(포인트)와 center/sigma만 있으면 적용 가능
# ---------------------------------------------------------------
def nelson_rules(series: pd.Series, center: float, sigma: float):
    """8가지 Nelson's Rules 적용. 반환: {rule_no: [violated_index, ...]}"""
    x = series.values.astype(float)
    idx = series.index
    n = len(x)
    z = (x - center) / sigma if sigma > 0 else np.zeros(n)
    viol = {i: [] for i in range(1, 9)}

    # Rule 1: 3-sigma 초과 1점
    for i in range(n):
        if abs(z[i]) > 3:
            viol[1].append(idx[i])

    # Rule 2: 같은 쪽에 9점 연속
    side = np.sign(z)
    for i in range(8, n):
        w = side[i-8:i+1]
        if np.all(w > 0) or np.all(w < 0):
            viol[2].append(idx[i])

    # Rule 3: 6점 연속 증가 또는 감소
    for i in range(5, n):
        w = x[i-5:i+1]
        if np.all(np.diff(w) > 0) or np.all(np.diff(w) < 0):
            viol[3].append(idx[i])

    # Rule 4: 14점 연속 교대(지그재그)
    for i in range(13, n):
        w = x[i-13:i+1]
        d = np.diff(w)
        if np.all(d[::2] > 0) and np.all(d[1::2] < 0):
            viol[4].append(idx[i])
        elif np.all(d[::2] < 0) and np.all(d[1::2] > 0):
            viol[4].append(idx[i])

    # Rule 5: 연속 3점 중 2점이 2-sigma 초과(같은 쪽)
    for i in range(2, n):
        w = z[i-2:i+1]
        pos = np.sum(w > 2)
        neg = np.sum(w < -2)
        if pos >= 2 or neg >= 2:
            viol[5].append(idx[i])

    # Rule 6: 연속 5점 중 4점이 1-sigma 초과(같은 쪽)
    for i in range(4, n):
        w = z[i-4:i+1]
        pos = np.sum(w > 1)
        neg = np.sum(w < -1)
        if pos >= 4 or neg >= 4:
            viol[6].append(idx[i])

    # Rule 7: 15점 연속이 +-1 sigma 이내 (계층화)
    for i in range(14, n):
        w = z[i-14:i+1]
        if np.all(np.abs(w) < 1):
            viol[7].append(idx[i])

    # Rule 8: 8점 연속이 +-1 sigma 밖, 양쪽 혼재 (혼합)
    for i in range(7, n):
        w = z[i-7:i+1]
        if np.all(np.abs(w) > 1):
            viol[8].append(idx[i])

    return {k: v for k, v in viol.items() if v}


RULE_DESC = {
    1: "한 점이 3-시그마 한계 밖",
    2: "9점 연속 중심선 한쪽",
    3: "6점 연속 증가/감소",
    4: "14점 연속 교대(지그재그)",
    5: "3점 중 2점이 2-시그마 초과",
    6: "5점 중 4점이 1-시그마 초과",
    7: "15점 연속 1-시그마 이내(계층화)",
    8: "8점 연속 1-시그마 밖(혼합)",
}


# ---------------------------------------------------------------
# 공정능력분석 Cp / Cpk / Pp / Ppk
# ---------------------------------------------------------------
def capability_indices(values: pd.Series, usl: float, lsl: float, sigma_within: float):
    mu = values.mean()
    sigma_overall = values.std(ddof=1)
    cp = (usl - lsl) / (6 * sigma_within) if sigma_within > 0 else np.nan
    cpu = (usl - mu) / (3 * sigma_within) if sigma_within > 0 else np.nan
    cpl = (mu - lsl) / (3 * sigma_within) if sigma_within > 0 else np.nan
    cpk = min(cpu, cpl)
    pp = (usl - lsl) / (6 * sigma_overall) if sigma_overall > 0 else np.nan
    ppu = (usl - mu) / (3 * sigma_overall) if sigma_overall > 0 else np.nan
    ppl = (mu - lsl) / (3 * sigma_overall) if sigma_overall > 0 else np.nan
    ppk = min(ppu, ppl)
    return dict(mu=mu, sigma_overall=sigma_overall, sigma_within=sigma_within,
                cp=cp, cpk=cpk, cpu=cpu, cpl=cpl, pp=pp, ppk=ppk, ppu=ppu, ppl=ppl)


def capability_status(cpk: float):
    if np.isnan(cpk):
        return "판단불가", "gray"
    if cpk >= 1.33:
        return "양호", "green"
    if cpk >= 1.00:
        return "주의", "orange"
    return "불량(개선 필요)", "red"


def normality_test(values: pd.Series):
    stat, p = stats.shapiro(values.sample(min(len(values), 5000), random_state=0))
    return dict(statistic=stat, p_value=p, is_normal=p > 0.05)


# ---------------------------------------------------------------
# [차별화 기능 1] 공정능력 추이(트렌드) 분석
# 전체 데이터를 시간순으로 여러 구간(window)으로 나눠 구간별 Cp/Cpk를 계산 →
# 공정능력이 시간에 따라 개선/악화되는지 추적
# ---------------------------------------------------------------
def rolling_capability(long_df: pd.DataFrame, value_col: str, kind: str,
                        usl: float, lsl: float, window_groups: int = 4):
    groups = sorted(long_df["__subgroup__"].unique())
    rows = []
    for start in range(0, len(groups), window_groups):
        block_groups = groups[start:start + window_groups]
        block_df = long_df[long_df["__subgroup__"].isin(block_groups)]
        if kind == "imr":
            if len(block_df) < 3:
                continue
            r = imr_chart(block_df, value_col)
        elif kind == "xbar_r":
            if block_df["__subgroup__"].nunique() < 2:
                continue
            r = xbar_r_chart(block_df, value_col)
        else:
            if block_df["__subgroup__"].nunique() < 2:
                continue
            r = xbar_s_chart(block_df, value_col)
        cap = capability_indices(block_df[value_col], usl, lsl, r["sigma_within"])
        rows.append({
            "구간": f"{block_groups[0]}~{block_groups[-1]}",
            "Cp": cap["cp"], "Cpk": cap["cpk"],
            "n_subgroups": len(block_groups),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# [차별화 기능 2] 룰 기반 자동 진단 코멘트 (한국어 자연어 리포트)
# 공정능력 지수 + Nelson's Rules 결과를 종합해 사람이 읽는 진단 문단을 자동 생성
# ---------------------------------------------------------------
def generate_diagnosis(cap: dict, status_label: str, violations: dict, n_subgroups: int):
    msgs = []
    msgs.append(f"현재 공정의 Cpk는 {cap['cpk']:.2f}로 '{status_label}' 수준으로 판정됩니다.")

    if cap["cpk"] >= 1.33:
        msgs.append("규격 한계에 대해 충분한 여유를 가지고 안정적으로 운영되고 있습니다.")
    elif cap["cpk"] >= 1.00:
        msgs.append("규격은 만족하고 있으나 여유가 크지 않아 변동 관리에 주의가 필요합니다.")
    else:
        msgs.append("규격 이탈 위험이 높은 상태로, 공정 산포 축소 또는 평균 조정이 필요합니다.")

    if not np.isnan(cap["cp"]) and not np.isnan(cap["cpk"]) and (cap["cp"] - cap["cpk"]) > 0.2:
        msgs.append("Cp 대비 Cpk가 상대적으로 낮아 공정 산포보다 평균(중심)이 목표에서 벗어난 것으로 보이며, "
                     "센터링(평균 조정)을 우선 검토할 필요가 있습니다.")

    if cap["pp"] is not None and not np.isnan(cap["pp"]) and (cap["pp"] - cap["cp"]) < -0.15:
        msgs.append("Pp(전체변동 기준)가 Cp(공정내변동 기준)보다 낮아, 서브그룹 간 변동(시간에 따른 흔들림)이 "
                     "전체 변동을 키우고 있을 가능성이 있습니다.")

    if violations:
        rule_no = sorted(violations.keys())
        desc = ", ".join(f"Rule {r}({RULE_DESC[r]})" for r in rule_no)
        n_pts = len(set(i for v in violations.values() for i in v))
        msgs.append(f"관리도에서 {desc} 위반이 검출되어({n_pts}개 지점), 특별원인에 의한 변동이 존재할 가능성이 있습니다. "
                     "해당 시점의 공정 조건(설비/원재료/작업자 등)을 확인하는 것을 권장합니다.")
    else:
        msgs.append("Nelson's Rules 8가지 위반이 검출되지 않아 공정이 통계적으로 안정 상태(in-control)로 판단됩니다.")

    return " ".join(msgs)
