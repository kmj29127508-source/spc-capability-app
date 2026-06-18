# 공정능력분석 + 통계적공정관리(SPC) 대시보드

강의(공정능력분석, 통계적공정관리) 내용을 기반으로 만든 Streamlit 웹앱입니다.
CSV 데이터를 업로드/교체하면 관리도, Nelson's Rules 위반 검출, 공정능력지수(Cp/Cpk/Pp/Ppk)가
자동으로 다시 계산됩니다.

## 주요 기능
- **데이터 개요**: 업로드 데이터 미리보기, 기술통계, 분포 히스토그램
- **공정능력분석**: Cp / Cpk / Pp / Ppk, 규격한계 포함 히스토그램, Shapiro-Wilk 정규성 검정
- **계량형 관리도**: 서브그룹 크기에 따라 Xbar-R / Xbar-S / I-MR 차트 자동 선택
- **계수형 관리도**: P chart(불량률, 표본크기 가변) / NP chart(불량개수, 표본크기 고정)
- **Nelson's Rules**: 8가지 규칙 전부 적용, 위반 지점과 사유 표로 표시
- **Phase I 수정**: Rule 1(3σ 초과) 위반 서브그룹 제외 후 관리한계·공정능력 재계산 (수정 전/후 비교)

## 차별화 기능 (다른 팀과 구분되는 포인트)
- **Cpk 게이지 차트**: 단순 숫자가 아닌 속도계 형태의 시각적 지표 (빨강/노랑/초록 구간 표시)
- **자동 진단 코멘트**: Cp/Cpk/Pp/Ppk와 Nelson's Rules 결과를 종합해 사람이 읽는 한국어 진단 문단을 룰 기반으로 자동 생성
- **공정능력 추이(트렌드) 분석**: 데이터를 시간 구간별로 나눠 Cp/Cpk가 개선/악화되는지 추적하는 전용 탭 (구간 크기 슬라이더로 조정 가능)
- **Excel 리포트 다운로드**: 원본데이터·공정능력지수·Nelson's Rules 위반·자동진단·추이 데이터를 한 번에 담은 .xlsx 파일을 클릭 한 번으로 다운로드
- **커스텀 테마 + 상태 배너**: 공정 상태(양호/주의/불량)를 색상 배너로 즉시 인지 가능, 카드형 메트릭 UI

## 채점 기준 대응
| 항목 | 대응 내용 |
|---|---|
| 기본기능(3) | Cp/Cpk/Pp/Ppk, Xbar-R/S, I-MR, P/NP 관리도, Nelson's Rules 8종, 데이터 변경 시 자동 재계산 |
| 기본UI(3) | 탭 구조, 카드형 메트릭, 컬러 테마, 사이드바 설정, Plotly 인터랙티브 차트 |
| 배포(2) | GitHub → Streamlit Community Cloud 무료 배포 |
| 차별화 기능/UI(2) | 게이지 차트, 자동 진단 코멘트, 능력 추이 분석, Excel 리포트 다운로드 |

## 로컬 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 데이터 형식
- 계량형: 측정값 컬럼 + 서브그룹 ID 컬럼(예: 시간/Lot 번호). `sample_data/sample_measurement.csv` 참고
- 계수형: 불량(결점)수 컬럼 + 표본크기 컬럼(P chart) 또는 고정 n(NP chart). `sample_data/sample_attribute.csv` 참고
- 사이드바에서 CSV를 업로드하지 않으면 샘플 데이터로 자동 시연됩니다.

## 배포
GitHub 저장소를 Streamlit Community Cloud(share.streamlit.io)에 연결하여 무료로 배포합니다.
자세한 절차는 과제 제출 시 안내받은 단계를 따르세요 (`app.py`를 메인 파일로 지정).
