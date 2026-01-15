import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 앱 페이지 설정
st.set_page_config(
    page_title="Institutional Accumulation Tracker",
    page_icon="🎯",
    layout="centered"
)

# 앱 제목 및 설명
st.title("기관 매집 분석기 (Institutional Tracker)")
st.markdown("""
이 앱은 **거래량 가중 평균 가격(VWAP)**과 **OBV(On-Balance Volume)**를 결합하여 
장중 기관의 '진성 매집' 여부를 수치화합니다.
""")

# 2. 사이드바 설정 (사용자 입력)
st.sidebar.header("🔍 분석 설정")
ticker = st.sidebar.text_input("분석할 티커를 입력하세요", "IONQ").upper()
period = st.sidebar.selectbox("데이터 조회 기간", ["1d", "2d", "5d", "1mo"], index=1)

# 시간 간격 설명 매핑
interval_mapping = {
    "1m (단기/초단타 - 노이즈 높음)": "1m",
    "2m (단기 변동성 분석)": "2m",
    "5m (데이트레이딩 표준 - 추천)": "5m",
    "15m (중기 추세 확인)": "15m",
    "60m (장기 수급 흐름)": "60m"
}

selected_interval_label = st.sidebar.selectbox(
    "시간 간격 (분석 해상도)",
    options=list(interval_mapping.keys()),
    index=2
)
interval = interval_mapping[selected_interval_label]

# 3. 데이터 분석 핵심 함수
def get_analysis_data(symbol, p, i):
    try:
        # 데이터 호출
        df = yf.download(symbol, period=p, interval=i, progress=False)
        
        if df.empty:
            return None, "데이터를 찾을 수 없습니다. 티커를 다시 확인해 주세요."

        # [오류 해결] MultiIndex 컬럼 평탄화 (yfinance 최신 버전 대응)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 지표 계산: Typical Price (TP), VWAP
        df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['VWAP'] = (df['TP'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        
        # 지표 계산: OBV (On-Balance Volume)
        df['OBV'] = (df['Volume'] * (~df['Close'].diff().le(0) * 2 - 1)).cumsum()
        
        # 기관 매집 점수(Accumulation Score) 산출 로직
        day_start_obv = df['OBV'].iloc[0]
        current_obv = df['OBV'].iloc[-1]
        
        # 거래량 상승 효율성 (평균 거래량 대비 OBV 변화량)
        obv_efficiency = (current_obv - day_start_obv) / df['Volume'].mean()
        
        # VWAP 대비 가격 이격 (기관 평단 대비 수익권 여부)
        vwap_efficiency = (df['Close'].iloc[-1] / df['VWAP'].iloc[-1]) - 1
        
        # 최종 점수 산출
        acc_score = obv_efficiency * (1 + vwap_efficiency * 100)
        
        return {"df": df, "score": acc_score}, None
    
    except Exception as e:
        return None, f"분석 중 오류 발생: {str(e)}"

# 4. 메인 화면 로직 및 시각화
if ticker:
    with st.spinner(f'{ticker} 데이터를 분석 중입니다...'):
        result, error = get_analysis_data(ticker, period, interval)

    if error:
        st.error(error)
    elif result:
        df = result['df']
        score = result['score']

        # 상단 스코어 요약 섹션
        col1, col2 = st.columns([1, 1])
        with col1:
            st.metric(label=f"현재 {ticker} 매집 강도", value=f"{score:.2f}")
        
        with col2:
            if score >= 8.0:
                st.error("🔥 매우 강력 / 과열 신호")
            elif score >= 3.0:
                st.success("🚀 강력 매집 중 (추천)")
            elif score >= 1.0:
                st.info("✅ 양호한 수급 흐름")
            else:
                st.warning("⚠️ 매집 약함 / 매도 우위")

        # 분석 기준 및 간격 리마크
        st.info(f"💡 현재 **{selected_interval_label}** 기준으로 분석 중입니다.")

        # [리마크 가이드] 점수 해석 표
        with st.expander("💡 기관 매집 점수(Accumulation Score) 해석 가이드", expanded=True):
            st.markdown("""
            | 점수 구간 | 상태 | 분석 리마크 |
            | :--- | :--- | :--- |
            | **8.0 이상** | **과열 (Overheated)** | 강력한 기관 매집 혹은 숏 스퀴즈. 급등 후 조정 가능성 유의. |
            | **3.0 ~ 8.0** | **강력 (Strong)** | **개미털기 후 진성 매수세 유입.** 주가가 기관 평단(VWAP) 위에서 지지됨. |
            | **1.0 ~ 3.0** | **양호 (Healthy)** | 안정적인 수급. 거래량이 주가 상승을 견조하게 뒷받침 중. |
            | **1.0 미만** | **주의 (Caution)** | 매집 세력 부재. 거래량 없는 상승이거나 매도 압력이 높은 상태. |
            """)
            st.caption("※ 본 점수는 거래량 대비 가격 상승 효율을 정량화한 지표로, 수치가 높을수록 '큰 손'의 개입 가능성이 큽니다.")

        st.divider()

        # 5. 차트 시각화 (Price vs VWAP & OBV Trend)
        # 차트 스타일 설정
        plt.rcParams['figure.facecolor'] = 'white'
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

        # 상단 차트: 주가와 기관 평균가(VWAP)
        ax1.plot(df.index, df['Close'], label='Close Price', color='black', linewidth=1.5, alpha=0.8)
        ax1.plot(df.index, df['VWAP'], label='VWAP (Inst. Average)', color='red', linestyle='--', linewidth=1.5)
        ax1.set_title(f"[{ticker}] Price vs Institutional Average", fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, linestyle=':', alpha=0.6)

        # 하단 차트: 거래량 추세(OBV)
        ax2.plot(df.index, df['OBV'], label='OBV (Accumulation Trend)', color='blue', linewidth=1.2)
        ax2.set_title(f"[{ticker}] Volume Accumulation Trend (OBV)", fontsize=14, fontweight='bold')
        ax2.fill_between(df.index, df['OBV'], color='blue', alpha=0.1) # 가독성을 위한 채우기
        ax2.legend(loc='upper left')
        ax2.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        st.pyplot(fig)

        # 하단: 원본 데이터 확인 (필요시)
        with st.expander("📊 실시간 데이터 프레임 확인 (최근 10개 캔들)"):
            st.dataframe(df.tail(10))

else:
    st.info("사이드바에서 티커를 입력하고 분석을 시작하세요.")