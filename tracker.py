import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 앱 페이지 설정
st.set_page_config(
    page_title="Dual-Frame Institutional Tracker",
    page_icon="📈",
    layout="centered"
)

st.title("📈 단기-중기 기관 매집 분석기")
st.markdown("""
이 앱은 **단기(Short-term) 수급**과 **중기(Mid-term) 추세**를 동시에 계산하여 
현재의 반등이 일시적인지, 아니면 지속적인 매집의 연장선인지 분석합니다.
""")

# 2. 사이드바 설정
st.sidebar.header("🔍 분석 설정")
ticker = st.sidebar.text_input("분석할 티커를 입력하세요", "IONQ").upper()

# 중기 기준이 될 전체 기간 선택
period = st.sidebar.selectbox("중기 분석 기간 (조회 범위)", ["5d", "1mo", "3mo"], index=0)

# 시간 간격 설명 매핑
interval_mapping = {
    "1m (단기/초단타)": "1m",
    "2m (변동성 분석)": "2m",
    "5m (표준 - 추천)": "5m",
    "15m (중기 추세)": "15m"
}
selected_interval_label = st.sidebar.selectbox(
    "시간 간격 (해상도)",
    options=list(interval_mapping.keys()),
    index=2
)
interval = interval_mapping[selected_interval_label]

# 3. 데이터 분석 및 듀얼 스코어 계산 함수
def analyze_dual_scores(symbol, p, i):
    try:
        df = yf.download(symbol, period=p, interval=i, progress=False)
        if df.empty: return None, "데이터를 찾을 수 없습니다."
        
        # MultiIndex 컬럼 평탄화
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 지표 계산 (전체 데이터 기반)
        df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['VWAP'] = (df['TP'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        df['OBV'] = (df['Volume'] * (~df['Close'].diff().le(0) * 2 - 1)).cumsum()
        
        # --- 점수 산출 로직 ---
        def calculate_score(data_slice):
            day_start_obv = data_slice['OBV'].iloc[0]
            current_obv = data_slice['OBV'].iloc[-1]
            obv_growth = (current_obv - day_start_obv) / data_slice['Volume'].mean()
            vwap_eff = (data_slice['Close'].iloc[-1] / data_slice['VWAP'].iloc[-1]) - 1
            return obv_growth * (1 + vwap_eff * 100)

        # 1. 중기 점수 (선택한 전체 기간)
        mid_term_score = calculate_score(df)
        
        # 2. 단기 점수 (최근 20%의 데이터 혹은 최근 1~2일치)
        # 데이터가 충분하다면 마지막 1/5 지점부터 계산
        short_slice_idx = int(len(df) * 0.8)
        short_term_df = df.iloc[short_slice_idx:]
        short_term_score = calculate_score(short_term_df)
        
        return {"df": df, "short": short_term_score, "mid": mid_term_score}, None
    except Exception as e:
        return None, str(e)

# 4. 메인 로직
if ticker:
    result, error = analyze_dual_scores(ticker, period, interval)

    if error:
        st.error(error)
    elif result:
        df = result['df']
        s_score = result['short']
        m_score = result['mid']

        # 상단 듀얼 메트릭
        st.subheader(f"📊 {ticker} 수급 분석 결과")
        c1, c2 = st.columns(2)
        c1.metric(label="⚡ 단기 매집 점수 (최근 흐름)", value=f"{s_score:.2f}", 
                  delta=f"{s_score - m_score:.2f} (대비)")
        c2.metric(label="🏢 중기 매집 점수 (누적 추세)", value=f"{m_score:.2f}")

        # 종합 진단 리마크
        st.divider()
        if s_score > 3.0 and m_score > 1.0:
            st.success("🌟 **강력 매집 일치:** 중기적 매집 추세 속에서 단기적인 강한 반등이 포착됩니다. 신뢰도가 매우 높습니다.")
        elif s_score > 3.0 and m_score <= 1.0:
            st.info("⚡ **단기 수급 유입:** 중기 추세는 정체되어 있으나, 최근 거래량이 실린 강한 반등(개미털기 후 반등 등)이 시작되었습니다.")
        elif s_score <= 1.0 and m_score > 3.0:
            st.warning("🐢 **중기 매집 유지 / 단기 조정:** 큰 세력은 이탈하지 않았으나 현재 단기적인 매도 압력이 있거나 쉬어가는 구간입니다.")
        else:
            st.error("⚠️ **주의:** 단기/중기 모두 매집 신호가 약합니다. 관망이 필요한 구간입니다.")

        # 점수 해석 가이드 (Expander)
        with st.expander("💡 듀얼 점수 해석 팁"):
            st.write("""
            - **단기 점수:** 최근 몇 시간~며칠 사이의 '개미털기' 여부와 즉각적인 반등 에너지를 보여줍니다.
            - **중기 점수:** 며칠~몇 주간 쌓아온 세력의 평균 단가(VWAP) 대비 현재 위치를 보여줍니다.
            - **Delta(삼각형):** 중기 추세 대비 단기 에너지가 얼마나 더 강한지를 나타냅니다. 플러스(+)일수록 반등의 탄력이 좋습니다.
            """)

        # 차트 출력
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        ax1.plot(df.index, df['Close'], color='black', label='Price', alpha=0.7)
        ax1.plot(df.index, df['VWAP'], color='red', linestyle='--', label='Institutional Avg (VWAP)')
        ax1.set_title("Price & VWAP Trend")
        ax1.legend(); ax1.grid(True, alpha=0.2)

        ax2.plot(df.index, df['OBV'], color='blue', label='OBV Trend')
        ax2.set_title("Volume Accumulation (OBV)")
        ax2.fill_between(df.index, df['OBV'], color='blue', alpha=0.1)
        ax2.legend(); ax2.grid(True, alpha=0.2)
        
        st.pyplot(fig)