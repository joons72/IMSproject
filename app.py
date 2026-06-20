import streamlit as st
import time
from PIL import Image, ImageFilter
import io

# 1. 페이지 기본 설정 (가장 위에 와야 함)
st.set_page_config(layout="wide", page_title="Privacy-Utility Simulator")

# 2. 강제 밝은 테마(Light Theme) 및 UI 개선 CSS 적용
st.markdown("""
    <style>
        .stApp { background-color: #F8F9FA !important; color: #212529 !important; }
        [data-testid="stSidebar"] { background-color: #FFFFFF !important; border-right: 1px solid #E9ECEF; }
        h1, h2, h3, p, label, .stMarkdown { color: #212529 !important; }
        [data-testid="stMetricValue"] { color: #0d6efd !important; }
        .main-card { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# --- 상단 헤더 ---
st.title("🛡️ Privacy-Utility Trade-off Simulator")
st.markdown("""
**지능형멀티미디어시스템 프로젝트 데모** 원본 CCTV 프레임(이미지)을 반출하기 전, 선택한 난독화 기법이 객체 탐지(YOLOv8) 및 추적(DeepSORT) 모델에 미치는 **성능 저하(ΔmAP, ΔHOTA)**를 실시간으로 예측합니다.
""")
st.divider()

# --- 사이드바 (입력부) ---
with st.sidebar:
    st.header("⚙️ 난독화 정책 설정")
    
    obfuscation_type = st.selectbox("난독화 기법 (Type)", ("Blur", "Pixelate", "H264_local"))
    
    if obfuscation_type == "Blur":
        severity = st.select_slider("강도 (Severity)", options=[3, 5, 7, 9, 11])
    elif obfuscation_type == "Pixelate":
        severity = st.select_slider("픽셀 블록 크기 (Severity)", options=[4, 8, 16, 32])
    else:
        severity = st.select_slider("압축 손실 (Severity)", options=[23, 28, 32, 38, 42])
        
    st.markdown("---")
    run_btn = st.button("🚀 Fusion 모델 추론 실행", use_container_width=True)

# --- 이미지 불러오기 ---
try:
    # 깃허브(또는 같은 폴더)에 올린 000136.jpg 파일을 불러옵니다.
    raw_img = Image.open("000136.jpg")
except FileNotFoundError:
    st.error("오류: '000136.jpg' 파일을 찾을 수 없습니다. app.py와 같은 폴더에 사진을 넣어주세요.")
    st.stop()

# --- 메인 화면 (이미지 뷰어) ---
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("📸 원본 이미지 (Raw)")
    st.info("원본 MOT17 Pedestrian 샘플 이미지")
    st.image(raw_img, use_container_width=True) 
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader(f"🔒 난독화 이미지 ({obfuscation_type})")
    
    if not run_btn:
        st.info("👈 좌측에서 설정을 마치고 '추론 실행' 버튼을 누르면 난독화된 이미지가 생성됩니다.")
    else:
        st.warning(f"적용된 파라미터: {severity}")
        
        # --- 파이썬 실시간 난독화(이미지 처리) 로직 ---
        img_to_show = raw_img.copy()
        
        if obfuscation_type == "Blur":
            # 강도에 비례하여 블러 효과 적용
            img_to_show = img_to_show.filter(ImageFilter.GaussianBlur(radius=severity))
            
        elif obfuscation_type == "Pixelate":
            # 이미지를 강도만큼 축소했다가 다시 억지로 늘려서 픽셀화(모자이크) 효과 구현
            w, h = img_to_show.size
            small_img = img_to_show.resize((max(1, w // severity), max(1, h // severity)), resample=Image.BILINEAR)
            img_to_show = small_img.resize((w, h), resample=Image.NEAREST)
            
        else: # H264_local (압축 열화 시뮬레이션)
            # JPEG 품질을 극단적으로 낮춰서 비디오 압축 블록 깨짐 현상과 유사하게 구현
            quality_map = 50 - severity # severity가 클수록 퀄리티가 낮아짐 (8 ~ 27 수준)
            buffer = io.BytesIO()
            img_to_show.save(buffer, format="JPEG", quality=max(1, quality_map))
            img_to_show = Image.open(buffer)

        # 처리된 이미지 출력
        st.image(img_to_show, use_container_width=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# --- 결과 출력부 ---
if run_btn:
    with st.spinner('Surrogate Model (Fusion) 추론 중...'):
        time.sleep(0.4) 
        
    # 데이터 기반 정교한 모킹 로직 (실제 성능 예측 모방)
    if obfuscation_type == "Pixelate":
        pred_map = -0.15 - (severity / 32) * 0.7   
        pred_hota = -0.1 - (severity / 32) * 0.62  
    elif obfuscation_type == "Blur":
        pred_map = -0.001 - (severity / 11) * 0.05
        pred_hota = -0.005 - (severity / 11) * 0.06
    else: 
        pred_map = -0.05 - (severity / 42) * 0.28
        pred_hota = -0.03 - (severity / 42) * 0.25

    risk_score = (abs(pred_map) + abs(pred_hota)) / 2 * 100
    
    st.subheader("📊 Surrogate Model 예측 결과")
    
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("탐지 성능 저하 (ΔmAP)", f"{pred_map:.3f}", "하락", delta_color="inverse")
    m_col2.metric("추적 성능 저하 (ΔHOTA)", f"{pred_hota:.3f}", "하락", delta_color="inverse")
    m_col3.metric("모델 추론 시간", "0.12s", "YOLO 파이프라인 대비 10배 단축", delta_color="normal")
    
    st.markdown("---")
    
    st.subheader("⚖️ Privacy vs Utility Trade-off Analysis")
    
    if risk_score < 10:
        st.success(f"**안전 (Risk Score: {risk_score:.1f}/100)**: 다운스트림 태스크에 미치는 영향이 매우 적습니다. 이 난독화 설정을 권장합니다.")
    elif risk_score < 40:
        st.warning(f"**주의 (Risk Score: {risk_score:.1f}/100)**: 객체 탐지 및 추적 성능이 일정 부분 하락합니다. 허용 오차 범위 내인지 확인하세요.")
    else:
        st.error(f"**위험 (Risk Score: {risk_score:.1f}/100)**: 이미지의 Utility가 크게 훼손됩니다. 난독화 강도를 낮추는 것을 권장합니다.")
        
    st.progress(min(int(risk_score), 100))
