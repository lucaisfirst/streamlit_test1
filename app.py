import os
import streamlit as st
import time
import base64
import uuid
import tempfile
import io
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import requests
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from pypdf import PdfReader
import subprocess
import platform
import fitz  # PyMuPDF

# 대시보드 기능 가져오기
try:
    from correction_dashboard import render_document_correction_dashboard
except ImportError:
    st.error("correction_dashboard.py 파일이 필요합니다.")

# 페이지 설정 - 타이틀 변경
st.set_page_config(
    page_title="문서교정 AI Agent by Refinery",  # 여기를 변경
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 반응형 디자인만 유지하고 기존 색상 복원
st.markdown("""
<style>
    /* 글로벌 폰트 설정 */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Noto Sans KR', sans-serif;
        font-size: 16px;
    }
    
    /* 헤더 스타일 */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Noto Sans KR', sans-serif;
        font-weight: 700;
        color: #1E293B;
    }
    
    h1 {
        font-size: 2.2rem;
        margin-bottom: 1.5rem;
    }
    
    h2 {
        font-size: 1.8rem;
        margin-bottom: 1.2rem;
    }
    
    h3 {
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* 메인 컨테이너 스타일 */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1200px;
    }
    
    /* 사이드바 스타일 */
    .sidebar .sidebar-content {
        background-color: #f8fafc;
        padding: 1.5rem;
    }
    
    /* 카드 스타일 */
    .card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        transition: transform 0.3s, box-shadow 0.3s;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
    }
    
    /* 메트릭 카드 스타일 */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        height: 100%;
    }
    
    .metric-card h3 {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 0.5rem;
    }
    
    .metric-card h2 {
        font-size: 1.8rem;
        color: #0f172a;
        margin-bottom: 0;
    }
    
    /* 갤러리 카드 스타일 개선 */
    .gallery-card {
        border: none;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        background-color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: transform 0.3s, box-shadow 0.3s;
        position: relative;
        overflow: hidden;
    }
    
    .gallery-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
    }
    
    .gallery-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(to bottom, #3b82f6, #60a5fa);
    }
    
    .gallery-card h4 {
        margin-top: 0;
        color: #1e40af;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    .gallery-card p {
        margin-bottom: 0.8rem;
        font-size: 0.95rem;
        color: #475569;
    }
    
    .gallery-card .metrics {
        display: flex;
        justify-content: space-between;
        margin-top: 1rem;
        font-size: 0.9rem;
        color: #64748b;
        background-color: #f1f5f9;
        padding: 0.5rem 0.8rem;
        border-radius: 8px;
    }
    
    .gallery-card .tag {
        display: inline-block;
        background-color: #e0f2fe;
        color: #0369a1;
        border-radius: 20px;
        padding: 0.3rem 0.8rem;
        margin-right: 0.5rem;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .gallery-card .date {
        position: absolute;
        top: 1rem;
        right: 1rem;
        font-size: 0.8rem;
        color: #94a3b8;
    }
    
    .gallery-card .thumbnail {
        width: 100%;
        height: 120px;
        object-fit: cover;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    /* 탭 스타일 개선 */
    .stTabs {
        background-color: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 0.3rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: auto;
        white-space: normal;
        background-color: transparent;
        border-radius: 6px;
        padding: 0.8rem 1.2rem;
        font-size: 1rem;
        font-weight: 500;
        color: #64748b;
        margin: 0 0.2rem;
        transition: all 0.2s;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
        font-weight: 600;
    }
    
    /* 버튼 스타일 개선 */
    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
        padding: 0.5rem 1rem;
        background-color: #3b82f6;
        color: white;
        border: none;
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        background-color: #2563eb;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }
    
    /* 검색 결과 카드 스타일 개선 */
    .search-result-card {
        border-left: 4px solid #3b82f6;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.2rem;
        background-color: white;
        border-radius: 0 12px 12px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: transform 0.3s, box-shadow 0.3s;
    }
    
    .search-result-card:hover {
        transform: translateX(5px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
    }
    
    .search-result-card h5 {
        margin-top: 0;
        color: #1e40af;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    .search-result-card .source {
        font-size: 0.85rem;
        color: #64748b;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }
    
    .search-result-card .source::before {
        content: '🔗';
        margin-right: 0.5rem;
    }
    
    .search-result-card .type {
        display: inline-block;
        background-color: #e0f2fe;
        color: #0369a1;
        border-radius: 20px;
        padding: 0.3rem 0.8rem;
        margin-left: 0.8rem;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .search-result-card .summary {
        font-size: 0.95rem;
        line-height: 1.6;
        color: #475569;
    }
    
    /* 프롬프트 카드 스타일 개선 */
    .prompt-card {
        background-color: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    .prompt-card h5 {
        color: #1e40af;
        margin-top: 0;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .prompt-card pre {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        overflow-x: auto;
        font-size: 0.95rem;
        line-height: 1.5;
        color: #334155;
        border: 1px solid #e2e8f0;
    }
    
    /* 메뉴 선택 스타일 개선 */
    .menu-container {
        display: flex;
        flex-direction: column;
        gap: 0.8rem;
        margin-bottom: 2rem;
    }
    
    .menu-item {
        display: flex;
        align-items: center;
        padding: 1rem;
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .menu-item:hover, .menu-item.active {
        background-color: #eff6ff;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
        transform: translateX(5px);
    }
    
    .menu-item.active {
        border-left: 4px solid #3b82f6;
    }
    
    .menu-icon {
        font-size: 1.5rem;
        margin-right: 1rem;
        color: #3b82f6;
    }
    
    .menu-text {
        font-weight: 500;
        color: #334155;
    }
    
    .menu-item.active .menu-text {
        color: #1e40af;
        font-weight: 600;
    }
    
    /* 모바일 반응형 */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }
        .stSidebar {
            width: 100% !important;
        }
        iframe {
            width: 100% !important;
            height: 400px !important;
        }
        .pdf-container {
            width: 100% !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# 필요한 라이브러리 임포트 확인
try:
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.llms import Ollama
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
except ImportError:
    st.error("필요한 라이브러리를 임포트할 수 없습니다.")

# 세션 상태 초기화
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    st.session_state.messages = []
    
    # 히스토리 데이터 초기화
    if "correction_history" not in st.session_state:
        st.session_state.correction_history = []
    
    # 검색 결과 캐시 초기화
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    
    # 추천 프롬프트 캐시 초기화
    if "recommended_prompts" not in st.session_state:
        st.session_state.recommended_prompts = []
    
    # 현재 선택된 메뉴
    if "current_menu" not in st.session_state:
        st.session_state.current_menu = "PDF 문서 챗봇"

# 환경 변수 설정
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "llama3.2")
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "llama3.2")

# 유틸리티 함수
def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None

# PDF 표시 함수 수정 - 스크롤 가능한 컨테이너 내에 이미지 표시
def display_pdf(file):
    try:
        # PDF 파일 읽기
        file.seek(0)
        file_bytes = file.read()
        
        # 다운로드 버튼
        st.download_button(
            label="PDF 다운로드",
            data=file_bytes,
            file_name=file.name,
            mime="application/pdf"
        )
        
        # PyMuPDF를 사용하여 PDF를 이미지로 변환
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        
        # 페이지 수 확인
        total_pages = len(doc)
        
        if total_pages == 0:
            st.warning("PDF에 페이지가 없습니다.")
            return
        
        # 미리보기 페이지 수 제한 (최대 5페이지)
        preview_pages = min(5, total_pages)
        
        st.markdown(f"### PDF 미리보기 (총 {total_pages}페이지 중 {preview_pages}페이지 표시)")
        
        # 이미지 데이터를 저장할 리스트
        images = []
        captions = []
        
        # 각 페이지를 이미지로 변환
        for page_num in range(preview_pages):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2))  # 해상도 조정
            img_bytes = pix.tobytes("png")
            images.append(img_bytes)
            captions.append(f"페이지 {page_num + 1}/{total_pages}")
        
        # 문서 닫기
        doc.close()
        
        # 스크롤 가능한 컨테이너 생성
        with st.container():
            # 스크롤 가능한 영역 CSS 적용
            st.markdown("""
            <style>
            .pdf-preview-box {
                height: 500px;
                overflow-y: auto;
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                padding: 15px;
                background-color: #f9f9f9;
                margin-bottom: 20px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # HTML로 스크롤 가능한 컨테이너 생성
            html_content = '<div class="pdf-preview-box">'
            
            # 이미지를 base64로 인코딩하여 HTML에 추가
            for i, img_bytes in enumerate(images):
                img_base64 = base64.b64encode(img_bytes).decode()
                html_content += f'<p style="text-align:center; color:#555; font-size:0.9rem;">{captions[i]}</p>'
                html_content += f'<img src="data:image/png;base64,{img_base64}" style="width:100%; margin-bottom:15px; border:1px solid #ddd; border-radius:5px;">'
                if i < len(images) - 1:
                    html_content += '<hr style="border-top:1px dashed #ccc; margin:15px 0;">'
            
            html_content += '</div>'
            
            # HTML 렌더링
            st.markdown(html_content, unsafe_allow_html=True)
        
        # 더 많은 페이지가 있는 경우 안내 메시지
        if total_pages > preview_pages:
            st.info(f"전체 {total_pages}페이지 중 {preview_pages}페이지만 미리보기로 표시됩니다. 전체 내용을 보려면 PDF를 다운로드하세요.")
        
    except Exception as e:
        st.error(f"PDF 표시 중 오류가 발생했습니다: {e}")
        st.error("PDF 미리보기를 표시할 수 없습니다. 다운로드 버튼을 사용하여 PDF를 확인하세요.")
        
        # 오류 발생 시에도 다운로드 버튼 제공
        file.seek(0)
        st.download_button(
            label="PDF 다운로드",
            data=file.read(),
            file_name=file.name,
            mime="application/pdf"
        )
    
    # 파일 포인터 위치 초기화
    file.seek(0)

# 문서 교정 히스토리 저장 함수
def save_correction_history(file_name, correction_type, errors_found, corrections_made):
    # 실제 구현에서는 데이터베이스에 저장하는 것이 좋습니다
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 썸네일 이미지 생성 (실제로는 PDF 첫 페이지를 이미지로 변환)
    # 여기서는 더미 이미지 데이터 사용
    colors = [(73, 109, 137), (210, 95, 95), (95, 210, 137), (210, 180, 95), (180, 95, 210)]
    dummy_image = Image.new('RGB', (300, 400), color=random.choice(colors))
    buffered = io.BytesIO()
    dummy_image.save(buffered, format="JPEG")
    thumbnail = base64.b64encode(buffered.getvalue()).decode()
    
    # 태그 생성
    possible_tags = ["문법", "맞춤법", "서식", "문체", "용어", "일관성", "가독성", "전문성", "간결성"]
    tags = random.sample(possible_tags, k=random.randint(2, 4))
    
    history_item = {
        "id": str(uuid.uuid4()),
        "timestamp": timestamp,
        "file_name": file_name,
        "correction_type": correction_type,
        "errors_found": errors_found,
        "corrections_made": corrections_made,
        "thumbnail": thumbnail,
        "tags": tags,
        "status": random.choice(["완료", "진행중"]),
        "completion_rate": random.randint(70, 100) if random.random() > 0.2 else random.randint(30, 69)
    }
    
    st.session_state.correction_history.append(history_item)
    return history_item

# 문서교정 대시보드 기능 개선
def render_document_correction_dashboard():
    st.markdown('<h1 style="text-align: center; margin-bottom: 2rem;">문서교정 통계 대시보드</h1>', unsafe_allow_html=True)
    
    # 탭 스타일 개선
    st.markdown("""
    <style>
        /* 전체 대시보드 컨테이너 */
        .dashboard-container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        /* 탭 컨테이너 정렬 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            background-color: #f8fafc;
            border-radius: 8px;
            padding: 0.5rem;
            justify-content: center;
            margin-bottom: 1.5rem;
        }
        
        /* 탭 버튼 스타일 */
        .stTabs [data-baseweb="tab"] {
            padding: 0.8rem 1.5rem;
            margin: 0 0.25rem;
            border-radius: 6px;
        }
        
        /* 메트릭 카드 그리드 */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        
        /* 메트릭 카드 */
        .metric-card {
            background-color: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            text-align: center;
            height: 100%;
            min-height: 140px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        
        /* 차트 컨테이너 */
        .chart-container {
            background-color: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            margin-bottom: 1.5rem;
            height: 100%;
        }
        
        /* 차트 제목 */
        .chart-title {
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 1rem;
            color: #1e293b;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # 대시보드 컨테이너 시작
    st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
    
    # 탭 생성으로 다양한 뷰 제공
    tabs = st.tabs(["📊 교정 현황 개요", "📈 상세 분석", "🖼️ 교정 히스토리 갤러리"])
    
    # 샘플 데이터 생성 함수는 그대로 유지
    def generate_sample_data():
        doc_types = ["계약서", "보고서", "제안서", "매뉴얼", "정책문서", "회의록", "법률문서", "기술문서"]
        dates = [(datetime.now() - timedelta(days=x)).strftime('%Y-%m-%d') for x in range(30)]
        
        data = []
        for date in dates:
            for doc_type in doc_types:
                correction_count = np.random.randint(1, 15)
                grammar_errors = np.random.randint(1, 10)
                spelling_errors = np.random.randint(1, 8)
                style_issues = np.random.randint(0, 5)
                formatting_issues = np.random.randint(0, 6)
                status = np.random.choice(["완료", "진행중", "대기중"], p=[0.7, 0.2, 0.1])
                correction_time = np.random.randint(10, 120)
                
                data.append({
                    "날짜": date,
                    "문서유형": doc_type,
                    "교정수량": correction_count,
                    "문법오류": grammar_errors,
                    "맞춤법오류": spelling_errors,
                    "문체오류": style_issues,
                    "서식오류": formatting_issues,
                    "상태": status,
                    "교정시간(분)": correction_time,
                    "총오류수": grammar_errors + spelling_errors + style_issues + formatting_issues,
                })
        
        return pd.DataFrame(data)
    
    # 샘플 데이터 로드
    df = generate_sample_data()
    
    # 탭 1: 교정 현황 개요
    with tabs[0]:
        # 주요 지표 카드 - HTML 그리드 사용
        total_docs = df["교정수량"].sum()
        total_errors = df["총오류수"].sum()
        avg_errors = round(df["총오류수"].sum() / df["교정수량"].sum(), 2)
        avg_time = round(df["교정시간(분)"].mean(), 1)
        
        st.markdown(f'''
        <div class="metrics-grid">
            <div class="metric-card">
                <h3 style="margin:0; color:#64748b; font-size:1rem; font-weight:500;">총 교정 문서</h3>
                <h2 style="margin:0.5rem 0; color:#1e293b; font-size:2rem; font-weight:700;">{total_docs:,}건</h2>
            </div>
            <div class="metric-card">
                <h3 style="margin:0; color:#64748b; font-size:1rem; font-weight:500;">총 발견 오류</h3>
                <h2 style="margin:0.5rem 0; color:#1e293b; font-size:2rem; font-weight:700;">{total_errors:,}건</h2>
            </div>
            <div class="metric-card">
                <h3 style="margin:0; color:#64748b; font-size:1rem; font-weight:500;">문서당 평균 오류</h3>
                <h2 style="margin:0.5rem 0; color:#1e293b; font-size:2rem; font-weight:700;">{avg_errors}건</h2>
            </div>
            <div class="metric-card">
                <h3 style="margin:0; color:#64748b; font-size:1rem; font-weight:500;">평균 교정 시간</h3>
                <h2 style="margin:0.5rem 0; color:#1e293b; font-size:2rem; font-weight:700;">{avg_time}분</h2>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # 문서 유형별 현황 및 오류 유형별 분포
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">문서 유형별 교정 현황</div>', unsafe_allow_html=True)
            
            doc_type_counts = df.groupby("문서유형").agg({
                "교정수량": "sum",
                "총오류수": "sum"
            }).reset_index()
            
            doc_type_counts = doc_type_counts.sort_values("교정수량", ascending=False)
            
            fig = px.bar(
                doc_type_counts,
                x="문서유형",
                y="교정수량",
                color="총오류수",
                color_continuous_scale="Blues",
                text_auto=True
            )
            
            fig.update_layout(
                height=350,
                margin=dict(t=10, b=40, l=40, r=10),
                xaxis_title="문서 유형",
                yaxis_title="교정 수량 (건)",
                coloraxis_colorbar_title="총 오류 수",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">오류 유형별 분포</div>', unsafe_allow_html=True)
            
            error_types = {
                "문법 오류": df["문법오류"].sum(),
                "맞춤법 오류": df["맞춤법오류"].sum(),
                "문체 문제": df["문체오류"].sum(),
                "서식 문제": df["서식오류"].sum(),
            }
            
            error_df = pd.DataFrame({
                "오류 유형": list(error_types.keys()),
                "오류 수": list(error_types.values())
            })
            
            fig_pie = px.pie(
                error_df,
                names="오류 유형",
                values="오류 수",
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.4
            )
            
            fig_pie.update_layout(
                height=350,
                margin=dict(t=10, b=10, l=10, r=10),
                legend_title="오류 유형",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 교정 진행 상태
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">교정 진행 상태</div>', unsafe_allow_html=True)
        
        status_counts = df["상태"].value_counts().reset_index()
        status_counts.columns = ["상태", "문서수"]
        
        fig_status = px.bar(
            status_counts,
            x="문서수",
            y="상태",
            color="상태",
            color_discrete_map={
                "완료": "#10b981",
                "진행중": "#3b82f6",
                "대기중": "#f59e0b"
            },
            orientation="h",
            text_auto=True
        )
        
        fig_status.update_layout(
            height=250,
            margin=dict(t=10, b=40, l=40, r=10),
            xaxis_title="문서 수",
            yaxis_title="상태",
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        
        st.plotly_chart(fig_status, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 탭 2: 상세 분석
    with tabs[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('### 문서 교정 상세 데이터')
        
        # 필터 옵션
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_doc_types = st.multiselect(
                "문서 유형 선택",
                options=sorted(df["문서유형"].unique()),
                default=sorted(df["문서유형"].unique())
            )
        
        with col2:
            date_range = st.date_input(
                "기간 선택",
                value=(
                    datetime.strptime(min(df["날짜"]), "%Y-%m-%d").date(),
                    datetime.strptime(max(df["날짜"]), "%Y-%m-%d").date()
                ),
                format="YYYY-MM-DD"
            )
        
        with col3:
            status_filter = st.multiselect(
                "상태 선택",
                options=sorted(df["상태"].unique()),
                default=sorted(df["상태"].unique())
            )
        
        # 필터 적용
        filtered_df = df.copy()
        
        if selected_doc_types:
            filtered_df = filtered_df[filtered_df["문서유형"].isin(selected_doc_types)]
        
        if len(date_range) == 2:
            start_date = date_range[0].strftime("%Y-%m-%d")
            end_date = date_range[1].strftime("%Y-%m-%d")
            filtered_df = filtered_df[(filtered_df["날짜"] >= start_date) & (filtered_df["날짜"] <= end_date)]
        
        if status_filter:
            filtered_df = filtered_df[filtered_df["상태"].isin(status_filter)]
        
        # 시간별 추이 그래프
        st.subheader("날짜별 교정 추이")
        
        daily_data = filtered_df.groupby("날짜").agg({
            "교정수량": "sum",
            "총오류수": "sum",
            "교정시간(분)": "mean"
        }).reset_index()
        
        daily_data["날짜"] = pd.to_datetime(daily_data["날짜"])
        daily_data = daily_data.sort_values("날짜")
        
        fig_trend = go.Figure()
        
        fig_trend.add_trace(go.Scatter(
            x=daily_data["날짜"],
            y=daily_data["교정수량"],
            name="교정 수량",
            line=dict(color="#1E88E5", width=3)
        ))
        
        fig_trend.add_trace(go.Scatter(
            x=daily_data["날짜"],
            y=daily_data["총오류수"],
            name="오류 수",
            line=dict(color="#E53935", width=3)
        ))
        
        fig_trend.update_layout(
            height=400,
            xaxis_title="날짜",
            yaxis_title="건수",
            legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # 교정 소요 시간 분석
        st.subheader("문서 유형별 평균 교정 시간")
        
        avg_time_by_type = filtered_df.groupby("문서유형")["교정시간(분)"].mean().reset_index()
        avg_time_by_type = avg_time_by_type.sort_values("교정시간(분)", ascending=False)
        
        fig_time = px.bar(
            avg_time_by_type,
            x="문서유형",
            y="교정시간(분)",
            color="교정시간(분)",
            color_continuous_scale="Viridis",
            text_auto='.1f'
        )
        
        fig_time.update_layout(
            height=400,
            xaxis_title="문서 유형",
            yaxis_title="평균 교정 시간 (분)",
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig_time, use_container_width=True)
        
        # 상세 데이터 테이블
        st.subheader("상세 데이터")
        
        display_columns = ["날짜", "문서유형", "교정수량", "총오류수", "문법오류", "맞춤법오류", "문체오류", "서식오류", "상태", "교정시간(분)"]
        st.dataframe(filtered_df[display_columns], use_container_width=True)
        
        # 데이터 다운로드 버튼
        csv = filtered_df[display_columns].to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="데이터 CSV 다운로드",
            data=csv,
            file_name="문서교정_데이터.csv",
            mime="text/csv"
        )
    
    # 탭 3: 교정 히스토리 갤러리
    with tabs[2]:
        st.subheader("문서 교정 히스토리")
        
        # 히스토리 필터링 옵션
        col1, col2 = st.columns(2)
        with col1:
            filter_type = st.selectbox(
                "교정 유형 필터",
                ["전체", "문법 교정", "맞춤법 교정", "문체 교정", "서식 교정"]
            )
        
        with col2:
            sort_option = st.selectbox(
                "정렬 기준",
                ["최신순", "오래된순", "오류 많은순", "교정 많은순"]
            )
        
        # 샘플 히스토리 데이터 생성 (실제로는 DB에서 가져옴)
        if len(st.session_state.correction_history) == 0:
            # 샘플 데이터 생성
            sample_files = ["사업계획서.pdf", "제안서_최종.pdf", "계약서_초안.pdf", 
                           "회의록_2023.pdf", "기술문서_v1.pdf", "정책보고서.pdf"]
            
            for file in sample_files:
                correction_type = random.choice(["문법 교정", "맞춤법 교정", "문체 교정", "서식 교정"])
                errors_found = random.randint(5, 30)
                corrections_made = random.randint(3, errors_found)
                save_correction_history(file, correction_type, errors_found, corrections_made)
        
        # 히스토리 필터링
        filtered_history = st.session_state.correction_history
        if filter_type != "전체":
            filtered_history = [h for h in filtered_history if h["correction_type"] == filter_type]
        
        # 정렬
        if sort_option == "최신순":
            filtered_history = sorted(filtered_history, key=lambda x: x["timestamp"], reverse=True)
        elif sort_option == "오래된순":
            filtered_history = sorted(filtered_history, key=lambda x: x["timestamp"])
        elif sort_option == "오류 많은순":
            filtered_history = sorted(filtered_history, key=lambda x: x["errors_found"], reverse=True)
        elif sort_option == "교정 많은순":
            filtered_history = sorted(filtered_history, key=lambda x: x["corrections_made"], reverse=True)
        
        # 갤러리 표시
        if not filtered_history:
            st.info("조건에 맞는 교정 히스토리가 없습니다.")
        else:
            # 3열 그리드로 표시
            cols = st.columns(3)
            for i, item in enumerate(filtered_history):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class="gallery-card">
                        <h4>{item["file_name"]}</h4>
                        <p><strong>교정 유형:</strong> {item["correction_type"]}</p>
                        <p><strong>교정 일시:</strong> {item["timestamp"]}</p>
                        <div class="metrics">
                            <span>발견된 오류: {item["errors_found"]}개</span>
                            <span>교정된 항목: {item["corrections_made"]}개</span>
                        </div>
                        <div style="margin-top: 8px;">
                            {"".join([f'<span class="tag">{tag}</span>' for tag in item["tags"]])}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 상세 보기 버튼 (실제로는 상세 페이지로 연결)
                    if st.button(f"상세 보기 #{i+1}", key=f"view_{item['id']}"):
                        st.session_state.selected_history = item["id"]
    
    # 대시보드 컨테이너 종료
    st.markdown('</div>', unsafe_allow_html=True)

# Ollama 설정 및 모델 관련 함수 수정
def get_ollama_llm(model_name="llama3.2"):
    """Ollama LLM 모델을 초기화합니다."""
    try:
        # 명시적으로 llama3.2 모델과 base_url 지정
        return Ollama(model="llama3.2", base_url=OLLAMA_BASE_URL)
    except Exception as e:
        st.error(f"Ollama 모델 초기화 중 오류 발생: {e}")
        return None

# PDF 처리 함수
def process_pdf(pdf_file):
    """PDF 파일을 처리하여 텍스트를 추출하고 벡터 저장소를 생성합니다."""
    try:
        # PDF에서 텍스트 추출
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        
        if not text.strip():
            st.warning("PDF에서 텍스트를 추출할 수 없습니다.")
            return None
        
        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        if not chunks:
            st.warning("PDF 내용을 처리할 수 없습니다.")
            return None
        
        # 임베딩 생성 및 벡터 저장소 초기화 - 모델명 명시적으로 지정
        embeddings = OllamaEmbeddings(model="llama3.2", base_url=OLLAMA_BASE_URL)
        vectorstore = FAISS.from_texts(chunks, embeddings)
        
        return vectorstore
    
    except Exception as e:
        st.error(f"PDF 처리 중 오류 발생: {e}")
        return None

# 챗봇 초기화 함수
def initialize_chatbot(vectorstore=None):
    """챗봇을 초기화합니다. vectorstore가 제공되면 RAG 기반 챗봇을, 아니면 일반 챗봇을 반환합니다."""
    try:
        # 명시적으로 llama3.2 모델 사용
        llm = get_ollama_llm("llama3.2")
        
        if not llm:
            st.error("LLM을 초기화할 수 없습니다.")
            return None
        
        if vectorstore:
            # RAG 기반 챗봇 설정
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # 프롬프트 템플릿 설정
            prompt_template = """
            당신은 문서 내용을 바탕으로 질문에 답변하는 AI 문서 도우미입니다.
            
            문맥: {context}
            
            질문: {question}
            
            답변:
            """
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # 대화형 검색 체인 생성
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                memory=memory,
                combine_docs_chain_kwargs={"prompt": PROMPT}
            )
            
            return chain
        else:
            # 일반 챗봇 설정 (RAG 없음)
            return llm
    
    except Exception as e:
        st.error(f"챗봇 초기화 중 오류 발생: {e}")
        return None

# Ollama 서버 상태 확인
def check_ollama_server():
    """Ollama 서버가 실행 중인지 확인합니다."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        return response.status_code == 200
    except:
        return False

# Ollama 모델 실행 함수
def run_ollama_model(model_name="llama3.2:latest"):
    """지정된 Ollama 모델을 실행합니다."""
    try:
        # 운영체제 확인
        system = platform.system()
        
        command = ["ollama", "run", model_name]
        
        # 명령어 실행 중임을 표시
        st.info(f"'{' '.join(command)}' 명령어를 실행 중입니다...")
        
        if system == "Darwin" or system == "Linux":  # macOS 또는 Linux
            # 백그라운드에서 Ollama 모델 실행
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 잠시 대기
            time.sleep(2)
            
            # 프로세스가 실행 중인지 확인
            if process.poll() is None:
                return True, f"{model_name} 모델이 성공적으로 로드되었습니다."
            else:
                stderr = process.stderr.read()
                return False, f"모델 로드 실패: {stderr}"
                
        elif system == "Windows":
            # Windows에서 Ollama 모델 실행
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            
            # 잠시 대기
            time.sleep(2)
            
            # 프로세스가 실행 중인지 확인
            if process.poll() is None:
                return True, f"{model_name} 모델이 성공적으로 로드되었습니다."
            else:
                stderr = process.stderr.read()
                return False, f"모델 로드 실패: {stderr}"
        else:
            return False, f"지원되지 않는 운영체제: {system}"
            
    except Exception as e:
        return False, f"Ollama 모델 실행 중 오류 발생: {e}"

# PDF 챗봇 기능 수정 - 오류 처리 강화
def render_pdf_chatbot():
    st.header("PDF 문서 챗봇", divider="green")
    
    # Ollama 서버 상태 확인
    if not check_ollama_server():
        st.error("Ollama 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
        st.info("로컬에서 Ollama를 실행하려면 터미널에서 'ollama serve' 명령어를 실행하세요.")
        return
    
    # 모델 가용성 확인
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        available_models = [model["name"] for model in response.json()["models"]]
        
        if "llama3.2" not in available_models and "llama3.2:latest" not in available_models:
            st.warning("llama3.2 모델이 설치되어 있지 않습니다. 터미널에서 'ollama pull llama3.2' 명령어를 실행하여 모델을 설치하세요.")
            st.info(f"사용 가능한 모델: {', '.join(available_models)}")
    except Exception as e:
        st.warning(f"모델 목록을 가져오는 중 오류 발생: {e}")
    
    # Ollama 설정
    with st.sidebar:
        st.subheader("AI 모델 설정")
        
        # Ollama 모델 선택 (UI만 제공)
        ollama_model = st.selectbox(
            "AI 모델 선택",
            ["llama3", "llama3.2:latest", "mistral", "gemma:7b", "phi3:mini"],
            index=1  # llama3.2:latest를 기본값으로 설정
        )
        
        # 모델 정보 표시
        st.info(f"현재 사용 중인 모델: llama3.2:latest")
        st.markdown("""
        <div style="background-color: #f0f7ff; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
            <p style="margin: 0; font-size: 0.9rem;">
                <strong>참고:</strong> 로컬에서 실행 중인 llama3.2 모델을 사용합니다.
                모델이 없는 경우 터미널에서 <code>ollama pull llama3.2</code> 명령어를 실행하세요.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 기존 PDF 업로드 섹션
        st.subheader("PDF 업로드")
        uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")
        
        # 챗봇 초기화 상태
        if "chatbot" not in st.session_state:
            st.session_state.chatbot = None
            st.session_state.vectorstore = None
        
        if uploaded_file is not None:
            # PDF 파일 처리
            if "pdf_processed" not in st.session_state or st.session_state.pdf_processed != uploaded_file.name:
                with st.spinner("PDF를 처리 중입니다..."):
                    # PDF 처리 및 벡터 저장소 생성
                    vectorstore = process_pdf(uploaded_file)
                    
                    if vectorstore:
                        st.session_state.vectorstore = vectorstore
                        st.session_state.chatbot = initialize_chatbot(vectorstore)
                        st.session_state.pdf_processed = uploaded_file.name
                        st.success("PDF 처리가 완료되었습니다!")
                    else:
                        st.error("PDF 처리 중 오류가 발생했습니다.")
            
            # 문서 교정 옵션
            st.subheader("문서 교정 옵션")
            correction_type = st.selectbox(
                "교정 유형 선택",
                ["문법 교정", "맞춤법 교정", "문체 교정", "서식 교정"]
            )
            
            if st.button("문서 교정 시작", use_container_width=True):
                with st.spinner("문서를 교정 중입니다..."):
                    # 실제 구현에서는 여기서 문서 교정 로직 실행
                    time.sleep(2)  # 교정 시간 시뮬레이션
                    
                    # 교정 결과 저장
                    errors_found = random.randint(5, 20)
                    corrections_made = random.randint(3, errors_found)
                    save_correction_history(
                        uploaded_file.name,
                        correction_type,
                        errors_found,
                        corrections_made
                    )
                    
                    st.success(f"문서 교정이 완료되었습니다! {errors_found}개의 오류를 발견하고 {corrections_made}개를 수정했습니다.")
        else:
            # PDF가 업로드되지 않은 경우 일반 챗봇 초기화
            if st.session_state.chatbot is None or st.session_state.vectorstore is not None:
                st.session_state.vectorstore = None
                st.session_state.chatbot = initialize_chatbot()
                st.session_state.pdf_processed = None
    
    # 메인 영역
    if uploaded_file is not None:
        # PDF 미리보기와 챗봇 영역 분리
        st.subheader("PDF 미리보기")
        display_pdf(uploaded_file)
        
        st.markdown("---")
    
    # 챗봇 인터페이스 (PDF 업로드 여부와 관계없이 항상 표시)
    st.subheader("AI 문서 도우미와 대화하기")
    
    # 초기 안내 메시지 설정
    if "messages" not in st.session_state or not st.session_state.messages:
        if uploaded_file is not None:
            st.session_state.messages = [
                {"role": "assistant", "content": f"안녕하세요! '{uploaded_file.name}' 문서에 대해 질문해 주세요. 문서 내용을 바탕으로 답변해 드리겠습니다."}
            ]
        else:
            st.session_state.messages = [
                {"role": "assistant", "content": "안녕하세요! 궁금한 점이 있으시면 질문해 주세요. PDF를 업로드하시면 문서 내용을 바탕으로 더 정확한 답변을 드릴 수 있습니다."}
            ]
    
    # 메시지 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 사용자 입력
    if prompt := st.chat_input("질문을 입력하세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 챗봇 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                if st.session_state.chatbot:
                    try:
                        if st.session_state.vectorstore:
                            # RAG 기반 응답 생성
                            response = st.session_state.chatbot({"question": prompt})
                            response_text = response["answer"]
                        else:
                            # 일반 LLM 응답 생성
                            response_text = st.session_state.chatbot.invoke(
                                f"사용자 질문: {prompt}\n\n답변:"
                            )
                    except Exception as e:
                        st.error(f"응답 생성 중 오류 발생: {e}")
                        response_text = "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다. 다시 시도해 주세요."
                else:
                    response_text = "챗봇이 초기화되지 않았습니다. 페이지를 새로고침하거나 다시 시도해 주세요."
                
                # 응답 표시
                st.markdown(response_text)
        
        # 챗봇 메시지 추가
        st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    # 대화 초기화 버튼
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("대화 초기화", key="reset_chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# JD/RFP 검색 및 AI 요약 기능
def render_jd_rfp_search():
    st.header("JD/RFP 검색 및 AI 요약", divider="blue")
    
    # 검색 섹션
    st.subheader("문서 검색")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("검색어를 입력하세요 (예: '프론트엔드 개발자 JD', '웹 애플리케이션 RFP')")
    
    with col2:
        search_type = st.selectbox("문서 유형", ["전체", "JD (채용공고)", "RFP (제안요청서)"])
        search_button = st.button("검색", use_container_width=True)
    
    # 검색 결과 표시
    if search_button and search_query:
        with st.spinner("검색 중..."):
            # 실제로는 API 호출 등을 통해 검색 결과를 가져옴
            # 여기서는 샘플 데이터 생성
            time.sleep(1)  # 검색 시간 시뮬레이션
            
            # 샘플 검색 결과
            sample_results = []
            if "개발자" in search_query.lower() or "프론트엔드" in search_query.lower():
                sample_results.extend([
                    {
                        "title": "시니어 프론트엔드 개발자 (React/TypeScript)",
                        "source": "테크 스타트업 A사",
                        "type": "JD (채용공고)",
                        "content": """
                        # 주요 업무
                        - React와 TypeScript를 활용한 웹 애플리케이션 개발
                        - 컴포넌트 기반 UI 설계 및 구현
                        - RESTful API 연동 및 상태 관리
                        - 성능 최적화 및 크로스 브라우저 호환성 유지
                        
                        # 자격 요건
                        - 프론트엔드 개발 경력 5년 이상
                        - React, TypeScript 숙련자
                        - 모던 웹 개발 도구 및 방법론에 대한 이해
                        - 반응형 웹 디자인 경험
                        """
                    },
                    {
                        "title": "주니어 프론트엔드 개발자",
                        "source": "IT 서비스 기업 B사",
                        "type": "JD (채용공고)",
                        "content": """
                        # 주요 업무
                        - HTML, CSS, JavaScript를 활용한 웹 페이지 개발
                        - Vue.js 프레임워크 기반 UI 구현
                        - 반응형 웹 디자인 적용
                        
                        # 자격 요건
                        - 프론트엔드 개발 경력 1-3년
                        - HTML, CSS, JavaScript 기본 지식
                        - Vue.js 사용 경험
                        - 웹 표준 및 접근성에 대한 이해
                        """
                    }
                ])
            
            if "rfp" in search_query.lower() or "제안" in search_query.lower() or "웹" in search_query.lower():
                sample_results.extend([
                    {
                        "title": "기업 웹사이트 리뉴얼 프로젝트 RFP",
                        "source": "대기업 C사",
                        "type": "RFP (제안요청서)",
                        "content": """
                        # 프로젝트 개요
                        - 기존 기업 웹사이트의 디자인 및 기능 개선
                        - 반응형 웹 디자인 적용
                        - 콘텐츠 관리
                        """
                    }
                ])

# 고객 관리 CRM 대시보드 기능 개선
def render_customer_management_crm():
    st.header("고객 관리 CRM", divider="orange")
    
    # CSS 스타일 정의
    st.markdown("""
    <style>
        /* 메트릭 카드 그리드 */
        .crm-metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        
        /* 메트릭 카드 */
        .crm-metric-card {
            background-color: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            text-align: center;
            height: 100%;
            min-height: 140px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            border-top: 4px solid #3b82f6;
        }
        
        /* 매출 메트릭 카드 */
        .revenue-card {
            border-top-color: #10b981;
        }
        
        /* 이번 달 메트릭 카드 */
        .monthly-card {
            border-top-color: #f59e0b;
        }
        
        /* 전월 대비 메트릭 카드 */
        .growth-card {
            border-top-color: #8b5cf6;
        }
        
        /* CRM 카드 스타일 */
        .crm-card {
            background-color: white;
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 16px;
            border: 1px solid #e2e8f0;
            transition: all 0.2s ease;
        }
        
        .crm-card:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border-color: #cbd5e1;
        }
        
        .customer-name {
            font-size: 1.1rem;
            font-weight: 600;
            color: #1e3a8a;
        }
        
        .customer-meta {
            font-size: 0.85rem;
            color: #64748b;
            margin-bottom: 8px;
        }
        
        .price-tag {
            display: inline-block;
            padding: 3px 8px;
            background-color: #e0f2fe;
            color: #0284c7;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: 500;
        }
        
        .deadline-tag {
            display: inline-block;
            padding: 3px 8px;
            background-color: #fee2e2;
            color: #b91c1c;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: 500;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # 샘플 고객 데이터 생성
    def generate_customers(n=15):
        customers = []
        names = ["김지민", "이서연", "박준호", "최민지", "정우진", "한소희", "강도현", "윤지원", "조현우", "신지은"]
        companies = ["ABC주식회사", "테크스타트", "글로벌기업", "신생벤처", "대학원생", "프리랜서", "중소기업", "대기업"]
        doc_types = ["이력서", "자기소개서", "논문", "기획서", "보고서", "번역", "사업계획서"]
        
        for i in range(n):
            days_ago = np.random.randint(1, 30)
            entry_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
            
            deadline_days = np.random.randint(1, 15)
            deadline = (datetime.now() + timedelta(days=deadline_days)).strftime('%Y-%m-%d')
            
            doc_type = np.random.choice(doc_types)
            price_ranges = {
                "이력서": [50000, 150000],
                "자기소개서": [70000, 200000],
                "논문": [200000, 800000],
                "기획서": [150000, 400000],
                "보고서": [100000, 300000],
                "번역": [150000, 500000],
                "사업계획서": [300000, 1000000]
            }
            
            price = np.random.randint(price_ranges[doc_type][0], price_ranges[doc_type][1])
            work_count = np.random.randint(2, 10) * 500
            
            customers.append({
                "id": i+1,
                "name": np.random.choice(names),
                "company": np.random.choice(companies),
                "doc_type": doc_type,
                "entry_date": entry_date,
                "deadline": deadline,
                "price": price,
                "work_count": work_count,
                "status": np.random.choice(["대기중", "진행중", "완료", "지연됨"], p=[0.2, 0.4, 0.3, 0.1])
            })
        
        return customers
    
    # 샘플 데이터 생성 (분석용 더 많은 데이터)
    all_customers = generate_customers(50)
    
    # 요약 통계 계산
    total_customers = len(all_customers)
    total_works = len(all_customers)  # 각 고객당 1개 작업 가정
    total_revenue = sum(customer["price"] for customer in all_customers)
    avg_revenue_per_customer = int(total_revenue / total_customers)
    
    # 이번 달 데이터 필터링 (현재 월에 해당하는 데이터)
    current_month = datetime.now().month
    this_month_customers = [
        customer for customer in all_customers 
        if datetime.strptime(customer["entry_date"], "%Y-%m-%d").month == current_month
    ]
    this_month_revenue = sum(customer["price"] for customer in this_month_customers)
    
    # 전월 대비 성장률 (랜덤 값으로 가정)
    growth_rate = np.random.uniform(-0.05, 0.15)  # -5%에서 +15% 사이
    
    # 상단 메트릭 카드 표시
    st.markdown(f'''
    <div class="crm-metrics-grid">
        <div class="crm-metric-card">
            <h3 style="margin:0; color:#64748b; font-size:1rem; font-weight:500;">총 고객 수</h3>
            <h2 style="margin:0.5rem 0; color:#1e293b; font-size:2rem; font-weight:700;">{total_customers}명</h2>
        </div>
        <div class="crm-metric-card revenue-card">
            <h3 style="margin:0; color:#64748b; font-size:1rem; font-weight:500;">총 매출</h3>
            <h2 style="margin:0.5rem 0; color:#1e293b; font-size:2rem; font-weight:700;">{total_revenue:,}원</h2>
        </div>
        <div class="crm-metric-card monthly-card">
            <h3 style="margin:0; color:#64748b; font-size:1rem; font-weight:500;">이번 달 매출</h3>
            <h2 style="margin:0.5rem 0; color:#1e293b; font-size:2rem; font-weight:700;">{this_month_revenue:,}원</h2>
        </div>
        <div class="crm-metric-card growth-card">
            <h3 style="margin:0; color:#64748b; font-size:1rem; font-weight:500;">전월 대비 성장률</h3>
            <h2 style="margin:0.5rem 0; color:{'#10b981' if growth_rate >= 0 else '#ef4444'}; font-size:2rem; font-weight:700;">{growth_rate*100:.1f}%</h2>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # 필터 영역
    col1, col2, col3 = st.columns(3)
    with col1:
        doc_type_filter = st.multiselect(
            "문서 타입",
            ["이력서", "자기소개서", "논문", "기획서", "보고서", "번역", "사업계획서"],
            default=[]
        )
    
    with col2:
        status_filter = st.multiselect(
            "상태",
            ["대기중", "진행중", "완료", "지연됨"],
            default=[]
        )
    
    with col3:
        sort_by = st.selectbox(
            "정렬 기준",
            ["마감일", "가격", "등록일", "고객명"],
            index=0
        )
    
    # 필터 적용된 고객 데이터
    customers = generate_customers(15)  # 표시용 데이터는 15개만
    
    # 필터 적용
    if doc_type_filter:
        customers = [c for c in customers if c["doc_type"] in doc_type_filter]
    
    if status_filter:
        customers = [c for c in customers if c["status"] in status_filter]
    
    # 정렬 적용
    if sort_by == "마감일":
        customers = sorted(customers, key=lambda x: x["deadline"])
    elif sort_by == "가격":
        customers = sorted(customers, key=lambda x: x["price"], reverse=True)
    elif sort_by == "등록일":
        customers = sorted(customers, key=lambda x: x["entry_date"], reverse=True)
    elif sort_by == "고객명":
        customers = sorted(customers, key=lambda x: x["name"])
    
    # 고객 목록 표시
    st.subheader(f"고객 목록 ({len(customers)}명)")
    
    if not customers:
        st.info("조건에 맞는 고객이 없습니다.")
    else:
        # 고객 카드 표시
        for customer in customers:
            status_color = {
                "대기중": "#f59e0b",
                "진행중": "#3b82f6",
                "완료": "#10b981",
                "지연됨": "#ef4444"
            }.get(customer["status"], "#6b7280")
            
            st.markdown(f"""
            <div class="crm-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <div class="customer-name">{customer["name"]}</div>
                        <div class="customer-meta">{customer["company"]} • {customer["doc_type"]}</div>
                        <div style="margin-top: 8px;">
                            <span class="price-tag">₩{customer["price"]:,}</span>
                            <span style="margin-left: 8px; color: #64748b; font-size: 0.85rem;">{customer["work_count"]:,}자</span>
                        </div>
                    </div>
                    <div>
                        <div style="text-align: right;">
                            <span style="background-color: {status_color}; color: white; padding: 3px 8px; border-radius: 15px; font-size: 0.8rem;">{customer["status"]}</span>
                        </div>
                        <div style="margin-top: 8px; text-align: right;">
                            <span class="customer-meta">등록일: {customer["entry_date"]}</span>
                        </div>
                        <div style="margin-top: 4px; text-align: right;">
                            <span class="deadline-tag">마감: {customer["deadline"]}</span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # 데이터 분석 차트
    st.subheader("고객 데이터 분석")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 문서 유형별 매출 분포
        doc_revenues = {}
        for customer in all_customers:
            doc_type = customer["doc_type"]
            doc_revenues[doc_type] = doc_revenues.get(doc_type, 0) + customer["price"]
        
        doc_df = pd.DataFrame({
            "문서 유형": list(doc_revenues.keys()),
            "매출": list(doc_revenues.values())
        })
        
        fig = px.pie(
            doc_df,
            names="문서 유형",
            values="매출",
            title="문서 유형별 매출 분포",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 월별 매출 트렌드 (가상 데이터로 생성)
        months = ["1월", "2월", "3월", "4월", "5월", "6월", "7월", "8월", "9월", "10월", "11월", "12월"]
        current_month = datetime.now().month
        
        # 현재 월까지의 실제 데이터로 가정하고, 이후는 예측치로 표시
        monthly_revenue = []
        for i in range(1, 13):
            if i <= current_month:
                # 실제 데이터처럼 보이게 하기 위한 값
                base = np.random.randint(5000000, 8000000)
                variation = np.random.uniform(0.8, 1.2)
                monthly_revenue.append(int(base * variation))
            else:
                # 예측치는 점선으로 표시할 더미 값
                monthly_revenue.append(None)
        
        monthly_df = pd.DataFrame({
            "월": months,
            "매출": monthly_revenue
        })
        
        fig = px.line(
            monthly_df,
            x="월",
            y="매출",
            title="월별 매출 트렌드",
            markers=True
        )
        
        # 현재 월까지 실선, 이후 점선으로 표시 (예측)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # 대시보드 컨테이너 종료
    st.markdown('</div>', unsafe_allow_html=True)

# 고객 챗 대시보드 기능
def render_customer_chat():
    st.header("고객 채팅", divider="violet")
    
    # CSS 스타일 정의
    st.markdown("""
    <style>
        /* 채팅 인터페이스 스타일 */
        .chat-container {
            display: flex;
            flex-direction: column;
            height: 500px;
            background-color: #f8fafc;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #e2e8f0;
        }
        
        .chat-header {
            padding: 15px;
            background-color: #1e40af;
            color: white;
            font-weight: 600;
            border-bottom: 1px solid #3b82f6;
        }
        
        .chat-messages {
            flex-grow: 1;
            padding: 15px;
            overflow-y: auto;
            background-color: #f8fafc;
        }
        
        .chat-input {
            padding: 15px;
            background-color: white;
            border-top: 1px solid #e2e8f0;
        }
        
        .message {
            margin-bottom: 15px;
            max-width: 80%;
        }
        
        .message-sent {
            margin-left: auto;
            background-color: #1e40af;
            color: white;
            border-radius: 18px 18px 0 18px;
            padding: 10px 15px;
        }
        
        .message-received {
            margin-right: auto;
            background-color: white;
            border: 1px solid #e2e8f0;
            border-radius: 18px 18px 18px 0;
            padding: 10px 15px;
        }
        
        .message-time {
            font-size: 0.7rem;
            color: #94a3b8;
            margin-top: 5px;
        }
        
        .file-shared {
            background-color: #f1f5f9;
            border-radius: 12px;
            padding: 10px;
            margin-bottom: 15px;
            font-size: 0.9rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # 사이드바에 고객 목록
    with st.sidebar:
        st.subheader("고객 목록")
        
        # 샘플 고객 데이터
        customers = [
            {"id": 1, "name": "김지민", "doc_type": "이력서", "status": "진행중"},
            {"id": 2, "name": "이서연", "doc_type": "자기소개서", "status": "대기중"},
            {"id": 3, "name": "박준호", "doc_type": "논문", "status": "진행중"},
            {"id": 4, "name": "최민지", "doc_type": "번역", "status": "완료"},
            {"id": 5, "name": "정우진", "doc_type": "사업계획서", "status": "지연됨"}
        ]
        
        # 고객 선택
        selected_customer = None
        for customer in customers:
            status_color = {
                "대기중": "#f59e0b",
                "진행중": "#3b82f6",
                "완료": "#10b981",
                "지연됨": "#ef4444"
            }.get(customer["status"], "#6b7280")
            
            if st.button(
                f"{customer['name']} - {customer['doc_type']}",
                key=f"customer_{customer['id']}",
                help=f"상태: {customer['status']}"
            ):
                selected_customer = customer
        
        if "selected_customer" not in st.session_state:
            st.session_state.selected_customer = customers[0]
        
        if selected_customer:
            st.session_state.selected_customer = selected_customer
    
    # 채팅 메시지 초기화
    if "chat_messages" not in st.session_state:
        # 샘플 채팅 메시지 - user는 고객, assistant는 작업자
        st.session_state.chat_messages = [
            {"role": "user", "content": "안녕하세요! 문서 교정 진행상황이 궁금합니다.", "time": "10:15 AM"},
            {"role": "assistant", "content": "안녕하세요. 현재 문서 교정이 약 60% 진행되었습니다. 예상보다 오류가 많아 조금 더 시간이 필요합니다.", "time": "10:16 AM"},
            {"role": "user", "content": "네, 알겠습니다. 대략 언제쯤 완료될까요?", "time": "10:17 AM"},
            {"role": "assistant", "content": "내일 오후 3시까지는 완료할 예정입니다. 혹시 더 빨리 필요하신가요?", "time": "10:18 AM"},
            {"role": "user", "content": "아니요, 내일이면 충분합니다. 감사합니다!", "time": "10:20 AM"},
            {"role": "assistant", "content": "네, 확인했습니다. 궁금하신 점 있으시면 언제든 문의해주세요.", "time": "10:21 AM"},
        ]
    
    # 선택된 고객 정보 표시
    customer = st.session_state.selected_customer
    
    st.subheader(f"{customer['name']}님과의 대화")
    st.caption(f"문서: {customer['doc_type']} | 상태: {customer['status']}")
    
    # 파일 업로드 섹션
    uploaded_file = st.file_uploader("파일 공유하기", type=["pdf", "docx", "txt", "jpg", "png"], key="customer_file_upload")
    
    if uploaded_file:
        # 파일 공유 메시지 추가
        file_time = datetime.now().strftime("%I:%M %p")
        st.session_state.chat_messages.append({
            "role": "assistant", 
            "content": f"파일을 공유했습니다: {uploaded_file.name}", 
            "time": file_time,
            "file": uploaded_file.name
        })
        
        st.success(f"파일 '{uploaded_file.name}'이(가) 성공적으로 공유되었습니다.")
    
    # 채팅 인터페이스
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.chat_messages:
            # 메시지 방향 수정 - assistant(작업자)가 오른쪽, user(고객)가 왼쪽에 표시되도록
            align = "right" if msg["role"] == "assistant" else "left"
            bg_color = "#1e40af" if msg["role"] == "assistant" else "white"
            text_color = "white" if msg["role"] == "assistant" else "#1e293b"
            
            if "file" in msg:
                st.markdown(f"""
                <div style="display: flex; justify-content: {align}; margin-bottom: 15px;">
                    <div class="file-shared" style="max-width: 80%;">
                        <div style="display: flex; align-items: center;">
                            <span style="font-size: 1.5rem; margin-right: 10px;">📎</span>
                            <div>
                                <div style="font-weight: 500; color: #1e293b;">{msg["file"]}</div>
                                <div class="message-time">{msg["time"]}</div>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # 메시지 말풍선 모양도 방향에 맞게 수정
                border_radius = "18px 18px 0 18px" if msg["role"] == "assistant" else "18px 18px 18px 0"
                border = "" if msg["role"] == "assistant" else "1px solid #e2e8f0"
                time_color = "rgba(255,255,255,0.7)" if msg["role"] == "assistant" else "#94a3b8"
                
                st.markdown(f"""
                <div style="display: flex; justify-content: {align}; margin-bottom: 15px;">
                    <div class="message" style="background-color: {bg_color}; color: {text_color}; border-radius: {border_radius}; padding: 10px 15px; border: {border};">
                        <div>{msg["content"]}</div>
                        <div class="message-time" style="color: {time_color};">{msg["time"]}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # 채팅 입력창
    prompt = st.text_input("메시지를 입력하세요...", key="customer_chat_input")
    col1, col2 = st.columns([4, 1])
    
    with col2:
        if st.button("전송", use_container_width=True, key="send_customer_chat"):
            if prompt:
                # 현재 시간 구하기
                current_time = datetime.now().strftime("%I:%M %p")
                
                # 작업자(assistant) 메시지 추가 - 여기서 역할 변경
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": prompt,
                    "time": current_time
                })
                
                # 자동 응답 (고객 응답 시뮬레이션)
                auto_responses = [
                    "감사합니다. 확인했습니다.",
                    "언제쯤 완료될까요?",
                    "수정사항을 반영해주셔서 감사합니다.",
                    "추가로 변경하고 싶은 부분이 있습니다.",
                    "네, 마감일 전에 완료해 주시면 감사하겠습니다."
                ]
                
                # 자동 응답 지연 효과
                time.sleep(1)
                
                # 고객(user) 응답 메시지 추가
                st.session_state.chat_messages.append({
                    "role": "user",
                    "content": random.choice(auto_responses),
                    "time": datetime.now().strftime("%I:%M %p")
                })
                
                # 채팅 인터페이스 업데이트 - experimental_rerun() 대신 rerun() 사용
                st.rerun()

# 메인 함수 수정 - 메뉴 추가
def main():
    # 사이드바 메뉴
    with st.sidebar:
        st.title("문서교정 AI Agent by Refinery")  # 여기를 변경
        st.markdown("---")
        
        # 메뉴 선택 (CRM과 채팅 메뉴 추가)
        page = st.radio(
            "메뉴 선택",
            ["PDF 문서 챗봇", "문서교정 대시보드", "JD/RFP 검색 및 요약", "고객 관리 CRM", "고객 채팅"],
            format_func=lambda x: f"📄 {x}" if x == "PDF 문서 챗봇" 
                    else (f"📊 {x}" if x == "문서교정 대시보드" 
                    else (f"🔍 {x}" if x == "JD/RFP 검색 및 요약" 
                    else (f"👥 {x}" if x == "고객 관리 CRM" 
                    else f"💬 {x}")))
        )
    
    # 선택된 페이지 렌더링
    if page == "PDF 문서 챗봇":
        render_pdf_chatbot()
    elif page == "문서교정 대시보드":
        render_document_correction_dashboard()
    elif page == "JD/RFP 검색 및 요약":
        render_jd_rfp_search()
    elif page == "고객 관리 CRM":
        render_customer_management_crm()
    elif page == "고객 채팅":
        render_customer_chat()

# 앱 실행
if __name__ == "__main__":
    main() 