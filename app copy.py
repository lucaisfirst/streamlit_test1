import requests
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pypdf import PdfReader

# 대시보드 기능 가져오기
try:
    from correction_dashboard import render_document_correction_dashboard
except ImportError:
    st.error("correction_dashboard.py 파일이 필요합니다.")

# 페이지 설정 - 모바일 호환성 개선
st.set_page_config(
    page_title="AI 문서 도우미",
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
    
    /* 태그 스타일 */
    .tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-right: 6px;
        margin-bottom: 6px;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .tag.selected {
        background-color: #1e3a8a;
        color: white;
    }
    
    .tag:not(.selected) {
        background-color: #e2e8f0;
        color: #475569;
    }
    
    .tag:hover:not(.selected) {
        background-color: #cbd5e1;
    }
    
    /* 필터 그룹 스타일 */
    .filter-group {
        margin-bottom: 15px;
    }
    
    .filter-label {
        font-size: 0.9rem;
        font-weight: 500;
        color: #334155;
        margin-bottom: 8px;
    }
    
    /* 갤러리 카드 스타일 */
    .gallery-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .gallery-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .card-title {
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 8px;
        color: #1e293b;
    }
    
    .card-meta {
        color: #64748b;
        font-size: 0.85rem;
        margin-bottom: 5px;
    }
    
    .card-tags {
        margin-top: 10px;
        margin-bottom: 10px;
    }
    
    .card-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin-right: 4px;
        background-color: #f1f5f9;
        color: #475569;
    }
    
    .card-status {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        color: white;
    }
    
    .card-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 12px;
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

# 문서교정 대시보드 기능
def render_document_correction_dashboard():
    # CSS 스타일 정의
    st.markdown("""
    <style>
        /* 전체 대시보드 스타일 */
        .dashboard-container {
            padding: 10px;
            background-color: #f8fafc;
            border-radius: 10px;
        }
        
        /* 헤더 스타일 */
        .dashboard-header {
            font-size: 1.5rem;
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #e2e8f0;
        }
        
        /* 탭 스타일 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            white-space: pre-wrap;
            border-radius: 4px;
            font-weight: 500;
            background-color: #f1f5f9;
            color: #64748b;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #1e3a8a !important;
            color: white !important;
        }
        
        /* 메트릭 카드 스타일 */
        .metric-container {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            text-align: center;
            height: 100%;
            border: 1px solid #e2e8f0;
        }
        
        .metric-title {
            font-size: 0.9rem;
            color: #64748b;
            margin-bottom: 8px;
        }
        
        .metric-value {
            font-size: 1.8rem;
            font-weight: 600;
            color: #1e293b;
        }
        
        /* 차트 컨테이너 스타일 */
        .chart-container {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border: 1px solid #e2e8f0;
        }
        
        .chart-title {
            font-size: 1rem;
            font-weight: 500;
            color: #1e293b;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 1px solid #e2e8f0;
        }
        
        /* 필터 컨테이너 스타일 */
        .filter-container {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border: 1px solid #e2e8f0;
        }
        
        .filter-label {
            font-size: 0.9rem;
            font-weight: 500;
            color: #1e293b;
            margin-bottom: 8px;
        }
        
        .filter-group {
            margin-bottom: 15px;
        }
        
        /* 태그 스타일 */
        .tag {
            display: inline-block;
            padding: 5px 10px;
            background-color: #f1f5f9;
            color: #64748b;
            border-radius: 20px;
            margin-right: 8px;
            margin-bottom: 8px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s;
            border: 1px solid #e2e8f0;
        }
        
        .tag:hover {
            background-color: #e2e8f0;
        }
        
        .tag.selected {
            background-color: #1e3a8a;
            color: white;
            border-color: #1e3a8a;
        }
        
        /* 데이터 테이블 스타일 */
        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        
        .data-table th {
            background-color: #f1f5f9;
            padding: 10px;
            text-align: left;
            font-weight: 500;
            color: #1e293b;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .data-table td {
            padding: 10px;
            border-bottom: 1px solid #e2e8f0;
            color: #475569;
        }
        
        .data-table tr:hover {
            background-color: #f8fafc;
        }
        
        /* 갤러리 카드 스타일 */
        .gallery-card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border: 1px solid #e2e8f0;
            transition: all 0.2s;
            height: 100%;
        }
        
        .gallery-card:hover {
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        
        .card-title {
            font-size: 1rem;
            font-weight: 500;
            color: #1e293b;
            margin-bottom: 8px;
        }
        
        .card-meta {
            font-size: 0.85rem;
            color: #64748b;
            margin-bottom: 5px;
        }
        
        .card-tags {
            margin-top: 10px;
            margin-bottom: 10px;
        }
        
        .card-tag {
            display: inline-block;
            padding: 3px 8px;
            background-color: #f1f5f9;
            color: #64748b;
            border-radius: 15px;
            margin-right: 5px;
            margin-bottom: 5px;
            font-size: 0.75rem;
        }
        
        .card-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 10px;
        }
        
        .card-status {
            padding: 3px 8px;
            border-radius: 15px;
            font-size: 0.75rem;
            color: white;
        }
        
        /* CRM 스타일 개선 */
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
            color: #1e293b;
            margin-bottom: 8px;
        }
        
        .customer-info {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
        }
        
        .customer-detail {
            font-size: 0.9rem;
            color: #475569;
        }
        
        .price-tag {
            background-color: #ecfdf5;
            color: #047857;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        
        .deadline-tag {
            background-color: #fff1f2;
            color: #be123c;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background-color: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .progress-value {
            height: 100%;
            border-radius: 4px;
        }
        
        /* 채팅 UI 개선 */
        .chat-container {
            display: flex;
            height: 600px;
            background-color: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border: 1px solid #e2e8f0;
            margin-bottom: 20px;
        }
        
        .chat-sidebar {
            width: 280px;
            background-color: #f8fafc;
            border-right: 1px solid #e2e8f0;
            display: flex;
            flex-direction: column;
            height: 100%;
        }
        
        .chat-search {
            padding: 15px;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .chat-search input {
            width: 100%;
            padding: 8px 12px;
            border-radius: 20px;
            border: 1px solid #e2e8f0;
            background-color: white;
            font-size: 0.9rem;
        }
        
        .chat-user {
            padding: 12px 15px;
            cursor: pointer;
            transition: all 0.2s;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .chat-user:hover {
            background-color: #f1f5f9;
        }
        
        .chat-user.active {
            background-color: #e0f2fe;
        }
        
        .chat-user-avatar {
            width: 40px;
            height: 40px;
            background-color: #1e3a8a;
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 500;
            margin-right: 12px;
        }
        
        .chat-user-name {
            font-size: 0.95rem;
            font-weight: 500;
            color: #1e293b;
            margin-bottom: 4px;
        }
        
        .chat-user-preview {
            font-size: 0.8rem;
            color: #64748b;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .chat-user-status {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }
        
        .status-online {
            background-color: #10b981;
        }
        
        .status-offline {
            background-color: #cbd5e1;
        }
        
        .chat-main {
            flex: 1;
            display: flex;
            flex-direction: column;
            height: 100%;
        }
        
        .chat-header {
            padding: 15px;
            border-bottom: 1px solid #e2e8f0;
            display: flex;
            align-items: center;
        }
        
        .chat-avatar {
            width: 40px;
            height: 40px;
            background-color: #1e3a8a;
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 500;
            margin-right: 12px;
        }
        
        .chat-messages {
            flex: 1;
            padding: 15px;
            overflow-y: auto;
            background-color: #f8fafc;
            display: flex;
            flex-direction: column;
        }
        
        .chat-message {
            margin-bottom: 15px;
            max-width: 70%;
            display: flex;
            flex-direction: column;
        }
        
        .chat-message.sent {
            align-self: flex-end;
        }
        
        .chat-message.received {
            align-self: flex-start;
        }
        
        .chat-bubble {
            padding: 12px 15px;
            border-radius: 18px;
            font-size: 0.95rem;
            line-height: 1.4;
            margin-bottom: 4px;
        }
        
        .chat-message.sent .chat-bubble {
            background-color: #1e3a8a;
            color: white;
            border-top-right-radius: 4px;
        }
        
        .chat-message.received .chat-bubble {
            background-color: white;
            color: #1e293b;
            border: 1px solid #e2e8f0;
            border-top-left-radius: 4px;
        }
        
        .chat-time {
            font-size: 0.75rem;
            color: #94a3b8;
            align-self: flex-end;
        }
        
        .file-attachment {
            display: flex;
            align-items: center;
            background-color: white;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 8px 12px;
            margin-bottom: 8px;
        }
        
        .file-icon {
            margin-right: 8px;
            font-size: 1.2rem;
        }
        
        .file-name {
            font-size: 0.85rem;
            color: #1e293b;
        }
        
        .chat-suggestions {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            padding: 10px 15px;
            border-top: 1px solid #e2e8f0;
            background-color: white;
        }
        
        .chat-suggestion {
            background-color: #f1f5f9;
            color: #1e293b;
            padding: 6px 12px;
            border-radius: 16px;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s;
            border: 1px solid #e2e8f0;
        }
        
        .chat-suggestion:hover {
            background-color: #e2e8f0;
        }
        
        .chat-input {
            padding: 15px;
            border-top: 1px solid #e2e8f0;
            background-color: white;
        }
        
        .chat-tools {
            display: flex;
            gap: 15px;
            margin-bottom: 10px;
        }
        
        .chat-tool {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background-color: #f1f5f9;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .chat-tool:hover {
            background-color: #e2e8f0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # 대시보드 헤더
    st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-header">문서 교정 대시보드</div>', unsafe_allow_html=True)
    
    # 탭 생성
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["교정 현황 개요", "상세 분석", "교정 히스토리 갤러리", "고객 관리 CRM", "고객 채팅"])
    
    # 샘플 데이터 생성
    np.random.seed(42)  # 일관된 결과를 위한 시드 설정
    
    # 문서 유형 및 오류 유형 정의
    doc_types = ["이력서", "자기소개서", "논문", "보고서", "제안서", "계약서", "이메일", "블로그 포스트"]
    error_types = ["맞춤법", "문법", "어휘", "문장 구조", "논리적 오류", "일관성", "형식", "참고문헌"]
    
    # 태그 정의
    tags = ["급함", "중요", "VIP고객", "신규", "재의뢰", "할인적용", "영문", "한글", "일문", "중문", "학술", "비즈니스", "기술", "의학", "법률"]
    
    # 고객 이름 생성 (customer_names 정의)
    first_names = ["김", "이", "박", "최", "정", "강", "조", "윤", "장", "임"]
    last_names = ["지훈", "민준", "서연", "지영", "현우", "예은", "도윤", "수빈", "준호", "민지"]
    customer_names = [f"{np.random.choice(first_names)}{np.random.choice(last_names)}" for _ in range(20)]
    
    # 고객 데이터 생성 (CRM용)
    customers = []
    for i in range(50):  # 50명의 고객 생성
        # 이름 생성
        name = np.random.choice(customer_names)
        
        # 회사 생성
        companies = ["삼성전자", "LG전자", "현대자동차", "SK하이닉스", "네이버", "카카오", "쿠팡", "배달의민족", "토스", "당근마켓", "개인"]
        company = np.random.choice(companies)
        
        # 문서 유형
        doc_type = np.random.choice(doc_types)
        
        # 가격 설정 (문서 유형별 다른 범위)
        if doc_type == "이력서":
            price = np.random.randint(50000, 150000)
        elif doc_type == "자기소개서":
            price = np.random.randint(80000, 200000)
        elif doc_type == "논문":
            price = np.random.randint(300000, 800000)
        elif doc_type == "보고서":
            price = np.random.randint(150000, 400000)
        elif doc_type == "제안서":
            price = np.random.randint(200000, 500000)
        elif doc_type == "계약서":
            price = np.random.randint(250000, 600000)
        else:
            price = np.random.randint(50000, 300000)
        
        # 날짜 생성
        entry_date = datetime.now() - timedelta(days=np.random.randint(1, 180))
        deadline_date = entry_date + timedelta(days=np.random.randint(3, 30))
        
        # 상태 설정
        statuses = ["신규", "진행중", "완료", "대기중", "휴면"]
        status = np.random.choice(statuses, p=[0.2, 0.3, 0.3, 0.1, 0.1])
        
        # 작업 수
        work_count = np.random.randint(1, 10)
        
        # 진행률
        progress = np.random.randint(0, 101)
        
        # 이메일 및 연락처
        email = f"{name.lower()}{np.random.randint(100, 999)}@{np.random.choice(['gmail.com', 'naver.com', 'kakao.com', 'daum.net'])}"
        phone = f"010-{np.random.randint(1000, 9999)}-{np.random.randint(1000, 9999)}"
        
        # 태그 설정
        customer_tags = np.random.choice(tags, size=np.random.randint(1, 4), replace=False)
        
        customers.append({
            "name": name,
            "company": company,
            "doc_type": doc_type,
            "price": price,
            "entry_date": entry_date,
            "deadline_date": deadline_date,
            "status": status,
            "work_count": work_count,
            "progress": progress,
            "email": email,
            "phone": phone,
            "tags": customer_tags
        })
    
    # 채팅 메시지 생성 (채팅 인터페이스용)
    chat_messages = {}
    
    # 10명의 고객에 대한 채팅 메시지 생성
    for i in range(10):
        customer_name = customers[i]["name"]
        messages = []
        
        # 각 고객별 3-8개의 메시지 생성
        for j in range(np.random.randint(3, 9)):
            # 메시지 시간 및 날짜
            msg_date = (datetime.now() - timedelta(days=np.random.randint(0, 7))).strftime("%Y-%m-%d")
            msg_time = f"{np.random.randint(9, 19):02d}:{np.random.randint(0, 60):02d}"
            
            # 발신자 (고객 또는 나)
            sender = np.random.choice(["customer", "me"])
            
            # 메시지 내용
            if sender == "customer":
                customer_messages = [
                    f"안녕하세요, {doc_types[i % len(doc_types)]} 교정 부탁드립니다.",
                    "언제쯤 완료될까요?",
                    "수정사항이 있는데 반영 가능할까요?",
                    "감사합니다. 결과물 확인했습니다.",
                    "추가 비용은 얼마인가요?",
                    "오늘 오후까지 가능할까요?",
                    "이 부분은 어떻게 수정하는 게 좋을까요?",
                    "파일 보내드립니다. 확인 부탁드려요."
                ]
                message = np.random.choice(customer_messages)
            else:
                my_messages = [
                    f"{doc_types[i % len(doc_types)]} 교정 요청 확인했습니다.",
                    "내일 오후까지 완료해드리겠습니다.",
                    "수정사항 반영해드렸습니다.",
                    "추가 비용은 5만원입니다.",
                    "오늘 오후 5시까지 완료 예정입니다.",
                    "파일 확인했습니다. 작업 진행하겠습니다.",
                    "수정된 파일 첨부해드립니다.",
                    "문의사항 있으시면 언제든지 말씀해주세요."
                ]
                message = np.random.choice(my_messages)
            
            # 파일 첨부 여부
            has_file = np.random.random() < 0.2  # 20% 확률로 파일 첨부
            msg_data = {
                "sender": sender,
                "message": message,
                "time": msg_time,
                "date": msg_date
            }
            
            if has_file:
                file_types = ["docx", "pdf", "txt", "pptx", "xlsx"]
                file_name = f"문서_{np.random.randint(1000, 9999)}.{np.random.choice(file_types)}"
                msg_data["file"] = file_name
            
            messages.append(msg_data)
        
        chat_messages[customer_name] = messages
    
    # 탭 1: 교정 현황 개요
    with tab1:
        # 주요 지표 행
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">총 교정 문서</div>
                <div class="metric-value">1,890</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">총 발견 오류</div>
                <div class="metric-value">3,373</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">평균 문서당 오류</div>
                <div class="metric-value">1.78</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">평균 교정 시간</div>
                <div class="metric-value">61.7분</div>
            </div>
            """, unsafe_allow_html=True)
        
        # 차트 행 1 - 문서 유형별 교정 현황 및 오류 유형 분포
        st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">문서 유형별 교정 현황</div>', unsafe_allow_html=True)
            
            # 문서 유형별 데이터 생성
            doc_counts = {}
            for doc_type in doc_types:
                doc_counts[doc_type] = np.random.randint(50, 300)
            
            # 데이터프레임 생성
            df_docs = pd.DataFrame({
                "문서 유형": list(doc_counts.keys()),
                "교정 수": list(doc_counts.values())
            })
            
            # 차트 생성
            fig_docs = px.bar(
                df_docs,
                x="문서 유형",
                y="교정 수",
                color="교정 수",
                color_continuous_scale="Blues",
                template="plotly_white"
            )
            
            fig_docs.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                xaxis_title="",
                yaxis_title="교정 수",
                coloraxis_showscale=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Malgun Gothic, Arial", size=12)
            )
            
            st.plotly_chart(fig_docs, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">오류 유형 분포</div>', unsafe_allow_html=True)
            
            # 오류 유형별 데이터 생성
            error_counts = {}
            for error_type in error_types:
                error_counts[error_type] = np.random.randint(100, 600)
            
            # 데이터프레임 생성
            df_errors = pd.DataFrame({
                "오류 유형": list(error_counts.keys()),
                "오류 수": list(error_counts.values())
            })
            
            # 차트 생성 - 파이 차트로 변경하여 겹침 문제 해결
            fig_errors = px.pie(
                df_errors,
                names="오류 유형",
                values="오류 수",
                color_discrete_sequence=px.colors.qualitative.Set3,
                template="plotly_white",
                hole=0.4
            )
            
            fig_errors.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10)
                ),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Malgun Gothic, Arial", size=12)
            )
            
            fig_errors.update_traces(textposition='inside', textinfo='percent')
            
            st.plotly_chart(fig_errors, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # 탭 2: 상세 분석 (태그 기반 필터링으로 수정)
    with tab2:
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        
        # 태그 기반 필터링 UI
        st.markdown('<div class="filter-group">', unsafe_allow_html=True)
        st.markdown('<div class="filter-label">문서 유형</div>', unsafe_allow_html=True)
        
        # 문서 유형 태그 (JavaScript로 선택 상태 토글)
        doc_type_tags_html = ""
        for doc_type in ["전체"] + doc_types:
            selected = " selected" if doc_type == "전체" else ""
            doc_type_tags_html += f'<span class="tag{selected}" onclick="this.classList.toggle(\'selected\')">{doc_type}</span>'
        
        st.markdown(f'<div>{doc_type_tags_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 상태 필터
        st.markdown('<div class="filter-group">', unsafe_allow_html=True)
        st.markdown('<div class="filter-label">상태</div>', unsafe_allow_html=True)
        
        # 상태 태그
        status_tags_html = ""
        for status in ["전체", "완료", "진행중", "대기중"]:
            selected = " selected" if status == "전체" else ""
            status_tags_html += f'<span class="tag{selected}" onclick="this.classList.toggle(\'selected\')">{status}</span>'
        
        st.markdown(f'<div>{status_tags_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 태그 필터
        st.markdown('<div class="filter-group">', unsafe_allow_html=True)
        st.markdown('<div class="filter-label">태그</div>', unsafe_allow_html=True)
        
        # 태그 목록
        tags_html = ""
        for tag in tags:
            tags_html += f'<span class="tag" onclick="this.classList.toggle(\'selected\')">{tag}</span>'
        
        st.markdown(f'<div>{tags_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 날짜 범위 및 필터 버튼
        col1, col2 = st.columns([3, 1])
        
        with col1:
            date_range = st.date_input(
                "기간 선택",
                value=(datetime.now() - timedelta(days=30), datetime.now()),
                max_value=datetime.now()
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            filter_button = st.button("필터 적용", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 필터링된 데이터 생성 (샘플)
        # 실제 구현 시 필터 조건에 따라 데이터를 필터링해야 함
        filtered_data = pd.DataFrame({
            "문서 ID": [f"DOC-{i:04d}" for i in range(1, 101)],  # 100개 데이터 생성
            "문서 유형": np.random.choice(doc_types, 100),
            "제목": [f"샘플 문서 {i}" for i in range(1, 101)],
            "상태": np.random.choice(["완료", "진행중", "대기중"], 100, p=[0.6, 0.3, 0.1]),
            "오류 수": np.random.randint(1, 20, 100),
            "교정 시간(분)": np.random.randint(30, 180, 100),
            "등록일": [(datetime.now() - timedelta(days=np.random.randint(1, 90))).strftime("%Y-%m-%d") for _ in range(100)],
            "태그": [", ".join(np.random.choice(tags, size=np.random.randint(1, 4), replace=False)) for _ in range(100)],
            "담당자": np.random.choice(["김교정", "이수정", "박편집", "최리뷰", "정검토"], 100),
            "고객명": np.random.choice(customer_names, 100),
            "완료일": [(datetime.now() - timedelta(days=np.random.randint(0, 60))).strftime("%Y-%m-%d") for _ in range(100)]
        })
        
        # 필터링된 데이터 차트
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container"><div class="chart-title">문서 유형별 오류 분포</div>', unsafe_allow_html=True)
            
            # 문서 유형별 오류 분포 데이터
            df_type_errors = filtered_data.groupby("문서 유형")["오류 수"].sum().reset_index()
            
            # 문서 유형별 오류 분포 차트
            fig_type_errors = px.bar(
                df_type_errors, 
                x="문서 유형", 
                y="오류 수",
                color="문서 유형",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                template="plotly_white"
            )
            
            fig_type_errors.update_layout(
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                xaxis_title="",
                yaxis_title="오류 수",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Malgun Gothic, Arial", size=12)
            )
            
            st.plotly_chart(fig_type_errors, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="chart-container"><div class="chart-title">태그별 문서 수</div>', unsafe_allow_html=True)
            
            # 태그 데이터 처리 (쉼표로 구분된 태그를 분리하여 카운트)
            all_tags = []
            for tag_list in filtered_data["태그"]:
                all_tags.extend([tag.strip() for tag in tag_list.split(",")])
            
            tag_counts = pd.Series(all_tags).value_counts().reset_index()
            tag_counts.columns = ["태그", "문서 수"]
            
            # 상위 8개 태그만 표시
            tag_counts = tag_counts.head(8)
            
            # 태그별 문서 수 차트
            fig_tag_counts = px.bar(
                tag_counts, 
                x="태그", 
                y="문서 수",
                color="태그",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                template="plotly_white"
            )
            
            fig_tag_counts.update_layout(
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                xaxis_title="",
                yaxis_title="문서 수",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Malgun Gothic, Arial", size=12)
            )
            
            st.plotly_chart(fig_tag_counts, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 필터링된 데이터 테이블
        st.markdown('<div class="chart-container"><div class="chart-title">필터링된 문서 목록</div>', unsafe_allow_html=True)
        st.dataframe(filtered_data, use_container_width=True, height=400)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 데이터 다운로드 버튼
        csv = filtered_data.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="CSV 다운로드",
            data=csv,
            file_name="문서교정_데이터.csv",
            mime="text/csv",
        )
    
    # 탭 3: 교정 히스토리 갤러리 (교정 수 및 태그 필터링 추가)
    with tab3:
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        
        # 갤러리 필터 - 첫 번째 행
        col1, col2 = st.columns(2)
        
        with col1:
            gallery_doc_type = st.selectbox("문서 유형 선택", ["전체"] + doc_types, key="gallery_doc_type")
        
        with col2:
            gallery_sort = st.selectbox("정렬 기준", ["최신순", "오류 많은 순", "교정 시간 긴 순"], key="gallery_sort")
        
        # 갤러리 필터 - 두 번째 행 (교정 수 범위)
        st.markdown('<div class="filter-label">교정 수 범위</div>', unsafe_allow_html=True)
        correction_range = st.slider("", 0, 20, (0, 10), key="correction_range")
        
        # 갤러리 필터 - 세 번째 행 (태그 필터)
        st.markdown('<div class="filter-group">', unsafe_allow_html=True)
        st.markdown('<div class="filter-label">태그 필터</div>', unsafe_allow_html=True)
        
        # 태그 목록
        gallery_tags_html = ""
        for tag in tags:
            gallery_tags_html += f'<span class="tag" onclick="this.classList.toggle(\'selected\')">{tag}</span>'
        
        st.markdown(f'<div>{gallery_tags_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 필터 적용 버튼
        gallery_filter_button = st.button("갤러리 필터 적용", use_container_width=True, key="gallery_filter")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 갤러리 아이템 (샘플)
        st.markdown('<h3 style="font-size: 1.2rem; margin: 20px 0 15px 0;">교정 문서 갤러리</h3>', unsafe_allow_html=True)
        
        # 갤러리 아이템 생성 (샘플 데이터)
        gallery_items = []
        for i in range(24):  # 24개 아이템 생성
            doc_id = f"DOC-{i+1:04d}"
            doc_title = f"샘플 문서 {i+1}"
            doc_type = np.random.choice(doc_types)
            doc_status = np.random.choice(["완료", "진행중", "대기중"])
            doc_errors = np.random.randint(1, 15)
            doc_corrections = np.random.randint(1, 15)
            doc_date = (datetime.now() - timedelta(days=np.random.randint(1, 60))).strftime("%Y-%m-%d")
            doc_tags = np.random.choice(tags, size=np.random.randint(1, 4), replace=False)
            
            gallery_items.append({
                "id": doc_id,
                "title": doc_title,
                "type": doc_type,
                "status": doc_status,
                "errors": doc_errors,
                "corrections": doc_corrections,
                "date": doc_date,
                "tags": doc_tags
            })
        
        # 3x4 그리드로 갤러리 아이템 표시 (한 페이지에 12개)
        for i in range(0, 12, 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(gallery_items):  # 아이템이 있는 경우만 표시
                    with cols[j]:
                        item = gallery_items[i + j]
                        
                        # 상태에 따른 색상 설정
                        status_color = "#4CAF50" if item["status"] == "완료" else "#2196F3" if item["status"] == "진행중" else "#FFC107"
                        
                        # 태그 HTML 생성
                        tags_html = ""  # 태그 HTML 문자열 초기화
                        for tag in item["tags"]:
                            tags_html += f'<span class="card-tag">{tag}</span>'
                        
                        # 카드 HTML 생성
                        st.markdown(f"""
                        <div class="gallery-card">
                            <div class="card-title">{item["title"]}</div>
                            <div class="card-meta">ID: {item["id"]}</div>
                            <div class="card-meta">유형: {item["type"]}</div>
                            <div class="card-meta">교정 수: {item["corrections"]}회</div>
                            <div class="card-tags">{tags_html}</div>
                            <div class="card-footer">
                                <span class="card-status" style="background-color: {status_color};">{item["status"]}</span>
                                <span style="color: #e53935; font-size: 0.85rem;">오류 {item["errors"]}건</span>
                            </div>
                            <div style="color: #888; font-size: 0.8rem; text-align: right; margin-top: 8px;">{item["date"]}</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # 더 보기 버튼
        st.button("더 보기", use_container_width=True, key="load_more")
    
    # 탭 4: 고객 관리 CRM
    with tab4:
        st.markdown('<h2 style="font-size: 1.3rem; margin-bottom: 15px;">고객 관리 CRM</h2>', unsafe_allow_html=True)
        
        # CRM 필터 컨테이너
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        
        # 필터 행
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            crm_doc_type = st.selectbox("문서 유형", ["전체"] + doc_types, key="crm_doc_type")
        
        with col2:
            crm_status = st.selectbox("고객 상태", ["전체", "신규", "진행중", "완료", "대기중", "휴면"], key="crm_status")
        
        with col3:
            crm_sort = st.selectbox("정렬 기준", ["최신순", "마감일순", "금액 높은순", "작업수 많은순"], key="crm_sort")
        
        with col4:
            crm_search = st.text_input("고객명/회사 검색", placeholder="검색어 입력...")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # CRM 대시보드 지표
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">총 고객 수</div>
                <div class="metric-value">127명</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">이번 달 신규 고객</div>
                <div class="metric-value">23명</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">평균 고객 단가</div>
                <div class="metric-value">32.5만원</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">이번 달 매출</div>
                <div class="metric-value">1,245만원</div>
            </div>
            """, unsafe_allow_html=True)
        
        # 고객 목록 및 상세 정보
        st.markdown('<div class="chart-container" style="margin-top: 20px;">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">고객 목록</div>', unsafe_allow_html=True)
        
        # 고객 목록 표시 (최대 10명)
        for i, customer in enumerate(customers[:10]):
            # 진행률 계산
            progress_style = f'width: {customer["progress"]}%;'
            progress_color = "#4CAF50" if customer["progress"] >= 80 else "#2196F3" if customer["progress"] >= 40 else "#FFC107"
            
            # 마감일까지 남은 일수 계산
            days_left = (customer["deadline_date"] - datetime.now()).days
            deadline_class = "deadline-tag" if days_left <= 7 else ""
            deadline_text = f"마감 {days_left}일 전" if days_left > 0 else "마감일 지남"
            
            # 가격 포맷팅
            price_formatted = f"{customer['price']:,}원"
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
            
            # 태그 HTML 생성
tags_html = ""
            for tag in customer["tags"]:
                tags_html += f'<span class="card-tag">{tag}</span>'
            
            # 고객 카드 HTML 생성
            st.markdown(f"""
            <div class="crm-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="customer-name">{customer["name"]} <span style="font-size: 0.8rem; color: #64748b;">({customer["company"]})</span></div>
                    <div>
                        <span class="price-tag">{price_formatted}</span>
                        <span class="{deadline_class}" style="margin-left: 8px;">{deadline_text}</span>
                    </div>
                </div>
                <div class="customer-info" style="margin-top: 8px;">
                    <div class="customer-detail">문서 유형: {customer["doc_type"]}</div>
                    <div class="customer-detail">상태: {customer["status"]}</div>
                    <div class="customer-detail">작업 수: {customer["work_count"]}회</div>
                </div>
                <div class="customer-info" style="margin-top: 5px;">
                    <div class="customer-detail">이메일: {customer["email"]}</div>
                    <div class="customer-detail">연락처: {customer["phone"]}</div>
                </div>
                <div class="card-tags" style="margin-top: 8px;">{tags_html}</div>
                <div style="margin-top: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                        <span style="font-size: 0.85rem; color: #64748b;">진행률</span>
                        <span style="font-size: 0.85rem; color: #64748b;">{customer["progress"]}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-value" style="{progress_style} background-color: {progress_color};"></div>
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 12px;">
                    <div style="color: #64748b; font-size: 0.8rem;">등록일: {customer["entry_date"].strftime("%Y-%m-%d")}</div>
                    <div style="color: #64748b; font-size: 0.8rem;">마감일: {customer["deadline_date"].strftime("%Y-%m-%d")}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # 더 보기 버튼
        st.button("더 많은 고객 보기", use_container_width=True, key="load_more_customers")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 고객 분석 차트
        st.markdown('<div class="chart-container" style="margin-top: 20px;">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">고객 분석</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 문서 유형별 고객 수 및 평균 가격
            # 데이터 준비
            doc_type_customer_data = {}
            for doc_type in doc_types:
                doc_customers = [c for c in customers if c["doc_type"] == doc_type]
                if doc_customers:
                    avg_price = sum(c["price"] for c in doc_customers) / len(doc_customers)
                    doc_type_customer_data[doc_type] = {
                        "count": len(doc_customers),
                        "avg_price": avg_price / 10000  # 만원 단위로 변환
                    }
            
            # 데이터프레임 생성
            df_doc_customers = pd.DataFrame([
                {"문서 유형": dt, "고객 수": data["count"], "평균 가격(만원)": data["avg_price"]}
                for dt, data in doc_type_customer_data.items()
            ])
            
            # 차트 생성
            fig_doc_customers = px.bar(
                df_doc_customers,
                x="문서 유형",
                y="고객 수",
                color="평균 가격(만원)",
                color_continuous_scale="Viridis",
                template="plotly_white"
            )
            
            fig_doc_customers.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                xaxis_title="",
                yaxis_title="고객 수",
                coloraxis_colorbar=dict(title="평균 가격(만원)"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Malgun Gothic, Arial", size=12),
                xaxis=dict(tickangle=-45)
            )
            
            st.plotly_chart(fig_doc_customers, use_container_width=True)
            
        with col2:
            # 월별 신규 고객 추이
            # 데이터 준비
            months = [(datetime.now() - timedelta(days=30*i)).strftime("%Y-%m") for i in range(6)]
            months.reverse()  # 오름차순 정렬
            
            monthly_new_customers = []
            for month in months:
                year, month_num = map(int, month.split("-"))
                month_start = datetime(year, month_num, 1)
                if month_num == 12:
                    month_end = datetime(year + 1, 1, 1) - timedelta(days=1)
                else:
                    month_end = datetime(year, month_num + 1, 1) - timedelta(days=1)
                
                # 해당 월에 등록한 고객 수 계산
                count = sum(1 for c in customers if month_start <= c["entry_date"] <= month_end)
                monthly_new_customers.append(count)
            
            # 데이터프레임 생성
            df_monthly_customers = pd.DataFrame({
                "월": months,
                "신규 고객 수": monthly_new_customers
            })
            
            # 차트 생성
            fig_monthly_customers = px.line(
                df_monthly_customers,
                x="월",
                y="신규 고객 수",
                markers=True,
                template="plotly_white"
            )
            
            fig_monthly_customers.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                xaxis_title="",
                yaxis_title="신규 고객 수",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Malgun Gothic, Arial", size=12)
            )
            
            st.plotly_chart(fig_monthly_customers, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 탭 5: 고객 채팅
    with tab5:
        st.markdown('<h2 style="font-size: 1.3rem; margin-bottom: 15px;">고객 채팅</h2>', unsafe_allow_html=True)
        
        # 채팅 인터페이스
        st.markdown("""
        <div class="chat-container">
            <div class="chat-sidebar">
                <div class="chat-search">
                    <input type="text" placeholder="고객 검색...">
                </div>
                <div style="overflow-y: auto; height: calc(100% - 60px);">
        """, unsafe_allow_html=True)
        
        # 채팅 사이드바 고객 목록
        for i, name in enumerate(chat_messages.keys()):
            # 마지막 메시지 가져오기
            last_message = chat_messages[name][-1]["message"]
            if len(last_message) > 30:
                last_message = last_message[:30] + "..."
            
            # 온라인 상태 (랜덤)
            is_online = np.random.random() < 0.3
            status_class = "status-online" if is_online else "status-offline"
            
            # 활성화 상태 (첫 번째 고객 선택)
            active_class = " active" if i == 0 else ""
            
            # 고객 이름의 첫 글자
            initial = name[0]
            
            st.markdown(f"""
            <div class="chat-user{active_class}">
                <div style="display: flex; align-items: center;">
                    <div class="chat-user-avatar">{initial}</div>
                    <div style="flex: 1;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div class="chat-user-name">{name}</div>
                            <div class="chat-user-status {status_class}"></div>
                        </div>
                        <div class="chat-user-preview">{last_message}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
                </div>
            </div>
            <div class="chat-main">
                <div class="chat-header">
                    <div class="chat-avatar">김</div>
                    <div>
                        <div style="font-weight: 500; color: #1e293b;">김지훈</div>
                        <div style="font-size: 0.8rem; color: #64748b;">삼성전자</div>
                    </div>
                </div>
                <div class="chat-messages">
        """, unsafe_allow_html=True)
        
        # 채팅 메시지 표시 (첫 번째 고객의 메시지)
        first_customer = list(chat_messages.keys())[0]
        for message in chat_messages[first_customer]:
            message_class = "sent" if message["sender"] == "me" else "received"
            
            # 파일 첨부 여부 확인
            file_html = ""
            if "file" in message:
                file_name = message["file"]
                file_ext = file_name.split(".")[-1]
                file_html = f"""
                <div class="file-attachment">
                    <div class="file-icon">📎</div>
                    <div class="file-name">{file_name}</div>
                </div>
                """
            
            st.markdown(f"""
            <div class="chat-message {message_class}">
                <div class="chat-bubble">{message["message"]}</div>
                {file_html}
                <div class="chat-time">{message["time"]} | {message["date"]}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # AI 추천 응답
        st.markdown("""
                <div class="chat-suggestions">
                    <div class="chat-suggestion">네, 확인해보겠습니다.</div>
                    <div class="chat-suggestion">수정 완료되었습니다.</div>
                    <div class="chat-suggestion">추가 비용이 발생할 수 있습니다.</div>
                    <div class="chat-suggestion">첨부파일 확인 부탁드립니다.</div>
                    <div class="chat-suggestion">오늘 오후까지 완료 예정입니다.</div>
                </div>
                </div>
                <div class="chat-input">
                    <div class="chat-tools">
                        <div class="chat-tool">📎</div>
                        <div class="chat-tool">📷</div>
                        <div class="chat-tool">📋</div>
                        <div class="chat-tool">🤖</div>
                    </div>
                    <div style="display: flex; gap: 10px;">
                        <input type="text" placeholder="메시지 입력..." style="flex: 1; padding: 10px; border-radius: 20px; border: 1px solid #e2e8f0;">
                        <button style="background-color: #1e3a8a; color: white; border: none; border-radius: 20px; padding: 0 20px; font-weight: 500;">전송</button>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 채팅 기능 설명
        st.markdown("""
        <div style="margin-top: 20px; padding: 15px; background-color: #f8fafc; border-radius: 10px; border: 1px solid #e2e8f0;">
            <h3 style="font-size: 1.1rem; margin-bottom: 10px; color: #1e3a8a;">채팅 기능 안내</h3>
            <ul style="margin-left: 20px; color: #475569;">
                <li>고객과 실시간 채팅으로 소통할 수 있습니다.</li>
                <li>파일 전송 기능을 통해 문서를 주고받을 수 있습니다.</li>
                <li>AI 추천 응답으로 빠르게 메시지를 작성할 수 있습니다.</li>
                <li>자동완성 기능으로 효율적인 채팅이 가능합니다.</li>
                <li>채팅 내역은 자동으로 저장되어 언제든지 확인할 수 있습니다.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # 실제 채팅 입력이 필요한 경우 고유한 key 추가
        chat_message = st.text_input("메시지 입력", key="crm_chat_input")
        if st.button("전송", key="crm_chat_send"):
            # 메시지 처리 로직
            pass

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
    st.title("PDF 챗봇")
    
    # Ollama 서버 상태 확인
    if not check_ollama_server():
        st.error("Ollama 서버가 실행되고 있지 않습니다.")
        st.info("터미널에서 'ollama serve' 명령어를 실행하여 Ollama 서버를 시작하세요.")
        return
    
    # ll

# 메인 함수 - 이전 메뉴 스타일로 복원

# 페이지 설정 - 모바일 호환성 개선
st.set_page_config(
    page_title="AI 문서 도우미",
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
    
    /* 태그 스타일 */
    .tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-right: 6px;
        margin-bottom: 6px;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .tag.selected {
        background-color: #1e3a8a;
        color: white;
    }
    
    .tag:not(.selected) {
        background-color: #e2e8f0;
        color: #475569;
    }
    
    .tag:hover:not(.selected) {
        background-color: #cbd5e1;
    }
    
    /* 필터 그룹 스타일 */
    .filter-group {
        margin-bottom: 15px;
    }
    
    .filter-label {
        font-size: 0.9rem;
        font-weight: 500;
        color: #334155;
        margin-bottom: 8px;
    }
    
    /* 갤러리 카드 스타일 */
    .gallery-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .gallery-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .card-title {
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 8px;
        color: #1e293b;
    }
    
    .card-meta {
        color: #64748b;
        font-size: 0.85rem;
        margin-bottom: 5px;
    }
    
    .card-tags {
        margin-top: 10px;
        margin-bottom: 10px;
    }
    
    .card-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin-right: 4px;
        background-color: #f1f5f9;
        color: #475569;
    }
    
    .card-status {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        color: white;
    }
    
    .card-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 12px;
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

# 문서교정 대시보드 기능
def render_document_correction_dashboard():
    # CSS 스타일 정의
    st.markdown("""
    <style>
        /* 전체 대시보드 스타일 */
        .dashboard-container {
            padding: 10px;
            background-color: #f8fafc;
            border-radius: 10px;
        }
        
        /* 헤더 스타일 */
        .dashboard-header {
            font-size: 1.5rem;
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #e2e8f0;
        }
        
        /* 탭 스타일 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            white-space: pre-wrap;
            border-radius: 4px;
            font-weight: 500;
            background-color: #f1f5f9;
            color: #64748b;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #1e3a8a !important;
            color: white !important;
        }
        
        /* 메트릭 카드 스타일 */
        .metric-container {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            text-align: center;
            height: 100%;
            border: 1px solid #e2e8f0;
        }
        
        .metric-title {
            font-size: 0.9rem;
            color: #64748b;
            margin-bottom: 8px;
        }
        
        .metric-value {
            font-size: 1.8rem;
            font-weight: 600;
            color: #1e293b;
        }
        
        /* 차트 컨테이너 스타일 */
        .chart-container {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border: 1px solid #e2e8f0;
        }
        
        .chart-title {
            font-size: 1rem;
            font-weight: 500;
            color: #1e293b;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 1px solid #e2e8f0;
        }
        
        /* 필터 컨테이너 스타일 */
        .filter-container {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border: 1px solid #e2e8f0;
        }
        
        .filter-label {
            font-size: 0.9rem;
            font-weight: 500;
            color: #1e293b;
            margin-bottom: 8px;
        }
        
        .filter-group {
            margin-bottom: 15px;
        }
        
        /* 태그 스타일 */
        .tag {
            display: inline-block;
            padding: 5px 10px;
            background-color: #f1f5f9;
            color: #64748b;
            border-radius: 20px;
            margin-right: 8px;
            margin-bottom: 8px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s;
            border: 1px solid #e2e8f0;
        }
        
        .tag:hover {
            background-color: #e2e8f0;
        }
        
        .tag.selected {
            background-color: #1e3a8a;
            color: white;
            border-color: #1e3a8a;
        }
        
        /* 데이터 테이블 스타일 */
        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        
        .data-table th {
            background-color: #f1f5f9;
            padding: 10px;
            text-align: left;
            font-weight: 500;
            color: #1e293b;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .data-table td {
            padding: 10px;
            border-bottom: 1px solid #e2e8f0;
            color: #475569;
        }
        
        .data-table tr:hover {
            background-color: #f8fafc;
        }
        
        /* 갤러리 카드 스타일 */
        .gallery-card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border: 1px solid #e2e8f0;
            transition: all 0.2s;
            height: 100%;
        }
        
        .gallery-card:hover {
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        
        .card-title {
            font-size: 1rem;
            font-weight: 500;
            color: #1e293b;
            margin-bottom: 8px;
        }
        
        .card-meta {
            font-size: 0.85rem;
            color: #64748b;
            margin-bottom: 5px;
        }
        
        .card-tags {
            margin-top: 10px;
            margin-bottom: 10px;
        }
        
        .card-tag {
            display: inline-block;
            padding: 3px 8px;
            background-color: #f1f5f9;
            color: #64748b;
            border-radius: 15px;
            margin-right: 5px;
            margin-bottom: 5px;
            font-size: 0.75rem;
        }
        
        .card-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 10px;
        }
        
        .card-status {
            padding: 3px 8px;
            border-radius: 15px;
            font-size: 0.75rem;
            color: white;
        }
        
        /* CRM 스타일 개선 */
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
            color: #1e293b;
            margin-bottom: 8px;
        }
        
        .customer-info {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
        }
        
        .customer-detail {
            font-size: 0.9rem;
            color: #475569;
        }
        
        .price-tag {
            background-color: #ecfdf5;
            color: #047857;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        
        .deadline-tag {
            background-color: #fff1f2;
            color: #be123c;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background-color: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .progress-value {
            height: 100%;
            border-radius: 4px;
        }
        
        /* 채팅 UI 개선 */
        .chat-container {
            display: flex;
            height: 600px;
            background-color: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border: 1px solid #e2e8f0;
            margin-bottom: 20px;
        }
        
        .chat-sidebar {
            width: 280px;
            background-color: #f8fafc;
            border-right: 1px solid #e2e8f0;
            display: flex;
            flex-direction: column;
            height: 100%;
        }
        
        .chat-search {
            padding: 15px;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .chat-search input {
            width: 100%;
            padding: 8px 12px;
            border-radius: 20px;
            border: 1px solid #e2e8f0;
            background-color: white;
            font-size: 0.9rem;
        }
        
        .chat-user {
            padding: 12px 15px;
            cursor: pointer;
            transition: all 0.2s;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .chat-user:hover {
            background-color: #f1f5f9;
        }
        
        .chat-user.active {
            background-color: #e0f2fe;
        }
        
        .chat-user-avatar {
            width: 40px;
            height: 40px;
            background-color: #1e3a8a;
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 500;
            margin-right: 12px;
        }
        
        .chat-user-name {
            font-size: 0.95rem;
            font-weight: 500;
            color: #1e293b;
            margin-bottom: 4px;
        }
        
        .chat-user-preview {
            font-size: 0.8rem;
            color: #64748b;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .chat-user-status {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }
        
        .status-online {
            background-color: #10b981;
        }
        
        .status-offline {
            background-color: #cbd5e1;
        }
        
        .chat-main {
            flex: 1;
            display: flex;
            flex-direction: column;
            height: 100%;
        }
        
        .chat-header {
            padding: 15px;
            border-bottom: 1px solid #e2e8f0;
            display: flex;
            align-items: center;
        }
        
        .chat-avatar {
            width: 40px;
            height: 40px;
            background-color: #1e3a8a;
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 500;
            margin-right: 12px;
        }
        
        .chat-messages {
            flex: 1;
            padding: 15px;
            overflow-y: auto;
            background-color: #f8fafc;
            display: flex;
            flex-direction: column;
        }
        
        .chat-message {
            margin-bottom: 15px;
            max-width: 70%;
            display: flex;
            flex-direction: column;
        }
        
        .chat-message.sent {
            align-self: flex-end;
        }
        
        .chat-message.received {
            align-self: flex-start;
        }
        
        .chat-bubble {
            padding: 12px 15px;
            border-radius: 18px;
            font-size: 0.95rem;
            line-height: 1.4;
            margin-bottom: 4px;
        }
        
        .chat-message.sent .chat-bubble {
            background-color: #1e3a8a;
            color: white;
            border-top-right-radius: 4px;
        }
        
        .chat-message.received .chat-bubble {
            background-color: white;
            color: #1e293b;
            border: 1px solid #e2e8f0;
            border-top-left-radius: 4px;
        }
        
        .chat-time {
            font-size: 0.75rem;
            color: #94a3b8;
            align-self: flex-end;
        }
        
        .file-attachment {
            display: flex;
            align-items: center;
            background-color: white;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 8px 12px;
            margin-bottom: 8px;
        }
        
        .file-icon {
            margin-right: 8px;
            font-size: 1.2rem;
        }
        
        .file-name {
            font-size: 0.85rem;
            color: #1e293b;
        }
        
        .chat-suggestions {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            padding: 10px 15px;
            border-top: 1px solid #e2e8f0;
            background-color: white;
        }
        
        .chat-suggestion {
            background-color: #f1f5f9;
            color: #1e293b;
            padding: 6px 12px;
            border-radius: 16px;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s;
            border: 1px solid #e2e8f0;
        }
        
        .chat-suggestion:hover {
            background-color: #e2e8f0;
        }
        
        .chat-input {
            padding: 15px;
            border-top: 1px solid #e2e8f0;
            background-color: white;
        }
        
        .chat-tools {
            display: flex;
            gap: 15px;
            margin-bottom: 10px;
        }
        
        .chat-tool {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background-color: #f1f5f9;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .chat-tool:hover {
            background-color: #e2e8f0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # 대시보드 헤더
    st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-header">문서 교정 대시보드</div>', unsafe_allow_html=True)
    
    # 탭 생성
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["교정 현황 개요", "상세 분석", "교정 히스토리 갤러리", "고객 관리 CRM", "고객 채팅"])
    
    # 샘플 데이터 생성
    np.random.seed(42)  # 일관된 결과를 위한 시드 설정
    
    # 문서 유형 및 오류 유형 정의
    doc_types = ["이력서", "자기소개서", "논문", "보고서", "제안서", "계약서", "이메일", "블로그 포스트"]
    error_types = ["맞춤법", "문법", "어휘", "문장 구조", "논리적 오류", "일관성", "형식", "참고문헌"]
    
    # 태그 정의
    tags = ["급함", "중요", "VIP고객", "신규", "재의뢰", "할인적용", "영문", "한글", "일문", "중문", "학술", "비즈니스", "기술", "의학", "법률"]
    
    # 고객 이름 생성 (customer_names 정의)
    first_names = ["김", "이", "박", "최", "정", "강", "조", "윤", "장", "임"]
    last_names = ["지훈", "민준", "서연", "지영", "현우", "예은", "도윤", "수빈", "준호", "민지"]
    customer_names = [f"{np.random.choice(first_names)}{np.random.choice(last_names)}" for _ in range(20)]
    
    # 고객 데이터 생성 (CRM용)
    customers = []
    for i in range(50):  # 50명의 고객 생성
        # 이름 생성
        name = np.random.choice(customer_names)
        
        # 회사 생성
        companies = ["삼성전자", "LG전자", "현대자동차", "SK하이닉스", "네이버", "카카오", "쿠팡", "배달의민족", "토스", "당근마켓", "개인"]
        company = np.random.choice(companies)
        
        # 문서 유형
        doc_type = np.random.choice(doc_types)
        
        # 가격 설정 (문서 유형별 다른 범위)
        if doc_type == "이력서":
            price = np.random.randint(50000, 150000)
        elif doc_type == "자기소개서":
            price = np.random.randint(80000, 200000)
        elif doc_type == "논문":
            price = np.random.randint(300000, 800000)
        elif doc_type == "보고서":
            price = np.random.randint(150000, 400000)
        elif doc_type == "제안서":
            price = np.random.randint(200000, 500000)
        elif doc_type == "계약서":
            price = np.random.randint(250000, 600000)
        else:
            price = np.random.randint(50000, 300000)
        
        # 날짜 생성
        entry_date = datetime.now() - timedelta(days=np.random.randint(1, 180))
        deadline_date = entry_date + timedelta(days=np.random.randint(3, 30))
        
        # 상태 설정
        statuses = ["신규", "진행중", "완료", "대기중", "휴면"]
        status = np.random.choice(statuses, p=[0.2, 0.3, 0.3, 0.1, 0.1])
        
        # 작업 수
        work_count = np.random.randint(1, 10)
        
        # 진행률
        progress = np.random.randint(0, 101)
        
        # 이메일 및 연락처
        email = f"{name.lower()}{np.random.randint(100, 999)}@{np.random.choice(['gmail.com', 'naver.com', 'kakao.com', 'daum.net'])}"
        phone = f"010-{np.random.randint(1000, 9999)}-{np.random.randint(1000, 9999)}"
        
        # 태그 설정
        customer_tags = np.random.choice(tags, size=np.random.randint(1, 4), replace=False)
        
        customers.append({
            "name": name,
            "company": company,
            "doc_type": doc_type,
            "price": price,
            "entry_date": entry_date,
            "deadline_date": deadline_date,
            "status": status,
            "work_count": work_count,
            "progress": progress,
            "email": email,
            "phone": phone,
            "tags": customer_tags
        })
    
    # 채팅 메시지 생성 (채팅 인터페이스용)
    chat_messages = {}
    
    # 10명의 고객에 대한 채팅 메시지 생성
    for i in range(10):
        customer_name = customers[i]["name"]
        messages = []
        
        # 각 고객별 3-8개의 메시지 생성
        for j in range(np.random.randint(3, 9)):
            # 메시지 시간 및 날짜
            msg_date = (datetime.now() - timedelta(days=np.random.randint(0, 7))).strftime("%Y-%m-%d")
            msg_time = f"{np.random.randint(9, 19):02d}:{np.random.randint(0, 60):02d}"
            
            # 발신자 (고객 또는 나)
            sender = np.random.choice(["customer", "me"])
            
            # 메시지 내용
            if sender == "customer":
                customer_messages = [
                    f"안녕하세요, {doc_types[i % len(doc_types)]} 교정 부탁드립니다.",
                    "언제쯤 완료될까요?",
                    "수정사항이 있는데 반영 가능할까요?",
                    "감사합니다. 결과물 확인했습니다.",
                    "추가 비용은 얼마인가요?",
                    "오늘 오후까지 가능할까요?",
                    "이 부분은 어떻게 수정하는 게 좋을까요?",
                    "파일 보내드립니다. 확인 부탁드려요."
                ]
                message = np.random.choice(customer_messages)
            else:
                my_messages = [
                    f"{doc_types[i % len(doc_types)]} 교정 요청 확인했습니다.",
                    "내일 오후까지 완료해드리겠습니다.",
                    "수정사항 반영해드렸습니다.",
                    "추가 비용은 5만원입니다.",
                    "오늘 오후 5시까지 완료 예정입니다.",
                    "파일 확인했습니다. 작업 진행하겠습니다.",
                    "수정된 파일 첨부해드립니다.",
                    "문의사항 있으시면 언제든지 말씀해주세요."
                ]
                message = np.random.choice(my_messages)
            
            # 파일 첨부 여부
            has_file = np.random.random() < 0.2  # 20% 확률로 파일 첨부
            msg_data = {
                "sender": sender,
                "message": message,
                "time": msg_time,
                "date": msg_date
            }
            
            if has_file:
                file_types = ["docx", "pdf", "txt", "pptx", "xlsx"]
                file_name = f"문서_{np.random.randint(1000, 9999)}.{np.random.choice(file_types)}"
                msg_data["file"] = file_name
            
            messages.append(msg_data)
        
        chat_messages[customer_name] = messages
    
    # 탭 1: 교정 현황 개요
    with tab1:
        # 주요 지표 행
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">총 교정 문서</div>
                <div class="metric-value">1,890</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">총 발견 오류</div>
                <div class="metric-value">3,373</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">평균 문서당 오류</div>
                <div class="metric-value">1.78</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">평균 교정 시간</div>
                <div class="metric-value">61.7분</div>
            </div>
            """, unsafe_allow_html=True)
        
        # 차트 행 1 - 문서 유형별 교정 현황 및 오류 유형 분포
        st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">문서 유형별 교정 현황</div>', unsafe_allow_html=True)
            
            # 문서 유형별 데이터 생성
            doc_counts = {}
            for doc_type in doc_types:
                doc_counts[doc_type] = np.random.randint(50, 300)
            
            # 데이터프레임 생성
            df_docs = pd.DataFrame({
                "문서 유형": list(doc_counts.keys()),
                "교정 수": list(doc_counts.values())
            })
            
            # 차트 생성
            fig_docs = px.bar(
                df_docs,
                x="문서 유형",
                y="교정 수",
                color="교정 수",
                color_continuous_scale="Blues",
                template="plotly_white"
            )
            
            fig_docs.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                xaxis_title="",
                yaxis_title="교정 수",
                coloraxis_showscale=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Malgun Gothic, Arial", size=12)
            )
            
            st.plotly_chart(fig_docs, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">오류 유형 분포</div>', unsafe_allow_html=True)
            
            # 오류 유형별 데이터 생성
            error_counts = {}
            for error_type in error_types:
                error_counts[error_type] = np.random.randint(100, 600)
            
            # 데이터프레임 생성
            df_errors = pd.DataFrame({
                "오류 유형": list(error_counts.keys()),
                "오류 수": list(error_counts.values())
            })
            
            # 차트 생성 - 파이 차트로 변경하여 겹침 문제 해결
            fig_errors = px.pie(
                df_errors,
                names="오류 유형",
                values="오류 수",
                color_discrete_sequence=px.colors.qualitative.Set3,
                template="plotly_white",
                hole=0.4
            )
            
            fig_errors.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10)
                ),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Malgun Gothic, Arial", size=12)
            )
            
            fig_errors.update_traces(textposition='inside', textinfo='percent')
            
            st.plotly_chart(fig_errors, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # 탭 2: 상세 분석 (태그 기반 필터링으로 수정)
    with tab2:
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        
        # 태그 기반 필터링 UI
        st.markdown('<div class="filter-group">', unsafe_allow_html=True)
        st.markdown('<div class="filter-label">문서 유형</div>', unsafe_allow_html=True)
        
        # 문서 유형 태그 (JavaScript로 선택 상태 토글)
        doc_type_tags_html = ""
        for doc_type in ["전체"] + doc_types:
            selected = " selected" if doc_type == "전체" else ""
            doc_type_tags_html += f'<span class="tag{selected}" onclick="this.classList.toggle(\'selected\')">{doc_type}</span>'
        
        st.markdown(f'<div>{doc_type_tags_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 상태 필터
        st.markdown('<div class="filter-group">', unsafe_allow_html=True)
        st.markdown('<div class="filter-label">상태</div>', unsafe_allow_html=True)
        
        # 상태 태그
        status_tags_html = ""
        for status in ["전체", "완료", "진행중", "대기중"]:
            selected = " selected" if status == "전체" else ""
            status_tags_html += f'<span class="tag{selected}" onclick="this.classList.toggle(\'selected\')">{status}</span>'
        
        st.markdown(f'<div>{status_tags_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 태그 필터
        st.markdown('<div class="filter-group">', unsafe_allow_html=True)
        st.markdown('<div class="filter-label">태그</div>', unsafe_allow_html=True)
        
        # 태그 목록
        tags_html = ""
        for tag in tags:
            tags_html += f'<span class="tag" onclick="this.classList.toggle(\'selected\')">{tag}</span>'
        
        st.markdown(f'<div>{tags_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 날짜 범위 및 필터 버튼
        col1, col2 = st.columns([3, 1])
        
        with col1:
            date_range = st.date_input(
                "기간 선택",
                value=(datetime.now() - timedelta(days=30), datetime.now()),
                max_value=datetime.now()
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            filter_button = st.button("필터 적용", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 필터링된 데이터 생성 (샘플)
        # 실제 구현 시 필터 조건에 따라 데이터를 필터링해야 함
        filtered_data = pd.DataFrame({
            "문서 ID": [f"DOC-{i:04d}" for i in range(1, 101)],  # 100개 데이터 생성
            "문서 유형": np.random.choice(doc_types, 100),
            "제목": [f"샘플 문서 {i}" for i in range(1, 101)],
            "상태": np.random.choice(["완료", "진행중", "대기중"], 100, p=[0.6, 0.3, 0.1]),
            "오류 수": np.random.randint(1, 20, 100),
            "교정 시간(분)": np.random.randint(30, 180, 100),
            "등록일": [(datetime.now() - timedelta(days=np.random.randint(1, 90))).strftime("%Y-%m-%d") for _ in range(100)],
            "태그": [", ".join(np.random.choice(tags, size=np.random.randint(1, 4), replace=False)) for _ in range(100)],
            "담당자": np.random.choice(["김교정", "이수정", "박편집", "최리뷰", "정검토"], 100),
            "고객명": np.random.choice(customer_names, 100),
            "완료일": [(datetime.now() - timedelta(days=np.random.randint(0, 60))).strftime("%Y-%m-%d") for _ in range(100)]
        })
        
        # 필터링된 데이터 차트
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container"><div class="chart-title">문서 유형별 오류 분포</div>', unsafe_allow_html=True)
            
            # 문서 유형별 오류 분포 데이터
            df_type_errors = filtered_data.groupby("문서 유형")["오류 수"].sum().reset_index()
            
            # 문서 유형별 오류 분포 차트
            fig_type_errors = px.bar(
                df_type_errors, 
                x="문서 유형", 
                y="오류 수",
                color="문서 유형",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                template="plotly_white"
            )
            
            fig_type_errors.update_layout(
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                xaxis_title="",
                yaxis_title="오류 수",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Malgun Gothic, Arial", size=12)
            )
            
            st.plotly_chart(fig_type_errors, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="chart-container"><div class="chart-title">태그별 문서 수</div>', unsafe_allow_html=True)
            
            # 태그 데이터 처리 (쉼표로 구분된 태그를 분리하여 카운트)
            all_tags = []
            for tag_list in filtered_data["태그"]:
                all_tags.extend([tag.strip() for tag in tag_list.split(",")])
            
            tag_counts = pd.Series(all_tags).value_counts().reset_index()
            tag_counts.columns = ["태그", "문서 수"]
            
            # 상위 8개 태그만 표시
            tag_counts = tag_counts.head(8)
            
            # 태그별 문서 수 차트
            fig_tag_counts = px.bar(
                tag_counts, 
                x="태그", 
                y="문서 수",
                color="태그",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                template="plotly_white"
            )
            
            fig_tag_counts.update_layout(
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                xaxis_title="",
                yaxis_title="문서 수",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Malgun Gothic, Arial", size=12)
            )
            
            st.plotly_chart(fig_tag_counts, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 필터링된 데이터 테이블
        st.markdown('<div class="chart-container"><div class="chart-title">필터링된 문서 목록</div>', unsafe_allow_html=True)
        st.dataframe(filtered_data, use_container_width=True, height=400)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 데이터 다운로드 버튼
        csv = filtered_data.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="CSV 다운로드",
            data=csv,
            file_name="문서교정_데이터.csv",
            mime="text/csv",
        )
    
    # 탭 3: 교정 히스토리 갤러리 (교정 수 및 태그 필터링 추가)
    with tab3:
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        
        # 갤러리 필터 - 첫 번째 행
        col1, col2 = st.columns(2)
        
        with col1:
            gallery_doc_type = st.selectbox("문서 유형 선택", ["전체"] + doc_types, key="gallery_doc_type")
        
        with col2:
            gallery_sort = st.selectbox("정렬 기준", ["최신순", "오류 많은 순", "교정 시간 긴 순"], key="gallery_sort")
        
        # 갤러리 필터 - 두 번째 행 (교정 수 범위)
        st.markdown('<div class="filter-label">교정 수 범위</div>', unsafe_allow_html=True)
        correction_range = st.slider("", 0, 20, (0, 10), key="correction_range")
        
        # 갤러리 필터 - 세 번째 행 (태그 필터)
        st.markdown('<div class="filter-group">', unsafe_allow_html=True)
        st.markdown('<div class="filter-label">태그 필터</div>', unsafe_allow_html=True)
        
        # 태그 목록
        gallery_tags_html = ""
        for tag in tags:
            gallery_tags_html += f'<span class="tag" onclick="this.classList.toggle(\'selected\')">{tag}</span>'
        
        st.markdown(f'<div>{gallery_tags_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 필터 적용 버튼
        gallery_filter_button = st.button("갤러리 필터 적용", use_container_width=True, key="gallery_filter")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 갤러리 아이템 (샘플)
        st.markdown('<h3 style="font-size: 1.2rem; margin: 20px 0 15px 0;">교정 문서 갤러리</h3>', unsafe_allow_html=True)
        
        # 갤러리 아이템 생성 (샘플 데이터)
        gallery_items = []
        for i in range(24):  # 24개 아이템 생성
            doc_id = f"DOC-{i+1:04d}"
            doc_title = f"샘플 문서 {i+1}"
            doc_type = np.random.choice(doc_types)
            doc_status = np.random.choice(["완료", "진행중", "대기중"])
            doc_errors = np.random.randint(1, 15)
            doc_corrections = np.random.randint(1, 15)
            doc_date = (datetime.now() - timedelta(days=np.random.randint(1, 60))).strftime("%Y-%m-%d")
            doc_tags = np.random.choice(tags, size=np.random.randint(1, 4), replace=False)
            
            gallery_items.append({
                "id": doc_id,
                "title": doc_title,
                "type": doc_type,
                "status": doc_status,
                "errors": doc_errors,
                "corrections": doc_corrections,
                "date": doc_date,
                "tags": doc_tags
            })
        
        # 3x4 그리드로 갤러리 아이템 표시 (한 페이지에 12개)
        for i in range(0, 12, 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(gallery_items):  # 아이템이 있는 경우만 표시
                    with cols[j]:
                        item = gallery_items[i + j]
                        
                        # 상태에 따른 색상 설정
                        status_color = "#4CAF50" if item["status"] == "완료" else "#2196F3" if item["status"] == "진행중" else "#FFC107"
                        
                        # 태그 HTML 생성
                        tags_html = ""
                        for tag in item["tags"]:
                            tags_html += f'<span class="card-tag">{tag}</span>'
                        
                        # 카드 HTML 생성
                        st.markdown(f"""
                        <div class="gallery-card">
                            <div class="card-title">{item["title"]}</div>
                            <div class="card-meta">ID: {item["id"]}</div>
                            <div class="card-meta">유형: {item["type"]}</div>
                            <div class="card-meta">교정 수: {item["corrections"]}회</div>
                            <div class="card-tags">{tags_html}</div>
                            <div class="card-footer">
                                <span class="card-status" style="background-color: {status_color};">{item["status"]}</span>
                                <span style="color: #e53935; font-size: 0.85rem;">오류 {item["errors"]}건</span>
                            </div>
                            <div style="color: #888; font-size: 0.8rem; text-align: right; margin-top: 8px;">{item["date"]}</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # 더 보기 버튼
        st.button("더 보기", use_container_width=True, key="load_more")
    
    # 탭 4: 고객 관리 CRM
    with tab4:
        st.markdown('<h2 style="font-size: 1.3rem; margin-bottom: 15px;">고객 관리 CRM</h2>', unsafe_allow_html=True)
        
        # CRM 필터 컨테이너
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        
        # 필터 행
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            crm_doc_type = st.selectbox("문서 유형", ["전체"] + doc_types, key="crm_doc_type")
        
        with col2:
            crm_status = st.selectbox("고객 상태", ["전체", "신규", "진행중", "완료", "대기중", "휴면"], key="crm_status")
        
        with col3:
            crm_sort = st.selectbox("정렬 기준", ["최신순", "마감일순", "금액 높은순", "작업수 많은순"], key="crm_sort")
        
        with col4:
            crm_search = st.text_input("고객명/회사 검색", placeholder="검색어 입력...")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # CRM 대시보드 지표
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">총 고객 수</div>
                <div class="metric-value">127명</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">이번 달 신규 고객</div>
                <div class="metric-value">23명</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">평균 고객 단가</div>
                <div class="metric-value">32.5만원</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">이번 달 매출</div>
                <div class="metric-value">1,245만원</div>
            </div>
            """, unsafe_allow_html=True)
        
        # 고객 목록 및 상세 정보
        st.markdown('<div class="chart-container" style="margin-top: 20px;">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">고객 목록</div>', unsafe_allow_html=True)
        
        # 고객 목록 표시 (최대 10명)
        for i, customer in enumerate(customers[:10]):
            # 진행률 계산
            progress_style = f'width: {customer["progress"]}%;'
            progress_color = "#4CAF50" if customer["progress"] >= 80 else "#2196F3" if customer["progress"] >= 40 else "#FFC107"
            
            # 마감일까지 남은 일수 계산
            days_left = (customer["deadline_date"] - datetime.now()).days
            deadline_class = "deadline-tag" if days_left <= 7 else ""
            deadline_text = f"마감 {days_left}일 전" if days_left > 0 else "마감일 지남"
            
            # 가격 포맷팅
            price_formatted = f"{customer['price']:,}원"
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
            
            # 태그 HTML 생성
            tags_html = ""
            for tag in customer["tags"]:
                tags_html += f'<span class="card-tag">{tag}</span>'
            
            # 고객 카드 HTML 생성
            st.markdown(f"""
            <div class="crm-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="customer-name">{customer["name"]} <span style="font-size: 0.8rem; color: #64748b;">({customer["company"]})</span></div>
                    <div>
                        <span class="price-tag">{price_formatted}</span>
                        <span class="{deadline_class}" style="margin-left: 8px;">{deadline_text}</span>
                    </div>
                </div>
                <div class="customer-info" style="margin-top: 8px;">
                    <div class="customer-detail">문서 유형: {customer["doc_type"]}</div>
                    <div class="customer-detail">상태: {customer["status"]}</div>
                    <div class="customer-detail">작업 수: {customer["work_count"]}회</div>
                </div>
                <div class="customer-info" style="margin-top: 5px;">
                    <div class="customer-detail">이메일: {customer["email"]}</div>
                    <div class="customer-detail">연락처: {customer["phone"]}</div>
                </div>
                <div class="card-tags" style="margin-top: 8px;">{tags_html}</div>
                <div style="margin-top: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                        <span style="font-size: 0.85rem; color: #64748b;">진행률</span>
                        <span style="font-size: 0.85rem; color: #64748b;">{customer["progress"]}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-value" style="{progress_style} background-color: {progress_color};"></div>
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 12px;">
                    <div style="color: #64748b; font-size: 0.8rem;">등록일: {customer["entry_date"].strftime("%Y-%m-%d")}</div>
                    <div style="color: #64748b; font-size: 0.8rem;">마감일: {customer["deadline_date"].strftime("%Y-%m-%d")}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # 더 보기 버튼
        st.button("더 많은 고객 보기", use_container_width=True, key="load_more_customers")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 고객 분석 차트
        st.markdown('<div class="chart-container" style="margin-top: 20px;">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">고객 분석</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 문서 유형별 고객 수 및 평균 가격
            # 데이터 준비
            doc_type_customer_data = {}
            for doc_type in doc_types:
                doc_customers = [c for c in customers if c["doc_type"] == doc_type]
                if doc_customers:
                    avg_price = sum(c["price"] for c in doc_customers) / len(doc_customers)
                    doc_type_customer_data[doc_type] = {
                        "count": len(doc_customers),
                        "avg_price": avg_price / 10000  # 만원 단위로 변환
                    }
            
            # 데이터프레임 생성
            df_doc_customers = pd.DataFrame([
                {"문서 유형": dt, "고객 수": data["count"], "평균 가격(만원)": data["avg_price"]}
                for dt, data in doc_type_customer_data.items()
            ])
            
            # 차트 생성
            fig_doc_customers = px.bar(
                df_doc_customers,
                x="문서 유형",
                y="고객 수",
                color="평균 가격(만원)",
                color_continuous_scale="Viridis",
                template="plotly_white"
            )
            
            fig_doc_customers.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                xaxis_title="",
                yaxis_title="고객 수",
                coloraxis_colorbar=dict(title="평균 가격(만원)"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Malgun Gothic, Arial", size=12),
                xaxis=dict(tickangle=-45)
            )
            
            st.plotly_chart(fig_doc_customers, use_container_width=True)
            
        with col2:
            # 월별 신규 고객 추이
            # 데이터 준비
            months = [(datetime.now() - timedelta(days=30*i)).strftime("%Y-%m") for i in range(6)]
            months.reverse()  # 오름차순 정렬
            
            monthly_new_customers = []
            for month in months:
                year, month_num = map(int, month.split("-"))
                month_start = datetime(year, month_num, 1)
                if month_num == 12:
                    month_end = datetime(year + 1, 1, 1) - timedelta(days=1)
                else:
                    month_end = datetime(year, month_num + 1, 1) - timedelta(days=1)
                
                # 해당 월에 등록한 고객 수 계산
                count = sum(1 for c in customers if month_start <= c["entry_date"] <= month_end)
                monthly_new_customers.append(count)
            
            # 데이터프레임 생성
            df_monthly_customers = pd.DataFrame({
                "월": months,
                "신규 고객 수": monthly_new_customers
            })
            
            # 차트 생성
            fig_monthly_customers = px.line(
                df_monthly_customers,
                x="월",
                y="신규 고객 수",
                markers=True,
                template="plotly_white"
            )
            
            fig_monthly_customers.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                xaxis_title="",
                yaxis_title="신규 고객 수",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Malgun Gothic, Arial", size=12)
            )
            
            st.plotly_chart(fig_monthly_customers, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 탭 5: 고객 채팅
    with tab5:
        st.markdown('<h2 style="font-size: 1.3rem; margin-bottom: 15px;">고객 채팅</h2>', unsafe_allow_html=True)
        
        # 채팅 인터페이스
        st.markdown("""
        <div class="chat-container">
            <div class="chat-sidebar">
                <div class="chat-search">
                    <input type="text" placeholder="고객 검색...">
                </div>
                <div style="overflow-y: auto; height: calc(100% - 60px);">
        """, unsafe_allow_html=True)
        
        # 채팅 사이드바 고객 목록
        for i, name in enumerate(chat_messages.keys()):
            # 마지막 메시지 가져오기
            last_message = chat_messages[name][-1]["message"]
            if len(last_message) > 30:
                last_message = last_message[:30] + "..."
            
            # 온라인 상태 (랜덤)
            is_online = np.random.random() < 0.3
            status_class = "status-online" if is_online else "status-offline"
            
            # 활성화 상태 (첫 번째 고객 선택)
            active_class = " active" if i == 0 else ""
            
            # 고객 이름의 첫 글자
            initial = name[0]
            
            st.markdown(f"""
            <div class="chat-user{active_class}">
                <div style="display: flex; align-items: center;">
                    <div class="chat-user-avatar">{initial}</div>
                    <div style="flex: 1;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div class="chat-user-name">{name}</div>
                            <div class="chat-user-status {status_class}"></div>
                        </div>
                        <div class="chat-user-preview">{last_message}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
                </div>
            </div>
            <div class="chat-main">
                <div class="chat-header">
                    <div class="chat-avatar">김</div>
                    <div>
                        <div style="font-weight: 500; color: #1e293b;">김지훈</div>
                        <div style="font-size: 0.8rem; color: #64748b;">삼성전자</div>
                    </div>
                </div>
                <div class="chat-messages">
        """, unsafe_allow_html=True)
        
        # 채팅 메시지 표시 (첫 번째 고객의 메시지)
        first_customer = list(chat_messages.keys())[0]
        for message in chat_messages[first_customer]:
            message_class = "sent" if message["sender"] == "me" else "received"
            
            # 파일 첨부 여부 확인
            file_html = ""
            if "file" in message:
                file_name = message["file"]
                file_ext = file_name.split(".")[-1]
                file_html = f"""
                <div class="file-attachment">
                    <div class="file-icon">📎</div>
                    <div class="file-name">{file_name}</div>
                </div>
                """
            
            st.markdown(f"""
            <div class="chat-message {message_class}">
                <div class="chat-bubble">{message["message"]}</div>
                {file_html}
                <div class="chat-time">{message["time"]} | {message["date"]}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # AI 추천 응답
        st.markdown("""
                <div class="chat-suggestions">
                    <div class="chat-suggestion">네, 확인해보겠습니다.</div>
                    <div class="chat-suggestion">수정 완료되었습니다.</div>
                    <div class="chat-suggestion">추가 비용이 발생할 수 있습니다.</div>
                    <div class="chat-suggestion">첨부파일 확인 부탁드립니다.</div>
                    <div class="chat-suggestion">오늘 오후까지 완료 예정입니다.</div>
                </div>
                </div>
                <div class="chat-input">
                    <div class="chat-tools">
                        <div class="chat-tool">📎</div>
                        <div class="chat-tool">📷</div>
                        <div class="chat-tool">📋</div>
                        <div class="chat-tool">🤖</div>
                    </div>
                    <div style="display: flex; gap: 10px;">
                        <input type="text" placeholder="메시지 입력..." style="flex: 1; padding: 10px; border-radius: 20px; border: 1px solid #e2e8f0;">
                        <button style="background-color: #1e3a8a; color: white; border: none; border-radius: 20px; padding: 0 20px; font-weight: 500;">전송</button>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 채팅 기능 설명
        st.markdown("""
        <div style="margin-top: 20px; padding: 15px; background-color: #f8fafc; border-radius: 10px; border: 1px solid #e2e8f0;">
            <h3 style="font-size: 1.1rem; margin-bottom: 10px; color: #1e3a8a;">채팅 기능 안내</h3>
            <ul style="margin-left: 20px; color: #475569;">
                <li>고객과 실시간 채팅으로 소통할 수 있습니다.</li>
                <li>파일 전송 기능을 통해 문서를 주고받을 수 있습니다.</li>
                <li>AI 추천 응답으로 빠르게 메시지를 작성할 수 있습니다.</li>
                <li>자동완성 기능으로 효율적인 채팅이 가능합니다.</li>
                <li>채팅 내역은 자동으로 저장되어 언제든지 확인할 수 있습니다.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # 실제 채팅 입력이 필요한 경우 고유한 key 추가
        chat_message = st.text_input("메시지 입력", key="crm_chat_input")
        if st.button("전송", key="crm_chat_send"):
            # 메시지 처리 로직
            pass

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
    st.title("PDF 챗봇")
    
    # Ollama 서버 상태 확인
    if not check_ollama_server():
        st.error("Ollama 서버가 실행되고 있지 않습니다.")
        st.info("터미널에서 'ollama serve' 명령어를 실행하여 Ollama 서버를 시작하세요.")
        return
    
    # ll

# 메인 함수 - 이전 메뉴 스타일로 복원

# 페이지 설정 - 모바일 호환성 개선
st.set_page_config(
    page_title="AI 문서 도우미",
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
    
    /* 태그 스타일 */
    .tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-right: 6px;
        margin-bottom: 6px;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .tag.selected {
        background-color: #1e3a8a;
        color: white;
    }
    
    .tag:not(.selected) {
        background-color: #e2e8f0;
        color: #475569;
    }
    
    .tag:hover:not(.selected) {
        background-color: #cbd5e1;
    }
    
    /* 필터 그룹 스타일 */
    .filter-group {
        margin-bottom: 15px;
    }
    
    .filter-label {
        font-size: 0.9rem;
        font-weight: 500;
        color: #334155;
        margin-bottom: 8px;
    }
    
    /* 갤러리 카드 스타일 */
    .gallery-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .gallery-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .card-title {
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 8px;
        color: #1e293b;
    }
    
    .card-meta {
        color: #64748b;
        font-size: 0.85rem;
        margin-bottom: 5px;
    }
    
    .card-tags {
        margin-top: 10px;
        margin-bottom: 10px;
    }
    
    .card-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin-right: 4px;
        background-color: #f1f5f9;
        color: #475569;
    }
    
    .card-status {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        color: white;
    }
    
    .card-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 12px;
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

# 문서교정 대시보드 기능
def render_document_correction_dashboard():
    # CSS 스타일 정의
    st.markdown("""
    <style>
        /* 전체 대시보드 스타일 */
        .dashboard-container {
            padding: 10px;
            background-color: #f8fafc;
            border-radius: 10px;
        }
        
        /* 헤더 스타일 */
        .dashboard-header {
            font-size: 1.5rem;
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #e2e8f0;
        }
        
        /* 탭 스타일 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            white-space: pre-wrap;
            border-radius: 4px;
            font-weight: 500;
            background-color: #f1f5f9;
            color: #64748b;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #1e3a8a !important;
            color: white !important;
        }
        
        /* 메트릭 카드 스타일 */
        .metric-container {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            text-align: center;
            height: 100%;
            border: 1px solid #e2e8f0;
        }
        
        .metric-title {
            font-size: 0.9rem;
            color: #64748b;
            margin-bottom: 8px;
        }
        
        .metric-value {
            font-size: 1.8rem;
            font-weight: 600;
            color: #1e293b;
        }
        
        /* 차트 컨테이너 스타일 */
        .chart-container {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border: 1px solid #e2e8f0;
        }
        
        .chart-title {
            font-size: 1rem;
            font-weight: 500;
            color: #1e293b;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 1px solid #e2e8f0;
        }
        
        /* 필터 컨테이너 스타일 */
        .filter-container {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border: 1px solid #e2e8f0;
        }
        
        .filter-label {
            font-size: 0.9rem;
            font-weight: 500;
            color: #1e293b;
            margin-bottom: 8px;
        }
        
        .filter-group {
            margin-bottom: 15px;
        }
        
        /* 태그 스타일 */
        .tag {
            display: inline-block;
            padding: 5px 10px;
            background-color: #f1f5f9;
            color: #64748b;
            border-radius: 20px;
            margin-right: 8px;
            margin-bottom: 8px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s;
            border: 1px solid #e2e8f0;
        }
        
        .tag:hover {
            background-color: #e2e8f0;
        }
        
        .tag.selected {
            background-color: #1e3a8a;
            color: white;
            border-color: #1e3a8a;
        }
        
        /* 데이터 테이블 스타일 */
        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        
        .data-table th {
            background-color: #f1f5f9;
            padding: 10px;
            text-align: left;
            font-weight: 500;
            color: #1e293b;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .data-table td {
            padding: 10px;
            border-bottom: 1px solid #e2e8f0;
            color: #475569;
        }
        
        .data-table tr:hover {
            background-color: #f8fafc;
        }
        
        /* 갤러리 카드 스타일 */
        .gallery-card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border: 1px solid #e2e8f0;
            transition: all 0.2s;
            height: 100%;
        }
        
        .gallery-card:hover {
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        
        .card-title {
            font-size: 1rem;
            font-weight: 500;
            color: #1e293b;
            margin-bottom: 8px;
        }
        
        .card-meta {
            font-size: 0.85rem;
            color: #64748b;
            margin-bottom: 5px;
        }
        
        .card-tags {
            margin-top: 10px;
            margin-bottom: 10px;
        }
        
        .card-tag {
            display: inline-block;
            padding: 3px 8px;
            background-color: #f1f5f9;
            color: #64748b;
            border-radius: 15px;
            margin-right: 5px;
            margin-bottom: 5px;
            font-size: 0.75rem;
        }
        
        .card-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 10px;
        }
        
        .card-status {
            padding: 3px 8px;
            border-radius: 15px;
            font-size: 0.75rem;
            color: white;
        }
        
        /* CRM 스타일 개선 */
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
import requests
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pypdf import PdfReader

# 대시보드 기능 가져오기
try:
    from correction_dashboard import render_document_correction_dashboard
except ImportError:
    st.error("correction_dashboard.py 파일이 필요합니다.")

# 페이지 설정 - 모바일 호환성 개선
st.set_page_config(
    page_title="AI 문서 도우미",
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
    
    /* 태그 스타일 */
    .tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-right: 6px;
        margin-bottom: 6px;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .tag.selected {
        background-color: #1e3a8a;
        color: white;
    }
    
    .tag:not(.selected) {
        background-color: #e2e8f0;
        color: #475569;
    }
    
    .tag:hover:not(.selected) {
        background-color: #cbd5e1;
    }
    
    /* 필터 그룹 스타일 */
    .filter-group {
        margin-bottom: 15px;
    }
    
    .filter-label {
        font-size: 0.9rem;
        font-weight: 500;
        color: #334155;
        margin-bottom: 8px;
    }
    
    /* 갤러리 카드 스타일 */
    .gallery-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .gallery-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .card-title {
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 8px;
        color: #1e293b;
    }
    
    .card-meta {
        color: #64748b;
        font-size: 0.85rem;
        margin-bottom: 5px;
    }
    
    .card-tags {
        margin-top: 10px;
        margin-bottom: 10px;
    }
    
    .card-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin-right: 4px;
        background-color: #f1f5f9;
        color: #475569;
    }
    
    .card-status {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        color: white;
    }
    
    .card-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 12px;
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

# 문서교정 대시보드 기능
def render_document_correction_dashboard():
    # CSS 스타일 정의
    st.markdown("""
    <style>
        /* 전체 대시보드 스타일 */
        .dashboard-container {
            padding: 10px;
            background-color: #f8fafc;
            border-radius: 10px;
        }
        
        /* 헤더 스타일 */
        .dashboard-header {
            font-size: 1.5rem;
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #e2e8f0;
        }
        
        /* 탭 스타일 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            white-space: pre-wrap;
            border-radius: 4px;
            font-weight: 500;
            background-color: #f1f5f9;
            color: #64748b;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #1e3a8a !important;
            color: white !important;
        }
        
        /* 메트릭 카드 스타일 */
        .metric-container {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            text-align: center;
            height: 100%;
            border: 1px solid #e2e8f0;
        }
        
        .metric-title {
            font-size: 0.9rem;
            color: #64748b;
            margin-bottom: 8px;
        }
        
        .metric-value {
            font-size: 1.8rem;
            font-weight: 600;
            color: #1e293b;
        }
        
        /* 차트 컨테이너 스타일 */
        .chart-container {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border: 1px solid #e2e8f0;
        }
        
        .chart-title {
            font-size: 1rem;
            font-weight: 500;
            color: #1e293b;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 1px solid #e2e8f0;
        }
        
        /* 필터 컨테이너 스타일 */
        .filter-container {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border: 1px solid #e2e8f0;
        }
        
        .filter-label {
            font-size: 0.9rem;
            font-weight: 500;
            color: #1e293b;
            margin-bottom: 8px;
        }
        
        .filter-group {
            margin-bottom: 15px;
        }
        
        /* 태그 스타일 */
        .tag {
            display: inline-block;
            padding: 5px 10px;
            background-color: #f1f5f9;
            color: #64748b;
            border-radius: 20px;
            margin-right: 8px;
            margin-bottom: 8px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s;
            border: 1px solid #e2e8f0;
        }
        
        .tag:hover {
            background-color: #e2e8f0;
        }
        
        .tag.selected {
            background-color: #1e3a8a;
            color: white;
            border-color: #1e3a8a;
        }
        
        /* 데이터 테이블 스타일 */
        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        
        .data-table th {
            background-color: #f1f5f9;
            padding: 10px;
            text-align: left;
            font-weight: 500;
            color: #1e293b;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .data-table td {
            padding: 10px;
            border-bottom: 1px solid #e2e8f0;
            color: #475569;
        }
        
        .data-table tr:hover {
            background-color: #f8fafc;
        }
        
        /* 갤러리 카드 스타일 */
        .gallery-card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border: 1px solid #e2e8f0;
            transition: all 0.2s;
            height: 100%;
        }
        
        .gallery-card:hover {
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        
        .card-title {
            font-size: 1rem;
            font-weight: 500;
            color: #1e293b;
            margin-bottom: 8px;
        }
        
        .card-meta {
            font-size: 0.85rem;
            color: #64748b;
            margin-bottom: 5px;
        }
        
        .card-tags {
            margin-top: 10px;
            margin-bottom: 10px;
        }
        
        .card-tag {
            display: inline-block;
            padding: 3px 8px;
            background-color: #f1f5f9;
            color: #64748b;
            border-radius: 15px;
            margin-right: 5px;
            margin-bottom: 5px;
            font-size: 0.75rem;
        }
        
        .card-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 10px;
        }
        
        .card-status {
            padding: 3px 8px;
            border-radius: 15px;
            font-size: 0.75rem;
            color: white;
        }
        
        /* CRM 스타일 개선 */
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
            color: #1e293b;
            margin-bottom: 8px;
        }
        
        .customer-info {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
        }
        
        .customer-detail {
            font-size: 0.9rem;
            color: #475569;
        }
        
        .price-tag {
            background-color: #ecfdf5;
            color: #047857;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        
        .deadline-tag {
            background-color: #fff1f2;
            color: #be123c;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background-color: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .progress-value {
            height: 100%;
            border-radius: 4px;
        }
        
        /* 채팅 UI 개선 */
        .chat-container {
            display: flex;
            height: 600px;
            background-color: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border: 1px solid #e2e8f0;
            margin-bottom: 20px;
        }
        
        .chat-sidebar {
            width: 280px;
            background-color: #f8fafc;
            border-right: 1px solid #e2e8f0;
            display: flex;
            flex-direction: column;
            height: 100%;
        }
        
        .chat-search {
            padding: 15px;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .chat-search input {
            width: 100%;
            padding: 8px 12px;
            border-radius: 20px;
            border: 1px solid #e2e8f0;
            background-color: white;
            font-size: 0.9rem;
        }
        
        .chat-user {
            padding: 12px 15px;
            cursor: pointer;
            transition: all 0.2s;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .chat-user:hover {
            background-color: #f1f5f9;
        }
        
        .chat-user.active {
            background-color: #e0f2fe;
        }
        
        .chat-user-avatar {
            width: 40px;
            height: 40px;
            background-color: #1e3a8a;
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 500;
            margin-right: 12px;
        }
        
        .chat-user-name {
            font-size: 0.95rem;
            font-weight: 500;
            color: #1e293b;
            margin-bottom: 4px;
        }
        
        .chat-user-preview {
            font-size: 0.8rem;
            color: #64748b;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .chat-user-status {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }
        
        .status-online {
            background-color: #10b981;
        }
        
        .status-offline {
            background-color: #cbd5e1;
        }
        
        .chat-main {
            flex: 1;
            display: flex;
            flex-direction: column;
            height: 100%;
        }
        
        .chat-header {
            padding: 15px;
            border-bottom: 1px solid #e2e8f0;
            display: flex;
            align-items: center;
        }
        
        .chat-avatar {
            width: 40px;
            height: 40px;
            background-color: #1e3a8a;
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 500;
            margin-right: 12px;
        }
        
        .chat-messages {
            flex: 1;
            padding: 15px;
            overflow-y: auto;
            background-color: #f8fafc;
            display: flex;
            flex-direction: column;
        }
        
        .chat-message {
            margin-bottom: 15px;
            max-width: 70%;
            display: flex;
            flex-direction: column;
        }
        
        .chat-message.sent {
            align-self: flex-end;
        }
        
        .chat-message.received {
            align-self: flex-start;
        }
        
        .chat-bubble {
            padding: 12px 15px;
            border-radius: 18px;
            font-size: 0.95rem;
            line-height: 1.4;
            margin-bottom: 4px;
        }
        
        .chat-message.sent .chat-bubble {
            background-color: #1e3a8a;
            color: white;
            border-top-right-radius: 4px;
        }
        
        .chat-message.received .chat-bubble {
            background-color: white;
            color: #1e293b;
            border: 1px solid #e2e8f0;
            border-top-left-radius: 4px;
        }
        
        .chat-time {
            font-size: 0.75rem;
            color: #94a3b8;
            align-self: flex-end;
        }
        
        .file-attachment {
            display: flex;
            align-items: center;
            background-color: white;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 8px 12px;
            margin-bottom: 8px;
        }
        
        .file-icon {
            margin-right: 8px;
            font-size: 1.2rem;
        }
        
        .file-name {
            font-size: 0.85rem;
            color: #1e293b;
        }
        
        .chat-suggestions {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            padding: 10px 15px;
            border-top: 1px solid #e2e8f0;
            background-color: white;
        }
        
        .chat-suggestion {
            background-color: #f1f5f9;
            color: #1e293b;
            padding: 6px 12px;
            border-radius: 16px;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s;
            border: 1px solid #e2e8f0;
        }
        
        .chat-suggestion:hover {
            background-color: #e2e8f0;
        }
        
        .chat-input {
            padding: 15px;
            border-top: 1px solid #e2e8f0;
            background-color: white;
        }
        
        .chat-tools {
            display: flex;
            gap: 15px;
            margin-bottom: 10px;
        }
        
        .chat-tool {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background-color: #f1f5f9;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .chat-tool:hover {
            background-color: #e2e8f0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # 대시보드 헤더
    st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-header">문서 교정 대시보드</div>', unsafe_allow_html=True)
    
    # 탭 생성
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["교정 현황 개요", "상세 분석", "교정 히스토리 갤러리", "고객 관리 CRM", "고객 채팅"])
    
    # 샘플 데이터 생성
    np.random.seed(42)  # 일관된 결과를 위한 시드 설정
    
    # 문서 유형 및 오류 유형 정의
    doc_types = ["이력서", "자기소개서", "논문", "보고서", "제안서", "계약서", "이메일", "블로그 포스트"]
    error_types = ["맞춤법", "문법", "어휘", "문장 구조", "논리적 오류", "일관성", "형식", "참고문헌"]
    
    # 태그 정의
    tags = ["급함", "중요", "VIP고객", "신규", "재의뢰", "할인적용", "영문", "한글", "일문", "중문", "학술", "비즈니스", "기술", "의학", "법률"]
    
    # 고객 이름 생성 (customer_names 정의)
    first_names = ["김", "이", "박", "최", "정", "강", "조", "윤", "장", "임"]
    last_names = ["지훈", "민준", "서연", "지영", "현우", "예은", "도윤", "수빈", "준호", "민지"]
    customer_names = [f"{np.random.choice(first_names)}{np.random.choice(last_names)}" for _ in range(20)]
    
    # 고객 데이터 생성 (CRM용)
    customers = []
    for i in range(50):  # 50명의 고객 생성
        # 이름 생성
        name = np.random.choice(customer_names)
        
        # 회사 생성
        companies = ["삼성전자", "LG전자", "현대자동차", "SK하이닉스", "네이버", "카카오", "쿠팡", "배달의민족", "토스", "당근마켓", "개인"]
        company = np.random.choice(companies)
        
        # 문서 유형
        doc_type = np.random.choice(doc_types)
        
        # 가격 설정 (문서 유형별 다른 범위)
        if doc_type == "이력서":
            price = np.random.randint(50000, 150000)
        elif doc_type == "자기소개서":
            price = np.random.randint(80000, 200000)
        elif doc_type == "논문":
            price = np.random.randint(300000, 800000)
        elif doc_type == "보고서":
            price = np.random.randint(150000, 400000)
        elif doc_type == "제안서":
            price = np.random.randint(200000, 500000)
        elif doc_type == "계약서":
            price = np.random.randint(250000, 600000)
        else:
            price = np.random.randint(50000, 300000)
        
        # 날짜 생성
        entry_date = datetime.now() - timedelta(days=np.random.randint(1, 180))
        deadline_date = entry_date + timedelta(days=np.random.randint(3, 30))
        
        # 상태 설정
        statuses = ["신규", "진행중", "완료", "대기중", "휴면"]
        status = np.random.choice(statuses, p=[0.2, 0.3, 0.3, 0.1, 0.1])
        
        # 작업 수
        work_count = np.random.randint(1, 10)
        
        # 진행률
        progress = np.random.randint(0, 101)
        
        # 이메일 및 연락처
        email = f"{name.lower()}{np.random.randint(100, 999)}@{np.random.choice(['gmail.com', 'naver.com', 'kakao.com', 'daum.net'])}"
        phone = f"010-{np.random.randint(1000, 9999)}-{np.random.randint(1000, 9999)}"
        
        # 태그 설정
        customer_tags = np.random.choice(tags, size=np.random.randint(1, 4), replace=False)
        
        customers.append({
            "name": name,
            "company": company,
            "doc_type": doc_type,
            "price": price,
            "entry_date": entry_date,
            "deadline_date": deadline_date,
            "status": status,
            "work_count": work_count,
            "progress": progress,
            "email": email,
            "phone": phone,
            "tags": customer_tags
        })
    
    # 채팅 메시지 생성 (채팅 인터페이스용)
    chat_messages = {}
    
    # 10명의 고객에 대한 채팅 메시지 생성
    for i in range(10):
        customer_name = customers[i]["name"]
        messages = []
        
        # 각 고객별 3-8개의 메시지 생성
        for j in range(np.random.randint(3, 9)):
            # 메시지 시간 및 날짜
            msg_date = (datetime.now() - timedelta(days=np.random.randint(0, 7))).strftime("%Y-%m-%d")
            msg_time = f"{np.random.randint(9, 19):02d}:{np.random.randint(0, 60):02d}"
            
            # 발신자 (고객 또는 나)
            sender = np.random.choice(["customer", "me"])
            
            # 메시지 내용
            if sender == "customer":
                customer_messages = [
                    f"안녕하세요, {doc_types[i % len(doc_types)]} 교정 부탁드립니다.",
                    "언제쯤 완료될까요?",
                    "수정사항이 있는데 반영 가능할까요?",
                    "감사합니다. 결과물 확인했습니다.",
                    "추가 비용은 얼마인가요?",
                    "오늘 오후까지 가능할까요?",
                    "이 부분은 어떻게 수정하는 게 좋을까요?",
                    "파일 보내드립니다. 확인 부탁드려요."
                ]
                message = np.random.choice(customer_messages)
            else:
                my_messages = [
                    f"{doc_types[i % len(doc_types)]} 교정 요청 확인했습니다.",
                    "내일 오후까지 완료해드리겠습니다.",
                    "수정사항 반영해드렸습니다.",
                    "추가 비용은 5만원입니다.",
                    "오늘 오후 5시까지 완료 예정입니다.",
                    "파일 확인했습니다. 작업 진행하겠습니다.",
                    "수정된 파일 첨부해드립니다.",
                    "문의사항 있으시면 언제든지 말씀해주세요."
                ]
                message = np.random.choice(my_messages)
            
            # 파일 첨부 여부
            has_file = np.random.random() < 0.2  # 20% 확률로 파일 첨부
            msg_data = {
                "sender": sender,
                "message": message,
                "time": msg_time,
                "date": msg_date
            }
            
            if has_file:
                file_types = ["docx", "pdf", "txt", "pptx", "xlsx"]
                file_name = f"문서_{np.random.randint(1000, 9999)}.{np.random.choice(file_types)}"
                msg_data["file"] = file_name
            
            messages.append(msg_data)
        
        chat_messages[customer_name] = messages
    
    # 탭 1: 교정 현황 개요
    with tab1:
        # 주요 지표 행
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">총 교정 문서</div>
                <div class="metric-value">1,890</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">총 발견 오류</div>
                <div class="metric-value">3,373</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">평균 문서당 오류</div>
                <div class="metric-value">1.78</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">평균 교정 시간</div>
                <div class="metric-value">61.7분</div>
            </div>
            """, unsafe_allow_html=True)
        
        # 차트 행 1 - 문서 유형별 교정 현황 및 오류 유형 분포
        st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">문서 유형별 교정 현황</div>', unsafe_allow_html=True)
            
            # 문서 유형별 데이터 생성
            doc_counts = {}
            for doc_type in doc_types:
                doc_counts[doc_type] = np.random.randint(50, 300)
            
            # 데이터프레임 생성
            df_docs = pd.DataFrame({
                "문서 유형": list(doc_counts.keys()),
                "교정 수": list(doc_counts.values())
            })
            
            # 차트 생성
            fig_docs = px.bar(
                df_docs,
                x="문서 유형",
                y="교정 수",
                color="교정 수",
                color_continuous_scale="Blues",
                template="plotly_white"
            )
            
            fig_docs.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                xaxis_title="",
                yaxis_title="교정 수",
                coloraxis_showscale=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Malgun Gothic, Arial", size=12)
            )
            
            st.plotly_chart(fig_docs, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">오류 유형 분포</div>', unsafe_allow_html=True)
            
            # 오류 유형별 데이터 생성
            error_counts = {}
            for error_type in error_types:
                error_counts[error_type] = np.random.randint(100, 600)
            
            # 데이터프레임 생성
            df_errors = pd.DataFrame({
                "오류 유형": list(error_counts.keys()),
                "오류 수": list(error_counts.values())
            })
            
            # 차트 생성 - 파이 차트로 변경하여 겹침 문제 해결
            fig_errors = px.pie(
                df_errors,
                names="오류 유형",
                values="오류 수",
                color_discrete_sequence=px.colors.qualitative.Set3,
                template="plotly_white",
                hole=0.4
            )
            
            fig_errors.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10)
                ),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Malgun Gothic, Arial", size=12)
            )
            
            fig_errors.update_traces(textposition='inside', textinfo='percent')
            
            st.plotly_chart(fig_errors, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # 탭 2: 상세 분석 (태그 기반 필터링으로 수정)
    with tab2:
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        
        # 태그 기반 필터링 UI
        st.markdown('<div class="filter-group">', unsafe_allow_html=True)
        st.markdown('<div class="filter-label">문서 유형</div>', unsafe_allow_html=True)
        
        # 문서 유형 태그 (JavaScript로 선택 상태 토글)
        doc_type_tags_html = ""
        for doc_type in ["전체"] + doc_types:
            selected = " selected" if doc_type == "전체" else ""
            doc_type_tags_html += f'<span class="tag{selected}" onclick="this.classList.toggle(\'selected\')">{doc_type}</span>'
        
        st.markdown(f'<div>{doc_type_tags_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 상태 필터
        st.markdown('<div class="filter-group">', unsafe_allow_html=True)
        st.markdown('<div class="filter-label">상태</div>', unsafe_allow_html=True)
        
        # 상태 태그
        status_tags_html = ""
        for status in ["전체", "완료", "진행중", "대기중"]:
            selected = " selected" if status == "전체" else ""
            status_tags_html += f'<span class="tag{selected}" onclick="this.classList.toggle(\'selected\')">{status}</span>'
        
        st.markdown(f'<div>{status_tags_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 태그 필터
        st.markdown('<div class="filter-group">', unsafe_allow_html=True)
        st.markdown('<div class="filter-label">태그</div>', unsafe_allow_html=True)
        
        # 태그 목록
        tags_html = ""
        for tag in tags:
            tags_html += f'<span class="tag" onclick="this.classList.toggle(\'selected\')">{tag}</span>'
        
        st.markdown(f'<div>{tags_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 날짜 범위 및 필터 버튼
        col1, col2 = st.columns([3, 1])
        
        with col1:
            date_range = st.date_input(
                "기간 선택",
                value=(datetime.now() - timedelta(days=30), datetime.now()),
                max_value=datetime.now()
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            filter_button = st.button("필터 적용", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 필터링된 데이터 생성 (샘플)
        # 실제 구현 시 필터 조건에 따라 데이터를 필터링해야 함
        filtered_data = pd.DataFrame({
            "문서 ID": [f"DOC-{i:04d}" for i in range(1, 101)],  # 100개 데이터 생성
            "문서 유형": np.random.choice(doc_types, 100),
            "제목": [f"샘플 문서 {i}" for i in range(1, 101)],
            "상태": np.random.choice(["완료", "진행중", "대기중"], 100, p=[0.6, 0.3, 0.1]),
            "오류 수": np.random.randint(1, 20, 100),
            "교정 시간(분)": np.random.randint(30, 180, 100),
            "등록일": [(datetime.now() - timedelta(days=np.random.randint(1, 90))).strftime("%Y-%m-%d") for _ in range(100)],
            "태그": [", ".join(np.random.choice(tags, size=np.random.randint(1, 4), replace=False)) for _ in range(100)],
            "담당자": np.random.choice(["김교정", "이수정", "박편집", "최리뷰", "정검토"], 100),
            "고객명": np.random.choice(customer_names, 100),
            "완료일": [(datetime.now() - timedelta(days=np.random.randint(0, 60))).strftime("%Y-%m-%d") for _ in range(100)]
        })
        
        # 필터링된 데이터 차트
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container"><div class="chart-title">문서 유형별 오류 분포</div>', unsafe_allow_html=True)
            
            # 문서 유형별 오류 분포 데이터
            df_type_errors = filtered_data.groupby("문서 유형")["오류 수"].sum().reset_index()
            
            # 문서 유형별 오류 분포 차트
            fig_type_errors = px.bar(
                df_type_errors, 
                x="문서 유형", 
                y="오류 수",
                color="문서 유형",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                template="plotly_white"
            )
            
            fig_type_errors.update_layout(
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                xaxis_title="",
                yaxis_title="오류 수",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Malgun Gothic, Arial", size=12)
            )
            
            st.plotly_chart(fig_type_errors, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="chart-container"><div class="chart-title">태그별 문서 수</div>', unsafe_allow_html=True)
            
            # 태그 데이터 처리 (쉼표로 구분된 태그를 분리하여 카운트)
            all_tags = []
            for tag_list in filtered_data["태그"]:
                all_tags.extend([tag.strip() for tag in tag_list.split(",")])
            
            tag_counts = pd.Series(all_tags).value_counts().reset_index()
            tag_counts.columns = ["태그", "문서 수"]
            
            # 상위 8개 태그만 표시
            tag_counts = tag_counts.head(8)
            
            # 태그별 문서 수 차트
            fig_tag_counts = px.bar(
                tag_counts, 
                x="태그", 
                y="문서 수",
                color="태그",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                template="plotly_white"
            )
            
            fig_tag_counts.update_layout(
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                xaxis_title="",
                yaxis_title="문서 수",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Malgun Gothic, Arial", size=12)
            )
            
            st.plotly_chart(fig_tag_counts, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 필터링된 데이터 테이블
        st.markdown('<div class="chart-container"><div class="chart-title">필터링된 문서 목록</div>', unsafe_allow_html=True)
        st.dataframe(filtered_data, use_container_width=True, height=400)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 데이터 다운로드 버튼
        csv = filtered_data.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="CSV 다운로드",
            data=csv,
            file_name="문서교정_데이터.csv",
            mime="text/csv",
        )
    
    # 탭 3: 교정 히스토리 갤러리 (교정 수 및 태그 필터링 추가)
    with tab3:
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        
        # 갤러리 필터 - 첫 번째 행
        col1, col2 = st.columns(2)
        
        with col1:
            gallery_doc_type = st.selectbox("문서 유형 선택", ["전체"] + doc_types, key="gallery_doc_type")
        
        with col2:
            gallery_sort = st.selectbox("정렬 기준", ["최신순", "오류 많은 순", "교정 시간 긴 순"], key="gallery_sort")
        
        # 갤러리 필터 - 두 번째 행 (교정 수 범위)
        st.markdown('<div class="filter-label">교정 수 범위</div>', unsafe_allow_html=True)
        correction_range = st.slider("", 0, 20, (0, 10), key="correction_range")
        
        # 갤러리 필터 - 세 번째 행 (태그 필터)
        st.markdown('<div class="filter-group">', unsafe_allow_html=True)
        st.markdown('<div class="filter-label">태그 필터</div>', unsafe_allow_html=True)
        
        # 태그 목록
        gallery_tags_html = ""
        for tag in tags:
            gallery_tags_html += f'<span class="tag" onclick="this.classList.toggle(\'selected\')">{tag}</span>'
        
        st.markdown(f'<div>{gallery_tags_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 필터 적용 버튼
        gallery_filter_button = st.button("갤러리 필터 적용", use_container_width=True, key="gallery_filter")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 갤러리 아이템 (샘플)
        st.markdown('<h3 style="font-size: 1.2rem; margin: 20px 0 15px 0;">교정 문서 갤러리</h3>', unsafe_allow_html=True)
        
        # 갤러리 아이템 생성 (샘플 데이터)
        gallery_items = []
        for i in range(24):  # 24개 아이템 생성
            doc_id = f"DOC-{i+1:04d}"
            doc_title = f"샘플 문서 {i+1}"
            doc_type = np.random.choice(doc_types)
            doc_status = np.random.choice(["완료", "진행중", "대기중"])
            doc_errors = np.random.randint(1, 15)
            doc_corrections = np.random.randint(1, 15)
            doc_date = (datetime.now() - timedelta(days=np.random.randint(1, 60))).strftime("%Y-%m-%d")
            doc_tags = np.random.choice(tags, size=np.random.randint(1, 4), replace=False)
            
            gallery_items.append({
                "id": doc_id,
                "title": doc_title,
                "type": doc_type,
                "status": doc_status,
                "errors": doc_errors,
                "corrections": doc_corrections,
                "date": doc_date,
                "tags": doc_tags
            })
        
        # 3x4 그리드로 갤러리 아이템 표시 (한 페이지에 12개)
        for i in range(0, 12, 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(gallery_items):  # 아이템이 있는 경우만 표시
                    with cols[j]:
                        item = gallery_items[i + j]
                        
                        # 상태에 따른 색상 설정
                        status_color = "#4CAF50" if item["status"] == "완료" else "#2196F3" if item["status"] == "진행중" else "#FFC107"
                        
                        # 태그 HTML 생성
                        tags_html = ""
                        for tag in item["tags"]:
                            tags_html += f'<span class="card-tag">{tag}</span>'
                        
                        # 카드 HTML 생성
                        st.markdown(f"""
                        <div class="gallery-card">
                            <div class="card-title">{item["title"]}</div>
                            <div class="card-meta">ID: {item["id"]}</div>
                            <div class="card-meta">유형: {item["type"]}</div>
                            <div class="card-meta">교정 수: {item["corrections"]}회</div>
                            <div class="card-tags">{tags_html}</div>
                            <div class="card-footer">
                                <span class="card-status" style="background-color: {status_color};">{item["status"]}</span>
                                <span style="color: #e53935; font-size: 0.85rem;">오류 {item["errors"]}건</span>
                            </div>
                            <div style="color: #888; font-size: 0.8rem; text-align: right; margin-top: 8px;">{item["date"]}</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # 더 보기 버튼
        st.button("더 보기", use_container_width=True, key="load_more")
    
    # 탭 4: 고객 관리 CRM
    with tab4:
        st.markdown('<h2 style="font-size: 1.3rem; margin-bottom: 15px;">고객 관리 CRM</h2>', unsafe_allow_html=True)
        
        # CRM 필터 컨테이너
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        
        # 필터 행
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            crm_doc_type = st.selectbox("문서 유형", ["전체"] + doc_types, key="crm_doc_type")
        
        with col2:
            crm_status = st.selectbox("고객 상태", ["전체", "신규", "진행중", "완료", "대기중", "휴면"], key="crm_status")
        
        with col3:
            crm_sort = st.selectbox("정렬 기준", ["최신순", "마감일순", "금액 높은순", "작업수 많은순"], key="crm_sort")
        
        with col4:
            crm_search = st.text_input("고객명/회사 검색", placeholder="검색어 입력...")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # CRM 대시보드 지표
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">총 고객 수</div>
                <div class="metric-value">127명</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">이번 달 신규 고객</div>
                <div class="metric-value">23명</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">평균 고객 단가</div>
                <div class="metric-value">32.5만원</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">이번 달 매출</div>
                <div class="metric-value">1,245만원</div>
            </div>
            """, unsafe_allow_html=True)
        
        # 고객 목록 및 상세 정보
        st.markdown('<div class="chart-container" style="margin-top: 20px;">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">고객 목록</div>', unsafe_allow_html=True)
        
        # 고객 목록 표시 (최대 10명)
        for i, customer in enumerate(customers[:10]):
            # 진행률 계산
            progress_style = f'width: {customer["progress"]}%;'
            progress_color = "#4CAF50" if customer["progress"] >= 80 else "#2196F3" if customer["progress"] >= 40 else "#FFC107"
            
            # 마감일까지 남은 일수 계산
            days_left = (customer["deadline_date"] - datetime.now()).days
            deadline_class = "deadline-tag" if days_left <= 7 else ""
            deadline_text = f"마감 {days_left}일 전" if days_left > 0 else "마감일 지남"
            
            # 가격 포맷팅
            price_formatted = f"{customer['price']:,}원"
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
            
            # 태그 HTML 생성
            tags_html = ""
            for tag in customer["tags"]:
                tags_html += f'<span class="card-tag">{tag}</span>'
            
            # 고객 카드 HTML 생성
            st.markdown(f"""
            <div class="crm-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="customer-name">{customer["name"]} <span style="font-size: 0.8rem; color: #64748b;">({customer["company"]})</span></div>
                    <div>
                        <span class="price-tag">{price_formatted}</span>
                        <span class="{deadline_class}" style="margin-left: 8px;">{deadline_text}</span>
                    </div>
                </div>
                <div class="customer-info" style="margin-top: 8px;">
                    <div class="customer-detail">문서 유형: {customer["doc_type"]}</div>
                    <div class="customer-detail">상태: {customer["status"]}</div>
                    <div class="customer-detail">작업 수: {customer["work_count"]}회</div>
                </div>
                <div class="customer-info" style="margin-top: 5px;">
                    <div class="customer-detail">이메일: {customer["email"]}</div>
                    <div class="customer-detail">연락처: {customer["phone"]}</div>
                </div>
                <div class="card-tags" style="margin-top: 8px;">{tags_html}</div>
                <div style="margin-top: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                        <span style="font-size: 0.85rem; color: #64748b;">진행률</span>
                        <span style="font-size: 0.85rem; color: #64748b;">{customer["progress"]}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-value" style="{progress_style} background-color: {progress_color};"></div>
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 12px;">
                    <div style="color: #64748b; font-size: 0.8rem;">등록일: {customer["entry_date"].strftime("%Y-%m-%d")}</div>
                    <div style="color: #64748b; font-size: 0.8rem;">마감일: {customer["deadline_date"].strftime("%Y-%m-%d")}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # 더 보기 버튼
        st.button("더 많은 고객 보기", use_container_width=True, key="load_more_customers")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 고객 분석 차트
        st.markdown('<div class="chart-container" style="margin-top: 20px;">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">고객 분석</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 문서 유형별 고객 수 및 평균 가격
            # 데이터 준비
            doc_type_customer_data = {}
            for doc_type in doc_types:
                doc_customers = [c for c in customers if c["doc_type"] == doc_type]
                if doc_customers:
                    avg_price = sum(c["price"] for c in doc_customers) / len(doc_customers)
                    doc_type_customer_data[doc_type] = {
                        "count": len(doc_customers),
                        "avg_price": avg_price / 10000  # 만원 단위로 변환
                    }
            
            # 데이터프레임 생성
            df_doc_customers = pd.DataFrame([
                {"문서 유형": dt, "고객 수": data["count"], "평균 가격(만원)": data["avg_price"]}
                for dt, data in doc_type_customer_data.items()
            ])
            
            # 차트 생성
            fig_doc_customers = px.bar(
                df_doc_customers,
                x="문서 유형",
                y="고객 수",
                color="평균 가격(만원)",
                color_continuous_scale="Viridis",
                template="plotly_white"
            )
            
            fig_doc_customers.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                xaxis_title="",
                yaxis_title="고객 수",
                coloraxis_colorbar=dict(title="평균 가격(만원)"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Malgun Gothic, Arial", size=12),
                xaxis=dict(tickangle=-45)
            )
            
            st.plotly_chart(fig_doc_customers, use_container_width=True)
            
        with col2:
            # 월별 신규 고객 추이
            # 데이터 준비
            months = [(datetime.now() - timedelta(days=30*i)).strftime("%Y-%m") for i in range(6)]
            months.reverse()  # 오름차순 정렬
            
            monthly_new_customers = []
            for month in months:
                year, month_num = map(int, month.split("-"))
                month_start = datetime(year, month_num, 1)
                if month_num == 12:
                    month_end = datetime(year + 1, 1, 1) - timedelta(days=1)
                else:
                    month_end = datetime(year, month_num + 1, 1) - timedelta(days=1)
                
                # 해당 월에 등록한 고객 수 계산
                count = sum(1 for c in customers if month_start <= c["entry_date"] <= month_end)
                monthly_new_customers.append(count)
            
            # 데이터프레임 생성
            df_monthly_customers = pd.DataFrame({
                "월": months,
                "신규 고객 수": monthly_new_customers
            })
            
            # 차트 생성
            fig_monthly_customers = px.line(
                df_monthly_customers,
                x="월",
                y="신규 고객 수",
                markers=True,
                template="plotly_white"
            )
            
            fig_monthly_customers.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                xaxis_title="",
                yaxis_title="신규 고객 수",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Malgun Gothic, Arial", size=12)
            )
            
            st.plotly_chart(fig_monthly_customers, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 탭 5: 고객 채팅
    with tab5:
        st.markdown('<h2 style="font-size: 1.3rem; margin-bottom: 15px;">고객 채팅</h2>', unsafe_allow_html=True)
        
        # 채팅 인터페이스
        st.markdown("""
        <div class="chat-container">
            <div class="chat-sidebar">
                <div class="chat-search">
                    <input type="text" placeholder="고객 검색...">
                </div>
                <div style="overflow-y: auto; height: calc(100% - 60px);">
        """, unsafe_allow_html=True)
        
        # 채팅 사이드바 고객 목록
        for i, name in enumerate(chat_messages.keys()):
            # 마지막 메시지 가져오기
            last_message = chat_messages[name][-1]["message"]
            if len(last_message) > 30:
                last_message = last_message[:30] + "..."
            
            # 온라인 상태 (랜덤)
            is_online = np.random.random() < 0.3
            status_class = "status-online" if is_online else "status-offline"
            
            # 활성화 상태 (첫 번째 고객 선택)
            active_class = " active" if i == 0 else ""
            
            # 고객 이름의 첫 글자
            initial = name[0]
            
            st.markdown(f"""
            <div class="chat-user{active_class}">
                <div style="display: flex; align-items: center;">
                    <div class="chat-user-avatar">{initial}</div>
                    <div style="flex: 1;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div class="chat-user-name">{name}</div>
                            <div class="chat-user-status {status_class}"></div>
                        </div>
                        <div class="chat-user-preview">{last_message}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
                </div>
            </div>
            <div class="chat-main">
                <div class="chat-header">
                    <div class="chat-avatar">김</div>
                    <div>
                        <div style="font-weight: 500; color: #1e293b;">김지훈</div>
                        <div style="font-size: 0.8rem; color: #64748b;">삼성전자</div>
                    </div>
                </div>
                <div class="chat-messages">
        """, unsafe_allow_html=True)
        
        # 채팅 메시지 표시 (첫 번째 고객의 메시지)
        first_customer = list(chat_messages.keys())[0]
        for message in chat_messages[first_customer]:
            message_class = "sent" if message["sender"] == "me" else "received"
            
            # 파일 첨부 여부 확인
            file_html = ""
            if "file" in message:
                file_name = message["file"]
                file_ext = file_name.split(".")[-1]
                file_html = f"""
                <div class="file-attachment">
                    <div class="file-icon">📎</div>
                    <div class="file-name">{file_name}</div>
                </div>
                """
            
            st.markdown(f"""
            <div class="chat-message {message_class}">
                <div class="chat-bubble">{message["message"]}</div>
                {file_html}
                <div class="chat-time">{message["time"]} | {message["date"]}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # AI 추천 응답
        st.markdown("""
                <div class="chat-suggestions">
                    <div class="chat-suggestion">네, 확인해보겠습니다.</div>
                    <div class="chat-suggestion">수정 완료되었습니다.</div>
                    <div class="chat-suggestion">추가 비용이 발생할 수 있습니다.</div>
                    <div class="chat-suggestion">첨부파일 확인 부탁드립니다.</div>
                    <div class="chat-suggestion">오늘 오후까지 완료 예정입니다.</div>
                </div>
                </div>
                <div class="chat-input">
                    <div class="chat-tools">
                        <div class="chat-tool">📎</div>
                        <div class="chat-tool">📷</div>
                        <div class="chat-tool">📋</div>
                        <div class="chat-tool">🤖</div>
                    </div>
                    <div style="display: flex; gap: 10px;">
                        <input type="text" placeholder="메시지 입력..." style="flex: 1; padding: 10px; border-radius: 20px; border: 1px solid #e2e8f0;">
                        <button style="background-color: #1e3a8a; color: white; border: none; border-radius: 20px; padding: 0 20px; font-weight: 500;">전송</button>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 채팅 기능 설명
        st.markdown("""
        <div style="margin-top: 20px; padding: 15px; background-color: #f8fafc; border-radius: 10px; border: 1px solid #e2e8f0;">
            <h3 style="font-size: 1.1rem; margin-bottom: 10px; color: #1e3a8a;">채팅 기능 안내</h3>
            <ul style="margin-left: 20px; color: #475569;">
                <li>고객과 실시간 채팅으로 소통할 수 있습니다.</li>
                <li>파일 전송 기능을 통해 문서를 주고받을 수 있습니다.</li>
                <li>AI 추천 응답으로 빠르게 메시지를 작성할 수 있습니다.</li>
                <li>자동완성 기능으로 효율적인 채팅이 가능합니다.</li>
                <li>채팅 내역은 자동으로 저장되어 언제든지 확인할 수 있습니다.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # 실제 채팅 입력이 필요한 경우 고유한 key 추가
        chat_message = st.text_input("메시지 입력", key="crm_chat_input")
        if st.button("전송", key="crm_chat_send"):
            # 메시지 처리 로직
            pass

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
    st.title("PDF 챗봇")
    
    # Ollama 서버 상태 확인
    if not check_ollama_server():
        st.error("Ollama 서버가 실행되고 있지 않습니다.")
        st.info("터미널에서 'ollama serve' 명령어를 실행하여 Ollama 서버를 시작하세요.")
        return
    
    # ll

# 메인 함수 - 이전 메뉴 스타일로 복원

# 페이지 설정 - 모바일 호환성 개선
st.set_page_config(
    page_title="AI 문서 도우미",
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
    
    /* 태그 스타일 */
    .tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-right: 6px;
        margin-bottom: 6px;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .tag.selected {
        background-color: #1e3a8a;
        color: white;
    }
    
    .tag:not(.selected) {
        background-color: #e2e8f0;
        color: #475569;
    }
    
    .tag:hover:not(.selected) {
        background-color: #cbd5e1;
    }
    
    /* 필터 그룹 스타일 */
    .filter-group {
        margin-bottom: 15px;
    }
    
    .filter-label {
        font-size: 0.9rem;
        font-weight: 500;
        color: #334155;
        margin-bottom: 8px;
    }
    
    /* 갤러리 카드 스타일 */
    .gallery-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .gallery-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .card-title {
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 8px;
        color: #1e293b;
    }
    
    .card-meta {
        color: #64748b;
        font-size: 0.85rem;
        margin-bottom: 5px;
    }
    
    .card-tags {
        margin-top: 10px;
        margin-bottom: 10px;
    }
    
    .card-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin-right: 4px;
        background-color: #f1f5f9;
        color: #475569;
    }
    
    .card-status {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        color: white;
    }
    
    .card-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 12px;
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

# 문서교정 대시보드 기능
def render_document_correction_dashboard():
    # CSS 스타일 정의
    st.markdown("""
    <style>
        /* 전체 대시보드 스타일 */
        .dashboard-container {
            padding: 10px;
            background-color: #f8fafc;
            border-radius: 10px;
        }
        
        /* 헤더 스타일 */
        .dashboard-header {
            font-size: 1.5rem;
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #e2e8f0;
        }
        
        /* 탭 스타일 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            white-space: pre-wrap;
            border-radius: 4px;
            font-weight: 500;
            background-color: #f1f5f9;
            color: #64748b;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #1e3a8a !important;
            color: white !important;
        }
        
        /* 메트릭 카드 스타일 */
        .metric-container {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            text-align: center;
            height: 100%;
            border: 1px solid #e2e8f0;
        }
        
        .metric-title {
            font-size: 0.9rem;
            color: #64748b;
            margin-bottom: 8px;
        }
        
        .metric-value {
            font-size: 1.8rem;
            font-weight: 600;
            color: #1e293b;
        }
        
        /* 차트 컨테이너 스타일 */
        .chart-container {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border: 1px solid #e2e8f0;
        }
        
        .chart-title {
            font-size: 1rem;
            font-weight: 500;
            color: #1e293b;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 1px solid #e2e8f0;
        }
        
        /* 필터 컨테이너 스타일 */
        .filter-container {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border: 1px solid #e2e8f0;
        }
        
        .filter-label {
            font-size: 0.9rem;
            font-weight: 500;
            color: #1e293b;
            margin-bottom: 8px;
        }
        
        .filter-group {
            margin-bottom: 15px;
        }
        
        /* 태그 스타일 */
        .tag {
            display: inline-block;
            padding: 5px 10px;
            background-color: #f1f5f9;
            color: #64748b;
            border-radius: 20px;
            margin-right: 8px;
            margin-bottom: 8px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s;
            border: 1px solid #e2e8f0;
        }
        
        .tag:hover {
            background-color: #e2e8f0;
        }
        
        .tag.selected {
            background-color: #1e3a8a;
            color: white;
            border-color: #1e3a8a;
        }
        
        /* 데이터 테이블 스타일 */
        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        
        .data-table th {
            background-color: #f1f5f9;
            padding: 10px;
            text-align: left;
            font-weight: 500;
            color: #1e293b;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .data-table td {
            padding: 10px;
            border-bottom: 1px solid #e2e8f0;
            color: #475569;
        }
        
        .data-table tr:hover {
            background-color: #f8fafc;
        }
        
        /* 갤러리 카드 스타일 */
        .gallery-card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border: 1px solid #e2e8f0;
            transition: all 0.2s;
            height: 100%;
        }
        
        .gallery-card:hover {
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        
        .card-title {
            font-size: 1rem;
            font-weight: 500;
            color: #1e293b;
            margin-bottom: 8px;
        }
        
        .card-meta {
            font-size: 0.85rem;
            color: #64748b;
            margin-bottom: 5px;
        }
        
        .card-tags {
            margin-top: 10px;
            margin-bottom: 10px;
        }
        
        .card-tag {
            display: inline-block;
            padding: 3px 8px;
            background-color: #f1f5f9;
            color: #64748b;
            border-radius: 15px;
            margin-right: 5px;
            margin-bottom: 5px;
            font-size: 0.75rem;
        }
        
        .card-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 10px;
        }
        
        .card-status {
            padding: 3px 8px;
            border-radius: 15px;
            font-size: 0.75rem;
            color: white;
        }
        
        /* CRM 스타일 개선 */
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
            color: #1e293b;
            margin-bottom: 8px;
        }
        
        .customer-info {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
        }
        
        .customer-detail {
            font-size: 0.9rem;
            color: #475569;
        }
        
        .price-tag {
            background-color: #ecfdf5;
            color: #047857;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        
        .deadline-tag {
            background-color: #fff1f2;
            color: #be123c;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background-color: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .progress-value {
            height: 100%;
            border-radius: 4px;
        }
        
        /* 채팅 UI 개선 */
        .chat-container {
            display: flex;
            height: 600px;
            background-color: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border: 1px solid #e2e8f0;
            margin-bottom: 20px;
        }
        
        .chat-sidebar {
            width: 280px;
            background-color: #f8fafc;
            border-right: 1px solid #e2e8f0;
            display: flex;
            flex-direction: column;
            height: 100%;
        }
        
        .chat-search {
            padding: 15px;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .chat-search input {
            width: 100%;
            padding: 8px 12px;
            border-radius: 20px;
            border: 1px solid #e2e8f0;
            background-color: white;
            font-size: 0.9rem;
        }
        
        .chat-user {
            padding: 12px 15px;
            cursor: pointer;
            transition: all 0.2s;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .chat-user:hover {
            background-color: #f1f5f9;
        }
        
        .chat-user.active {
            background-color: #e0f2fe;
        }
        
        .chat-user-avatar {
            width: 40px;
            height: 40px;
            background-color: #1e3a8a;
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 500;
            margin-right: 12px;
        }
        
        .chat-user-name {
            font-size: 0.95rem;
            font-weight: 500;
            color: #1e293b;
            margin-bottom: 4px;
        }
        
        .chat-user-preview {
            font-size: 0.8rem;
            color: #64748b;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .chat-user-status {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }
        
        .status-online {
            background-color: #10b981;
        }
        
        .status-offline {
            background-color: #cbd5e1;
        }
        
        .chat-main {
            flex: 1;
            display: flex;
            flex-direction: column;
            height: 100%;
        }
        
        .chat-header {
            padding: 15px;
            border-bottom: 1px solid #e2e8f0;
            display: flex;
            align-items: center;
        }
        
        .chat-avatar {
            width: 40px;
            height: 40px;
            background-color: #1e3a8a;
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 500;
            margin-right: 12px;
        }
        
        .chat-messages {
            flex: 1;
            padding: 15px;
            overflow-y: auto;
            background-color: #f8fafc;
            display: flex;
            flex-direction: column;
        }
        
        .chat-message {
            margin-bottom: 15px;
            max-width: 70%;
            display: flex;
            flex-direction: column;
        }
        
        .chat-message.sent {
            align-self: flex-end;
        }
        
        .chat-message.received {
            align-self: flex-start;
        }
        
        .chat-bubble {
            padding: 12px 15px;
            border-radius: 18px;
            font-size: 0.95rem;
            line-height: 1.4;
            margin-bottom: 4px;
        }
        
        .chat-message.sent .chat-bubble {
            background-color: #1e3a8a;
            color: white;
            border-top-right-radius: 4px;
        }
        
        .chat-message.received .chat-bubble {
            background-color: white;
            color: #1e293b;
            border: 1px solid #e2e8f0;
            border-top-left-radius: 4px;
        }
        
        .chat-time {
            font-size: 0.75rem;
            color: #94a3b8;
            align-self: flex-end;
        }
        
        .file-attachment {
            display: flex;
            align-items: center;
            background-color: white;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 8px 12px;
            margin-bottom: 8px;
        }
        
        .file-icon {
            margin-right: 8px;
            font-size: 1.2rem;
        }
        
        .file-name {
            font-size: 0.85rem;
            color: #1e293b;
        }
        
        .chat-suggestions {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            padding: 10px 15px;
            border-top: 1px solid #e2e8f0;
            background-color: white;
        }
        
        .chat-suggestion {
            background-color: #f1f5f9;
            color: #1e293b;
            padding: 6px 12px;
            border-radius: 16px;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s;
            border: 1px solid #e2e8f0;
        }
        
        .chat-suggestion:hover {
            background-color: #e2e8f0;
        }
        
        .chat-input {
            padding: 15px;
            border-top: 1px solid #e2e8f0;
            background-color: white;
        }
        
        .chat-tools {
            display: flex;
            gap: 15px;
            margin-bottom: 10px;
        }
        
        .chat-tool {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background-color: #f1f5f9;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .chat-tool:hover {
            background-color: #e2e8f0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # 대시보드 헤더
    st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-header">문서 교정 대시보드</div>', unsafe_allow_html=True)
    
    # 탭 생성
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["교정 현황 개요", "상세 분석", "교정 히스토리 갤러리", "고객 관리 CRM", "고객 채팅"])
    
    # 샘플 데이터 생성
    np.random.seed(42)  # 일관된 결과를 위한 시드 설정
    
    # 문서 유형 및 오류 유형 정의
    doc_types = ["이력서", "자기소개서", "논문", "보고서", "제안서", "계약서", "이메일", "블로그 포스트"]
    error_types = ["맞춤법", "문법", "어휘", "문장 구조", "논리적 오류", "일관성", "형식", "참고문헌"]
    
    # 태그 정의
    tags = ["급함", "중요", "VIP고객", "신규", "재의뢰", "할인적용", "영문", "한글", "일문", "중문", "학술", "비즈니스", "기술", "의학", "법률"]
    
    # 고객 이름 생성 (customer_names 정의)
    first_names = ["김", "이", "박", "최", "정", "강", "조", "윤", "장", "임"]
    last_names = ["지훈", "민준", "서연", "지영", "현우", "예은", "도윤", "수빈", "준호", "민지"]
    customer_names = [f"{np.random.choice(first_names)}{np.random.choice(last_names)}" for _ in range(20)]
    
    # 고객 데이터 생성 (CRM용)
    customers = []
    for i in range(50):  # 50명의 고객 생성
        # 이름 생성
        name = np.random.choice(customer_names)
        
        # 회사 생성
        companies = ["삼성전자", "LG전자", "현대자동차", "SK하이닉스", "네이버", "카카오", "쿠팡", "배달의민족", "토스", "당근마켓", "개인"]
        company = np.random.choice(companies)
        
        # 문서 유형
        doc_type = np.random.choice(doc_types)
        
        # 가격 설정 (문서 유형별 다른 범위)
        if doc_type == "이력서":
            price = np.random.randint(50000, 150000)
        elif doc_type == "자기소개서":
            price = np.random.randint(80000, 200000)
        elif doc_type == "논문":
            price = np.random.randint(300000, 800000)
        elif doc_type == "보고서":
            price = np.random.randint(150000, 400000)
        elif doc_type == "제안서":
            price = np.random.randint(200000, 500000)
        elif doc_type == "계약서":
            price = np.random.randint(250000, 600000)
        else:
            price = np.random.randint(50000, 300000)
        
        # 날짜 생성
        entry_date = datetime.now() - timedelta(days=np.random.randint(1, 180))
        deadline_date = entry_date + timedelta(days=np.random.randint(3, 30))
        
        # 상태 설정
        statuses = ["신규", "진행중", "완료", "대기중", "휴면"]
        status = np.random.choice(statuses, p=[0.2, 0.3, 0.3, 0.1, 0.1])
        
        # 작업 수
        work_count = np.random.randint(1, 10)
        
        # 진행률
        progress = np.random.randint(0, 101)
        
        # 이메일 및 연락처
        email = f"{name.lower()}{np.random.randint(100, 999)}@{np.random.choice(['gmail.com', 'naver.com', 'kakao.com', 'daum.net'])}"
        phone = f"010-{np.random.randint(1000, 9999)}-{np.random.randint(1000, 9999)}"
        
        # 태그 설정
        customer_tags = np.random.choice(tags, size=np.random.randint(1, 4), replace=False)
        
        customers.append({
            "name": name,
            "company": company,
            "doc_type": doc_type,
            "price": price,
            "entry_date": entry_date,
            "deadline_date": deadline_date,
            "status": status,
            "work_count": work_count,
            "progress": progress,
            "email": email,
            "phone": phone,
            "tags": customer_tags
        })
    
    # 채팅 메시지 생성 (채팅 인터페이스용)
    chat_messages = {}
    
    # 10명의 고객에 대한 채팅 메시지 생성
    for i in range(10):
        customer_name = customers[i]["name"]
        messages = []
        
        # 각 고객별 3-8개의 메시지 생성
        for j in range(np.random.randint(3, 9)):
            # 메시지 시간 및 날짜
            msg_date = (datetime.now() - timedelta(days=np.random.randint(0, 7))).strftime("%Y-%m-%d")
            msg_time = f"{np.random.randint(9, 19):02d}:{np.random.randint(0, 60):02d}"
            
            # 발신자 (고객 또는 나)
            sender = np.random.choice(["customer", "me"])
            
            # 메시지 내용
            if sender == "customer":
                customer_messages = [
                    f"안녕하세요, {doc_types[i % len(doc_types)]} 교정 부탁드립니다.",
                    "언제쯤 완료될까요?",
                    "수정사항이 있는데 반영 가능할까요?",
                    "감사합니다. 결과물 확인했습니다.",
                    "추가 비용은 얼마인가요?",
                    "오늘 오후까지 가능할까요?",
                    "이 부분은 어떻게 수정하는 게 좋을까요?",
                    "파일 보내드립니다. 확인 부탁드려요."
                ]
                message = np.random.choice(customer_messages)
            else:
                my_messages = [
                    f"{doc_types[i % len(doc_types)]} 교정 요청 확인했습니다.",
                    "내일 오후까지 완료해드리겠습니다.",
                    "수정사항 반영해드렸습니다.",
                    "추가 비용은 5만원입니다.",
                    "오늘 오후 5시까지 완료 예정입니다.",
                    "파일 확인했습니다. 작업 진행하겠습니다.",
                    "수정된 파일 첨부해드립니다.",
                    "문의사항 있으시면 언제든지 말씀해주세요."
                ]
                message = np.random.choice(my_messages)
            
            # 파일 첨부 여부
            has_file = np.random.random() < 0.2  # 20% 확률로 파일 첨부
            msg_data = {
                "sender": sender,
                "message": message,
                "time": msg_time,
                "date": msg_date
            }
            
            if has_file:
                file_types = ["docx", "pdf", "txt", "pptx", "xlsx"]
                file_name = f"문서_{np.random.randint(1000, 9999)}.{np.random.choice(file_types)}"
                msg_data["file"] = file_name
            
            messages.append(msg_data)
        
        chat_messages[customer_name] = messages
    
    # 탭 1: 교정 현황 개요
    with tab1:
        # 주요 지표 행
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">총 교정 문서</div>
                <div class="metric-value">1,890</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">총 발견 오류</div>
                <div class="metric-value">3,373</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">평균 문서당 오류</div>
                <div class="metric-value">1.78</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">평균 교정 시간</div>
                <div class="metric-value">61.7분</div>
            </div>
            """, unsafe_allow_html=True)
        
        # 차트 행 1 - 문서 유형별 교정 현황 및 오류 유형 분포
        st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">문서 유형별 교정 현황</div>', unsafe_allow_html=True)
            
            # 문서 유형별 데이터 생성
            doc_counts = {}
            for doc_type in doc_types:
                doc_counts[doc_type] = np.random.randint(50, 300)
            
            # 데이터프레임 생성
            df_docs = pd.DataFrame({
                "문서 유형": list(doc_counts.keys()),
                "교정 수": list(doc_counts.values())
            })
            
            # 차트 생성
            fig_docs = px.bar(
                df_docs,
                x="문서 유형",
                y="교정 수",
                color="교정 수",
                color_continuous_scale="Blues",
                template="plotly_white"
            )
            
            fig_docs.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                xaxis_title="",
                yaxis_title="교정 수",
                coloraxis_showscale=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Malgun Gothic, Arial", size=12)
            )
            
            st.plotly_chart(fig_docs, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">오류 유형 분포</div>', unsafe_allow_html=True)
            
            # 오류 유형별 데이터 생성
            error_counts = {}
            for error_type in error_types:
                error_counts[error_type] = np.random.randint(100, 600)
            
            # 데이터프레임 생성
            df_errors = pd.DataFrame({
                "오류 유형": list(error_counts.keys()),
                "오류 수": list(error_counts.values())
            })
            
            # 차트 생성 - 파이 차트로 변경하여 겹침 문제 해결
            fig_errors = px.pie(
                df_errors,
                names="오류 유형",
                values="오류 수",
                color_discrete_sequence=px.colors.qualitative.Set3,
                template="plotly_white",
                hole=0.4
            )
            
            fig_errors.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10)
                ),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Malgun Gothic, Arial", size=12)
            )
            
            fig_errors.update_traces(textposition='inside', textinfo='percent')
            
            st.plotly_chart(fig_errors, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # 탭 2: 상세 분석 (태그 기반 필터링으로 수정)
    with tab2:
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        
        # 태그 기반 필터링 UI
        st.markdown('<div class="filter-group">', unsafe_allow_html=True)
        st.markdown('<div class="filter-label">문서 유형</div>', unsafe_allow_html=True)
        
        # 문서 유형 태그 (JavaScript로 선택 상태 토글)
        doc_type_tags_html = ""
        for doc_type in ["전체"] + doc_types:
            selected = " selected" if doc_type == "전체" else ""
            doc_type_tags_html += f'<span class="tag{selected}" onclick="this.classList.toggle(\'selected\')">{doc_type}</span>'
        
        st.markdown(f'<div>{doc_type_tags_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 상태 필터
        st.markdown('<div class="filter-group">', unsafe_allow_html=True)
        st.markdown('<div class="filter-label">상태</div>', unsafe_allow_html=True)
        
        # 상태 태그
        status_tags_html = ""
        for status in ["전체", "완료", "진행중", "대기중"]:
            selected = " selected" if status == "전체" else ""
            status_tags_html += f'<span class="tag{selected}" onclick="this.classList.toggle(\'selected\')">{status}</span>'
        
        st.markdown(f'<div>{status_tags_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 태그 필터
        st.markdown('<div class="filter-group">', unsafe_allow_html=True)
        st.markdown('<div class="filter-label">태그</div>', unsafe_allow_html=True)
        
        # 태그 목록
        tags_html = ""
        for tag in tags:
            tags_html += f'<span class="tag" onclick="this.classList.toggle(\'selected\')">{tag}</span>'
        
        st.markdown(f'<div>{tags_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 날짜 범위 및 필터 버튼
        col1, col2 = st.columns([3, 1])
        
        with col1:
            date_range = st.date_input(
                "기간 선택",
                value=(datetime.now() - timedelta(days=30), datetime.now()),
                max_value=datetime.now()
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            filter_button = st.button("필터 적용", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 필터링된 데이터 생성 (샘플)
        # 실제 구현 시 필터 조건에 따라 데이터를 필터링해야 함
        filtered_data = pd.DataFrame({
            "문서 ID": [f"DOC-{i:04d}" for i in range(1, 101)],  # 100개 데이터 생성
            "문서 유형": np.random.choice(doc_types, 100),
            "제목": [f"샘플 문서 {i}" for i in range(1, 101)],
            "상태": np.random.choice(["완료", "진행중", "대기중"], 100, p=[0.6, 0.3, 0.1]),
            "오류 수": np.random.randint(1, 20, 100),
            "교정 시간(분)": np.random.randint(30, 180, 100),
            "등록일": [(datetime.now() - timedelta(days=np.random.randint(1, 90))).strftime("%Y-%m-%d") for _ in range(100)],
            "태그": [", ".join(np.random.choice(tags, size=np.random.randint(1, 4), replace=False)) for _ in range(100)],
            "담당자": np.random.choice(["김교정", "이수정", "박편집", "최리뷰", "정검토"], 100),
            "고객명": np.random.choice(customer_names, 100),
            "완료일": [(datetime.now() - timedelta(days=np.random.randint(0, 60))).strftime("%Y-%m-%d") for _ in range(100)]
        })
        
        # 필터링된 데이터 차트
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container"><div class="chart-title">문서 유형별 오류 분포</div>', unsafe_allow_html=True)
            
            # 문서 유형별 오류 분포 데이터
            df_type_errors = filtered_data.groupby("문서 유형")["오류 수"].sum().reset_index()
            
            # 문서 유형별 오류 분포 차트
            fig_type_errors = px.bar(
                df_type_errors, 
                x="문서 유형", 
                y="오류 수",
                color="문서 유형",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                template="plotly_white"
            )
            
            fig_type_errors.update_layout(
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                xaxis_title="",
                yaxis_title="오류 수",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Malgun Gothic, Arial", size=12)
            )
            
            st.plotly_chart(fig_type_errors, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="chart-container"><div class="chart-title">태그별 문서 수</div>', unsafe_allow_html=True)
            
            # 태그 데이터 처리 (쉼표로 구분된 태그를 분리하여 카운트)
            all_tags = []
            for tag_list in filtered_data["태그"]:
                all_tags.extend([tag.strip() for tag in tag_list.split(",")])
            
            tag_counts = pd.Series(all_tags).value_counts().reset_index()
            tag_counts.columns = ["태그", "문서 수"]
            
            # 상위 8개 태그만 표시
            tag_counts = tag_counts.head(8)
            
            # 태그별 문서 수 차트
            fig_tag_counts = px.bar(
                tag_counts, 
                x="태그", 
                y="문서 수",
                color="태그",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                template="plotly_white"
            )
            
            fig_tag_counts.update_layout(
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                xaxis_title="",
                yaxis_title="문서 수",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Malgun Gothic, Arial", size=12)
            )
            
            st.plotly_chart(fig_tag_counts, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 필터링된 데이터 테이블
        st.markdown('<div class="chart-container"><div class="chart-title">필터링된 문서 목록</div>', unsafe_allow_html=True)
        st.dataframe(filtered_data, use_container_width=True, height=400)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 데이터 다운로드 버튼
        csv = filtered_data.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="CSV 다운로드",
            data=csv,
            file_name="문서교정_데이터.csv",
            mime="text/csv",
        )
    
    # 탭 3: 교정 히스토리 갤러리 (교정 수 및 태그 필터링 추가)
    with tab3:
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        
        # 갤러리 필터 - 첫 번째 행
        col1, col2 = st.columns(2)
        
        with col1:
            gallery_doc_type = st.selectbox("문서 유형 선택", ["전체"] + doc_types, key="gallery_doc_type")
        
        with col2:
            gallery_sort = st.selectbox("정렬 기준", ["최신순", "오류 많은 순", "교정 시간 긴 순"], key="gallery_sort")
        
        # 갤러리 필터 - 두 번째 행 (교정 수 범위)
        st.markdown('<div class="filter-label">교정 수 범위</div>', unsafe_allow_html=True)
        correction_range = st.slider("", 0, 20, (0, 10), key="correction_range")
        
        # 갤러리 필터 - 세 번째 행 (태그 필터)
        st.markdown('<div class="filter-group">', unsafe_allow_html=True)
        st.markdown('<div class="filter-label">태그 필터</div>', unsafe_allow_html=True)
        
        # 태그 목록
        gallery_tags_html = ""
        for tag in tags:
            gallery_tags_html += f'<span class="tag" onclick="this.classList.toggle(\'selected\')">{tag}</span>'
        
        st.markdown(f'<div>{gallery_tags_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 필터 적용 버튼
        gallery_filter_button = st.button("갤러리 필터 적용", use_container_width=True, key="gallery_filter")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 갤러리 아이템 (샘플)
        st.markdown('<h3 style="font-size: 1.2rem; margin: 20px 0 15px 0;">교정 문서 갤러리</h3>', unsafe_allow_html=True)
        
        # 갤러리 아이템 생성 (샘플 데이터)
        gallery_items = []
        for i in range(24):  # 24개 아이템 생성
            doc_id = f"DOC-{i+1:04d}"
            doc_title = f"샘플 문서 {i+1}"
            doc_type = np.random.choice(doc_types)
            doc_status = np.random.choice(["완료", "진행중", "대기중"])
            doc_errors = np.random.randint(1, 15)
            doc_corrections = np.random.randint(1, 15)
            doc_date = (datetime.now() - timedelta(days=np.random.randint(1, 60))).strftime("%Y-%m-%d")
            doc_tags = np.random.choice(tags, size=np.random.randint(1, 4), replace=False)
            
            gallery_items.append({
                "id": doc_id,
                "title": doc_title,
                "type": doc_type,
                "status": doc_status,
                "errors": doc_errors,
                "corrections": doc_corrections,
                "date": doc_date,
                "tags": doc_tags
            })
        
        # 3x4 그리드로 갤러리 아이템 표시 (한 페이지에 12개)
        for i in range(0, 12, 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(gallery_items):  # 아이템이 있는 경우만 표시
                    with cols[j]:
                        item = gallery_items[i + j]
                        
                        # 상태에 따른 색상 설정
                        status_color = "#4CAF50" if item["status"] == "완료" else "#2196F3" if item["status"] == "진행중" else "#FFC107"
                        
                        # 태그 HTML 생성
                        tags_html = ""
                        for tag in item["tags"]:
                            tags_html += f'<span class="card-tag">{tag}</span>'
                        
                        # 카드 HTML 생성
                        st.markdown(f"""
                        <div class="gallery-card">
                            <div class="card-title">{item["title"]}</div>
                            <div class="card-meta">ID: {item["id"]}</div>
                            <div class="card-meta">유형: {item["type"]}</div>
                            <div class="card-meta">교정 수: {item["corrections"]}회</div>
                            <div class="card-tags">{tags_html}</div>
                            <div class="card-footer">
                                <span class="card-status" style="background-color: {status_color};">{item["status"]}</span>
                                <span style="color: #e53935; font-size: 0.85rem;">오류 {item["errors"]}건</span>
                            </div>
                            <div style="color: #888; font-size: 0.8rem; text-align: right; margin-top: 8px;">{item["date"]}</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # 더 보기 버튼
        st.button("더 보기", use_container_width=True, key="load_more")
    
    # 탭 4: 고객 관리 CRM
    with tab4:
        st.markdown('<h2 style="font-size: 1.3rem; margin-bottom: 15px;">고객 관리 CRM</h2>', unsafe_allow_html=True)
        
        # CRM 필터 컨테이너
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        
        # 필터 행
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            crm_doc_type = st.selectbox("문서 유형", ["전체"] + doc_types, key="crm_doc_type")
        
        with col2:
            crm_status = st.selectbox("고객 상태", ["전체", "신규", "진행중", "완료", "대기중", "휴면"], key="crm_status")
        
        with col3:
            crm_sort = st.selectbox("정렬 기준", ["최신순", "마감일순", "금액 높은순", "작업수 많은순"], key="crm_sort")
        
        with col4:
            crm_search = st.text_input("고객명/회사 검색", placeholder="검색어 입력...")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # CRM 대시보드 지표
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">총 고객 수</div>
                <div class="metric-value">127명</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">이번 달 신규 고객</div>
                <div class="metric-value">23명</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">평균 고객 단가</div>
                <div class="metric-value">32.5만원</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">이번 달 매출</div>
                <div class="metric-value">1,245만원</div>
            </div>
            """, unsafe_allow_html=True)
        
        # 고객 목록 및 상세 정보
        st.markdown('<div class="chart-container" style="margin-top: 20px;">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">고객 목록</div>', unsafe_allow_html=True)
        
        # 고객 목록 표시 (최대 10명)
        for i, customer in enumerate(customers[:10]):
            # 진행률 계산
            progress_style = f'width: {customer["progress"]}%;'
            progress_color = "#4CAF50" if customer["progress"] >= 80 else "#2196F3" if customer["progress"] >= 40 else "#FFC107"
            
            # 마감일까지 남은 일수 계산
            days_left = (customer["deadline_date"] - datetime.now()).days
            deadline_class = "deadline-tag" if days_left <= 7 else ""
            deadline_text = f"마감 {days_left}일 전" if days_left > 0 else "마감일 지남"
            
            # 가격 포맷팅
            price_formatted = f"{customer['price']:,}원"
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
            
            # 태그 HTML 생성
            tags_html = ""
            for tag in customer["tags"]:
                tags_html += f'<span class="card-tag">{tag}</span>'
            
            # 고객 카드 HTML 생성
            st.markdown(f"""
            <div class="crm-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="customer-name">{customer["name"]} <span style="font-size: 0.8rem; color: #64748b;">({customer["company"]})</span></div>
                    <div>
                        <span class="price-tag">{price_formatted}</span>
                        <span class="{deadline_class}" style="margin-left: 8px;">{deadline_text}</span>
                    </div>
                </div>
                <div class="customer-info" style="margin-top: 8px;">
                    <div class="customer-detail">문서 유형: {customer["doc_type"]}</div>
                    <div class="customer-detail">상태: {customer["status"]}</div>
                    <div class="customer-detail">작업 수: {customer["work_count"]}회</div>
                </div>
                <div class="customer-info" style="margin-top: 5px;">
                    <div class="customer-detail">이메일: {customer["email"]}</div>
                    <div class="customer-detail">연락처: {customer["phone"]}</div>
                </div>
                <div class="card-tags" style="margin-top: 8px;">{tags_html}</div>
                <div style="margin-top: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                        <span style="font-size: 0.85rem; color: #64748b;">진행률</span>
                        <span style="font-size: 0.85rem; color: #64748b;">{customer["progress"]}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-value" style="{progress_style} background-color: {progress_color};"></div>
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 12px;">
                    <div style="color: #64748b; font-size: 0.8rem;">등록일: {customer["entry_date"].strftime("%Y-%m-%d")}</div>
                    <div style="color: #64748b; font-size: 0.8rem;">마감일: {customer["deadline_date"].strftime("%Y-%m-%d")}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # 더 보기 버튼
        st.button("더 많은 고객 보기", use_container_width=True, key="load_more_customers")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 고객 분석 차트
        st.markdown('<div class="chart-container" style="margin-top: 20px;">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">고객 분석</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 문서 유형별 고객 수 및 평균 가격
            # 데이터 준비
            doc_type_customer_data = {}
            for doc_type in doc_types:
                doc_customers = [c for c in customers if c["doc_type"] == doc_type]
                if doc_customers:
                    avg_price = sum(c["price"] for c in doc_customers) / len(doc_customers)
                    doc_type_customer_data[doc_type] = {
                        "count": len(doc_customers),
                        "avg_price": avg_price / 10000  # 만원 단위로 변환
                    }
            
            # 데이터프레임 생성
            df_doc_customers = pd.DataFrame([
                {"문서 유형": dt, "고객 수": data["count"], "평균 가격(만원)": data["avg_price"]}
                for dt, data in doc_type_customer_data.items()
            ])
            
            # 차트 생성
            fig_doc_customers = px.bar(
                df_doc_customers,
                x="문서 유형",
                y="고객 수",
                color="평균 가격(만원)",
                color_continuous_scale="Viridis",
                template="plotly_white"
            )
            
            fig_doc_customers.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                xaxis_title="",
                yaxis_title="고객 수",
                coloraxis_colorbar=dict(title="평균 가격(만원)"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Malgun Gothic, Arial", size=12),
                xaxis=dict(tickangle=-45)
            )
            
            st.plotly_chart(fig_doc_customers, use_container_width=True)
            
        with col2:
            # 월별 신규 고객 추이
            # 데이터 준비
            months = [(datetime.now() - timedelta(days=30*i)).strftime("%Y-%m") for i in range(6)]
            months.reverse()  # 오름차순 정렬
            
            monthly_new_customers = []
            for month in months:
                year, month_num = map(int, month.split("-"))
                month_start = datetime(year, month_num, 1)
                if month_num == 12:
                    month_end = datetime(year + 1, 1, 1) - timedelta(days=1)
                else:
                    month_end = datetime(year, month_num + 1, 1) - timedelta(days=1)
                
                # 해당 월에 등록한 고객 수 계산
                count = sum(1 for c in customers if month_start <= c["entry_date"] <= month_end)
                monthly_new_customers.append(count)
            
            # 데이터프레임 생성
            df_monthly_customers = pd.DataFrame({
                "월": months,
                "신규 고객 수": monthly_new_customers
            })
            
            # 차트 생성
            fig_monthly_customers = px.line(
                df_monthly_customers,
                x="월",
                y="신규 고객 수",
                markers=True,
                template="plotly_white"
            )
            
            fig_monthly_customers.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                xaxis_title="",
                yaxis_title="신규 고객 수",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Malgun Gothic, Arial", size=12)
            )
            
            st.plotly_chart(fig_monthly_customers, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 탭 5: 고객 채팅
    with tab5:
        st.markdown('<h2 style="font-size: 1.3rem; margin-bottom: 15px;">고객 채팅</h2>', unsafe_allow_html=True)
        
        # 채팅 인터페이스
        st.markdown("""
        <div class="chat-container">
            <div class="chat-sidebar">
                <div class="chat-search">
                    <input type="text" placeholder="고객 검색...">
                </div>
                <div style="overflow-y: auto; height: calc(100% - 60px);">
        """, unsafe_allow_html=True)
        
        # 채팅 사이드바 고객 목록
        for i, name in enumerate(chat_messages.keys()):
            # 마지막 메시지 가져오기
            last_message = chat_messages[name][-1]["message"]
            if len(last_message) > 30:
                last_message = last_message[:30] + "..."
            
            # 온라인 상태 (랜덤)
            is_online = np.random.random() < 0.3
            status_class = "status-online" if is_online else "status-offline"
            
            # 활성화 상태 (첫 번째 고객 선택)
            active_class = " active" if i == 0 else ""
            
            # 고객 이름의 첫 글자
            initial = name[0]
            
            st.markdown(f"""
            <div class="chat-user{active_class}">
                <div style="display: flex; align-items: center;">
                    <div class="chat-user-avatar">{initial}</div>
                    <div style="flex: 1;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div class="chat-user-name">{name}</div>
                            <div class="chat-user-status {status_class}"></div>
                        </div>
                        <div class="chat-user-preview">{last_message}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
                </div>
            </div>
            <div class="chat-main">
                <div class="chat-header">
                    <div class="chat-avatar">김</div>
                    <div>
                        <div style="font-weight: 500; color: #1e293b;">김지훈</div>
                        <div style="font-size: 0.8rem; color: #64748b;">삼성전자</div>
                    </div>
                </div>
                <div class="chat-messages">
        """, unsafe_allow_html=True)
        
        # 채팅 메시지 표시 (첫 번째 고객의 메시지)
        first_customer = list(chat_messages.keys())[0]
        for message in chat_messages[first_customer]:
            message_class = "sent" if message["sender"] == "me" else "received"
            
            # 파일 첨부 여부 확인
            file_html = ""
            if "file" in message:
                file_name = message["file"]
                file_ext = file_name.split(".")[-1]
                file_html = f"""
                <div class="file-attachment">
                    <div class="file-icon">📎</div>
                    <div class="file-name">{file_name}</div>
                </div>
                """
            
            st.markdown(f"""
            <div class="chat-message {message_class}">
                <div class="chat-bubble">{message["message"]}</div>
                {file_html}
                <div class="chat-time">{message["time"]} | {message["date"]}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # AI 추천 응답
        st.markdown("""
                <div class="chat-suggestions">
                    <div class="chat-suggestion">네, 확인해보겠습니다.</div>
                    <div class="chat-suggestion">수정 완료되었습니다.</div>
                    <div class="chat-suggestion">추가 비용이 발생할 수 있습니다.</div>
                    <div class="chat-suggestion">첨부파일 확인 부탁드립니다.</div>
                    <div class="chat-suggestion">오늘 오후까지 완료 예정입니다.</div>
                </div>
                </div>
                <div class="chat-input">
                    <div class="chat-tools">
                        <div class="chat-tool">📎</div>
                        <div class="chat-tool">📷</div>
                        <div class="chat-tool">📋</div>
                        <div class="chat-tool">🤖</div>
                    </div>
                    <div style="display: flex; gap: 10px;">
                        <input type="text" placeholder="메시지 입력..." style="flex: 1; padding: 10px; border-radius: 20px; border: 1px solid #e2e8f0;">
                        <button style="background-color: #1e3a8a; color: white; border: none; border-radius: 20px; padding: 0 20px; font-weight: 500;">전송</button>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 채팅 기능 설명
        st.markdown("""
        <div style="margin-top: 20px; padding: 15px; background-color: #f8fafc; border-radius: 10px; border: 1px solid #e2e8f0;">
            <h3 style="font-size: 1.1rem; margin-bottom: 10px; color: #1e3a8a;">채팅 기능 안내</h3>
            <ul style="margin-left: 20px; color: #475569;">
                <li>고객과 실시간 채팅으로 소통할 수 있습니다.</li>
                <li>파일 전송 기능을 통해 문서를 주고받을 수 있습니다.</li>
                <li>AI 추천 응답으로 빠르게 메시지를 작성할 수 있습니다.</li>
                <li>자동완성 기능으로 효율적인 채팅이 가능합니다.</li>
                <li>채팅 내역은 자동으로 저장되어 언제든지 확인할 수 있습니다.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # 실제 채팅 입력이 필요한 경우 고유한 key 추가
        chat_message = st.text_input("메시지 입력", key="crm_chat_input")
        if st.button("전송", key="crm_chat_send"):
            # 메시지 처리 로직
            pass

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
    st.title("PDF 챗봇")
    
    # Ollama 서버 상태 확인
    if not check_ollama_server():
        st.error("Ollama 서버가 실행되고 있지 않습니다.")
        st.info("터미널에서 'ollama serve' 명령어를 실행하여 Ollama 서버를 시작하세요.")
        return
    
    # ll

# 메인 함수 - 이전 메뉴 스타일로 복원

# 페이지 설정 - 모바일 호환성 개선
st.set_page_config(
    page_title="AI 문서 도우미",
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
    
    /* 태그 스타일 */
    .tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-right: 6px;
        margin-bottom: 6px;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .tag.selected {
        background-color: #1e3a8a;
        color: white;
    }
    
    .tag:not(.selected) {
        background-color: #e2e8f0;
        color: #475569;
    }
    
    .tag:hover:not(.selected) {
        background-color: #cbd5e1;
    }
    
    /* 필터 그룹 스타일 */
    .filter-group {
        margin-bottom: 15px;
    }
    
    .filter-label {
        font-size: 0.9rem;
        font-weight: 500;
        color: #334155;
        margin-bottom: 8px;
    }
    
    /* 갤러리 카드 스타일 */
    .gallery-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .gallery-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .card-title {
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 8px;
        color: #1e293b;
    }
    
    .card-meta {
        color: #64748b;
        font-size: 0.85rem;
        margin-bottom: 5px;
    }
    
    .card-tags {
        margin-top: 10px;
        margin-bottom: 10px;
    }
    
    .card-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin-right: 4px;
        background-color: #f1f5f9;
        color: #475569;
    }
    
    .card-status {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        color: white;
    }
    
    .card-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 12px;
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

# 문서교정 대시보드 기능
def render_document_correction_dashboard():
    # CSS 스타일 정의
    st.markdown("""
    <style>
        /* 전체 대시보드 스타일 */
        .dashboard-container {
            padding: 10px;
            background-color: #f8fafc;
            border-radius: 10px;
        }
        
        /* 헤더 스타일 */
        .dashboard-header {
            font-size: 1.5rem;
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #e2e8f0;
        }
        
        /* 탭 스타일 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            white-space: pre-wrap;
            border-radius: 4px;
            font-weight: 500;
            background-color: #f1f5f9;
            color: #64748b;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #1e3a8a !important;
            color: white !important;
        }
        
        /* 메트릭 카드 스타일 */
        .metric-container {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            text-align: center;
            height: 100%;
            border: 1px solid #e2e8f0;
        }
        
        .metric-title {
            font-size: 0.9rem;
            color: #64748b;
            margin-bottom: 8px;
        }
        
        .metric-value {
            font-size: 1.8rem;
            font-weight: 600;
            color: #1e293b;
        }
        
        /* 차트 컨테이너 스타일 */
        .chart-container {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border: 1px solid #e2e8f0;
        }
        
        .chart-title {
            font-size: 1rem;
            font-weight: 500;
            color: #1e293b;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 1px solid #e2e8f0;
        }
        
        /* 필터 컨테이너 스타일 */
        .filter-container {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border: 1px solid #e2e8f0;
        }
        
        .filter-label {
            font-size: 0.9rem;
            font-weight: 500;
            color: #1e293b;
            margin-bottom: 8px;
        }
        
        .filter-group {
            margin-bottom: 15px;
        }
        
        /* 태그 스타일 */
        .tag {
            display: inline-block;
            padding: 5px 10px;
            background-color: #f1f5f9;
            color: #64748b;
            border-radius: 20px;
            margin-right: 8px;
            margin-bottom: 8px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s;
            border: 1px solid #e2e8f0;
        }
        
        .tag:hover {
            background-color: #e2e8f0;
        }
        
        .tag.selected {
            background-color: #1e3a8a;
            color: white;
            border-color: #1e3a8a;
        }
        
        /* 데이터 테이블 스타일 */
        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        
        .data-table th {
            background-color: #f1f5f9;
            padding: 10px;
            text-align: left;
            font-weight: 500;
            color: #1e293b;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .data-table td {
            padding: 10px;
            border-bottom: 1px solid #e2e8f0;
            color: #475569;
        }
        
        .data-table tr:hover {
            background-color: #f8fafc;
        }
        
        /* 갤러리 카드 스타일 */
        .gallery-card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border: 1px solid #e2e8f0;
            transition: all 0.2s;
            height: 100%;
        }
        
        .gallery-card:hover {
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        
        .card-title {
            font-size: 1rem;
            font-weight: 500;
            color: #1e293b;
            margin-bottom: 8px;
        }
        
        .card-meta {
            font-size: 0.85rem;
            color: #64748b;
            margin-bottom: 5px;
        }
        
        .card-tags {
            margin-top: 10px;
            margin-bottom: 10px;
        }
        
        .card-tag {
            display: inline-block;
            padding: 3px 8px;
            background-color: #f1f5f9;
            color: #64748b;
            border-radius: 15px;
            margin-right: 5px;
            margin-bottom: 5px;
            font-size: 0.75rem;
        }
        
        .card-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 10px;
        }
        
        .card-status {
            padding: 3px 8px;
            border-radius: 15px;
            font-size: 0.75rem;
            color: white;
        }
        
        /* CRM 스타일 개선 */
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
            color: #1e293b;
            margin-bottom: 8px;
        }
        
        .customer-info {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
        }
        
        .customer-detail {
            font-size: 0.9rem;
            color: #475569;
        }
        
        .price-tag {
            background-color: #ecfdf5;
            color: #047857;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        
        .deadline-tag {
            background-color: #fff1f2;
            color: #be123c;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background-color: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .progress-value {
            height: 100%;
            border-radius: 4px;
        }
        
        /* 채팅 UI 개선 */
        .chat-container {
            display: flex;
            height: 600px;
            background-color: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border: 1px solid #e2e8f0;
            margin-bottom: 20px;
        }
        
        .chat-sidebar {
            width: 280px;
            background-color: #f8fafc;
            border-right: 1px solid #e2e8f0;
            display: flex;
            flex-direction: column;
            height: 100%;
        }
        
        .chat-search {
            padding: 15px;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .chat-search input {
            width: 100%;
            padding: 8px 12px;
            border-radius: 20px;
            border: 1px solid #e2e8f0;
            background-color: white;
            font-size: 0.9rem;
        }
        
        .chat-user {
            padding: 12px 15px;
            cursor: pointer;
            transition: all 0.2s;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .chat-user:hover {
            background-color: #f1f5f9;
        }
        
        .chat-user.active {
            background-color: #e0f2fe;
        }
        
        .chat-user-avatar {
            width: 40px;
            height: 40px;
            background-color: #1e3a8a;
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 500;
            margin-right: 12px;
        }
        
        .chat-user-name {
            font-size: 0.95rem;
            font-weight: 500;
            color: #1e293b;
            margin-bottom: 4px;
        }
        
        .chat-user-preview {
            font-size: 0.8rem;
            color: #64748b;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .chat-user-status {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }
        
        .status-online {
            background-color: #10b981;
        }
        
        .status-offline {
            background-color: #cbd5e1;
        }
        
        .chat-main {
            flex: 1;
            display: flex;
            flex-direction: column;
            height: 100%;
        }
        
        .chat-header {
            padding: 15px;
            border-bottom: 1px solid #e2e8f0;
            display: flex;
            align-items: center;
        }
        
        .chat-avatar {
            width: 40px;
            height: 40px;
            background-color: #1e3a8a;
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 500;
            margin-right: 12px;
        }
        
        .chat-messages {
            flex: 1;
            padding: 15px;
            overflow-y: auto;
            background-color: #f8fafc;
            display: flex;
            flex-direction: column;
        }
        
        .chat-message {
            margin-bottom: 15px;
            max-width: 70%;
            display: flex;
            flex-direction: column;
        }
        
        .chat-message.sent {
            align-self: flex-end;
        }
        
        .chat-message.received {
            align-self: flex-start;
        }
        
        .chat-bubble {
            padding: 12px 15px;
            border-radius: 18px;
            font-size: 0.95rem;
            line-height: 1.4;
            margin-bottom: 4px;
        }
        
        .chat-message.sent .chat-bubble {
            background-color: #1e3a8a;
            color: white;
            border-top-right-radius: 4px;
        }
        
        .chat-message.received .chat-bubble {
            background-color: white;
            color: #1e293b;
            border: 1px solid #e2e8f0;
            border-top-left-radius: 4px;
        }
        
        .chat-time {
            font-size: 0.75rem;
            color: #94a3b8;
            align-self: flex-end;
        }
        
        .file-attachment {
            display: flex;
            align-items: center;
            background-color: white;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 8px 12px;
            margin-bottom: 8px;
        }
        
        .file-icon {
            margin-right: 8px;
            font-size: 1.2rem;
        }
        
        .file-name {
            font-size: 0.85rem;
            color: #1e293b;
        }
        
        .chat-suggestions {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            padding: 10px 15px;
            border-top: 1px solid #e2e8f0;
            background-color: white;
        }
        
        .chat-suggestion {
            background-color: #f1f5f9;
            color: #1e293b;
            padding: 6px 12px;
            border-radius: 16px;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s;
            border: 1px solid #e2e8f0;
        }
        
        .chat-suggestion:hover {
            background-color: #e2e8f0;
        }
        
        .chat-input {
            padding: 15px;
            border-top: 1px solid #e2e8f0;
            background-color: white;
        }
        
        .chat-tools {
            display: flex;
            gap: 15px;
            margin-bottom: 10px;
        }
        
        .chat-tool {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background-color: #f1f5f9;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .chat-tool:hover {
            background-color: #e2e8f0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # 대시보드 헤더
    st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-header">문서 교정 대시보드</div>', unsafe_allow_html=True)
    
    # 탭 생성
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["교정 현황 개요", "상세 분석", "교정 히스토리 갤러리", "고객 관리 CRM", "고객 채팅"])
    
    # 샘플 데이터 생성
    np.random.seed(42)  # 일관된 결과를 위한 시드 설정
    
    # 문서 유형 및 오류 유형 정의
    doc_types = ["이력서", "자기소개서", "논문", "보고서", "제안서", "계약서", "이메일", "블로그 포스트"]
    error_types = ["맞춤법", "문법", "어휘", "문장 구조", "논리적 오류", "일관성", "형식", "참고문헌"]
    
    # 태그 정의
    tags = ["급함", "중요", "VIP고객", "신규", "재의뢰", "할인적용", "영문", "한글", "일문", "중문", "학술", "비즈니스", "기술", "의학", "법률"]
    
    # 고객 이름 생성 (customer_names 정의)
    first_names = ["김", "이", "박", "최", "정", "강", "조", "윤", "장", "임"]
    last_names = ["지훈", "민준", "서연", "지영", "현우", "예은", "도윤", "수빈", "준호", "민지"]
    customer_names = [f"{np.random.choice(first_names)}{np.random.choice(last_names)}" for _ in range(20)]
    
    # 고객 데이터 생성 (CRM용)
    customers = []
    for i in range(50):  # 50명의 고객 생성
        # 이름 생성
        name = np.random.choice(customer_names)
        
        # 회사 생성
        companies = ["삼성전자", "LG전자", "현대자동차", "SK하이닉스", "네이버", "카카오", "쿠팡", "배달의민족", "토스", "당근마켓", "개인"]
        company = np.random.choice(companies)
        
        # 문서 유형
        doc_type = np.random.choice(doc_types)
        
        # 가격 설정 (문서 유형별 다른 범위)
        if doc_type == "이력서":
            price = np.random.randint(50000, 150000)
        elif doc_type == "자기소개서":
            price = np.random.randint(80000, 200000)
        elif doc_type == "논문":
            price = np.random.randint(300000, 800000)
        elif doc_type == "보고서":
            price = np.random.randint(150000, 400000)
        elif doc_type == "제안서":
            price = np.random.randint(200000, 500000)
        elif doc_type == "계약서":
            price = np.random.randint(250000, 600000)
        else:
            price = np.random.randint(50000, 300000)
        
        # 날짜 생성
        entry_date = datetime.now() - timedelta(days=np.random.randint(1, 180))
        deadline_date = entry_date + timedelta(days=np.random.randint(3, 30))
        
        # 상태 설정
        statuses = ["신규", "진행중", "완료", "대기중", "휴면"]
        status = np.random.choice(statuses, p=[0.2, 0.3, 0.3, 0.1, 0.1])
        
        # 작업 수
        work_count = np.random.randint(1, 10)
        
        # 진행률
        progress = np.random.randint(0, 101)
        
        # 이메일 및 연락처
        email = f"{name.lower()}{np.random.randint(100, 999)}@{np.random.choice(['gmail.com', 'naver.com', 'kakao.com', 'daum.net'])}"
        phone = f"010-{np.random.randint(1000, 9999)}-{np.random.randint(1000, 9999)}"
        
        # 태그 설정
        customer_tags = np.random.choice(tags, size=np.random.randint(1, 4), replace=False)
        
        customers.append({
            "name": name,
            "company": company,
            "doc_type": doc_type,
            "price": price,
            "entry_date": entry_date,
            "deadline_date": deadline_date,
            "status": status,
            "work_count": work_count,
            "progress": progress,
            "email": email,
            "phone": phone,
            "tags": customer_tags
        })
    
    # 채팅 메시지 생성 (채팅 인터페이스용)
    chat_messages = {}
    
    # 10명의 고객에 대한 채팅 메시지 생성
    for i in range(10):
        customer_name = customers[i]["name"]
        messages = []
        
        # 각 고객별 3-8개의 메시지 생성
        for j in range(np.random.randint(3, 9)):
            # 메시지 시간 및 날짜
            msg_date = (datetime.now() - timedelta(days=np.random.randint(0, 7))).strftime("%Y-%m-%d")
            msg_time = f"{np.random.randint(9, 19):02d}:{np.random.randint(0, 60):02d}"
            
            # 발신자 (고객 또는 나)
            sender = np.random.choice(["customer", "me"])
            
            # 메시지 내용
            if sender == "customer":
                customer_messages = [
                    f"안녕하세요, {doc_types[i % len(doc_types)]} 교정 부탁드립니다.",
                    "언제쯤 완료될까요?",
                    "수정사항이 있는데 반영 가능할까요?",
                    "감사합니다. 결과물 확인했습니다.",
                    "추가 비용은 얼마인가요?",
                    "오늘 오후까지 가능할까요?",
                    "이 부분은 어떻게 수정하는 게 좋을까요?",
                    "파일 보내드립니다. 확인 부탁드려요."
                ]
                message = np.random.choice(customer_messages)
            else:
                my_messages = [
                    f"{doc_types[i % len(doc_types)]} 교정 요청 확인했습니다.",
                    "내일 오후까지 완료해드리겠습니다.",
                    "수정사항 반영해드렸습니다.",
                    "추가 비용은 5만원입니다.",
                    "오늘 오후 5시까지 완료 예정입니다.",
                    "파일 확인했습니다. 작업 진행하겠습니다.",
                    "수정된 파일 첨부해드립니다.",
                    "문의사항 있으시면 언제든지 말씀해주세요."
                ]
                message = np.random.choice(my_messages)
            
            # 파일 첨부 여부
            has_file = np.random.random() < 0.2  # 20% 확률로 파일 첨부
            msg_data = {
                "sender": sender,
                "message": message,
                "time": msg_time,
                "date": msg_date
            }
            
            if has_file:
                file_types = ["docx", "pdf", "txt", "pptx", "xlsx"]
                file_name = f"문서_{np.random.randint(1000, 9999)}.{np.random.choice(file_types)}"
                msg_data["file"] = file_name
            
            messages.append(msg_data)
        
        chat_messages[customer_name] = messages
    
    # 탭 1: 교정 현황 개요
    with tab1:
        # 주요 지표 행
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">총 교정 문서</div>
                <div class="metric-value">1,890</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">총 발견 오류</div>
                <div class="metric-value">3,373</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">평균 문서당 오류</div>
                <div class="metric-value">1.78</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">평균 교정 시간</div>
                <div class="metric-value">61.7분</div>
            </div>
            """, unsafe_allow_html=True)
        
        # 차트 행 1 - 문서 유형별 교정 현황 및 오류 유형 분포
        st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">문서 유형별 교정 현황</div>', unsafe_allow_html=True)
            
            # 문서 유형별 데이터 생성
            doc_counts = {}
            for doc_type in doc_types:
                doc_counts[doc_type] = np.random.randint(50, 300)
            
            # 데이터프레임 생성
            df_docs = pd.DataFrame({
                "문서 유형": list(doc_counts.keys()),
                "교정 수": list(doc_counts.values())
            })
            
            # 차트 생성
            fig_docs = px.bar(
                df_docs,
                x="문서 유형",
                y="교정 수",
                color="교정 수",
                color_continuous_scale="Blues",
                template="plotly_white"
            )
            
            fig_docs.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                xaxis_title="",
                yaxis_title="교정 수",
                coloraxis_showscale=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Malgun Gothic, Arial", size=12)
            )
            
            st.plotly_chart(fig_docs, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">오류 유형 분포</div>', unsafe_allow_html=True)
            
            # 오류 유형별 데이터 생성
            error_counts = {}
            for error_type in error_types:
                error_counts[error_type] = np.random.randint(100, 600)
            
            # 데이터프레임 생성
            df_errors = pd.DataFrame({
                "오류 유형": list(error_counts.keys()),
                "오류 수": list(error_counts.values())
            })
            
            # 차트 생성 - 파이 차트로 변경하여 겹침 문제 해결
            fig_errors = px.pie(
                df_errors,
                names="오류 유형",
                values="오류 수",
                color_discrete_sequence=px.colors.qualitative.Set3,
                template="plotly_white",
                hole=0.4
            )
            
            fig_errors.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10)
                ),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Malgun Gothic, Arial", size=12)
            )
            
            fig_errors.update_traces(textposition='inside', textinfo='percent')
            
            st.plotly_chart(fig_errors, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # 탭 2: 상세 분석 (태그 기반 필터링으로 수정)
    with tab2:
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        
        # 태그 기반 필터링 UI
        st.markdown('<div class="filter-group">', unsafe_allow_html=True)
        st.markdown('<div class="filter-label">문서 유형</div>', unsafe_allow_html=True)
        
        # 문서 유형 태그 (JavaScript로 선택 상태 토글)
        doc_type_tags_html = ""
        for doc_type in ["전체"] + doc_types:
            selected = " selected" if doc_type == "전체" else ""
            doc_type_tags_html += f'<span class="tag{selected}" onclick="this.classList.toggle(\'selected\')">{doc_type}</span>'
        
        st.markdown(f'<div>{doc_type_tags_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 상태 필터
        st.markdown('<div class="filter-group">', unsafe_allow_html=True)
        st.markdown('<div class="filter-label">상태</div>', unsafe_allow_html=True)
        
        # 상태 태그
        status_tags_html = ""
        for status in ["전체", "완료", "진행중", "대기중"]:
            selected = " selected" if status == "전체" else ""
            status_tags_html += f'<span class="tag{selected}" onclick="this.classList.toggle(\'selected\')">{status}</span>'
        
        st.markdown(f'<div>{status_tags_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 태그 필터
        st.markdown('<div class="filter-group">', unsafe_allow_html=True)
        st.markdown('<div class="filter-label">태그</div>', unsafe_allow_html=True)
        
        # 태그 목록
        tags_html = ""
        for tag in tags:
            tags_html += f'<span class="tag" onclick="this.classList.toggle(\'selected\')">{tag}</span>'
        
        st.markdown(f'<div>{tags_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 날짜 범위 및 필터 버튼
        col1, col2 = st.columns([3, 1])
        
        with col1:
            date_range = st.date_input(
                "기간 선택",
                value=(datetime.now() - timedelta(days=30), datetime.now()),
                max_value=datetime.now()
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            filter_button = st.button("필터 적용", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 필터링된 데이터 생성 (샘플)
        # 실제 구현 시 필터 조건에 따라 데이터를 필터링해야 함
        filtered_data = pd.DataFrame({
            "문서 ID": [f"DOC-{i:04d}" for i in range(1, 101)],  # 100개 데이터 생성
            "문서 유형": np.random.choice(doc_types, 100),
            "제목": [f"샘플 문서 {i}" for i in range(1, 101)],
            "상태": np.random.choice(["완료", "진행중", "대기중"], 100, p=[0.6, 0.3, 0.1]),
            "오류 수": np.random.randint(1, 20, 100),
            "교정 시간(분)": np.random.randint(30, 180, 100),
            "등록일": [(datetime.now() - timedelta(days=np.random.randint(1, 90))).strftime("%Y-%m-%d") for _ in range(100)],
            "태그": [", ".join(np.random.choice(tags, size=np.random.randint(1, 4), replace=False)) for _ in range(100)],
            "담당자": np.random.choice(["김교정", "이수정", "박편집", "최리뷰", "정검토"], 100),
            "고객명": np.random.choice(customer_names, 100),
            "완료일": [(datetime.now() - timedelta(days=np.random.randint(0, 60))).strftime("%Y-%m-%d") for _ in range(100)]
        })
        
        # 필터링된 데이터 차트
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container"><div class="chart-title">문서 유형별 오류 분포</div>', unsafe_allow_html=True)
            
            # 문서 유형별 오류 분포 데이터
            df_type_errors = filtered_data.groupby("문서 유형")["오류 수"].sum().reset_index()
            
            # 문서 유형별 오류 분포 차트
            fig_type_errors = px.bar(
                df_type_errors, 
                x="문서 유형", 
                y="오류 수",
                color="문서 유형",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                template="plotly_white"
            )
            
            fig_type_errors.update_layout(
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                xaxis_title="",
                yaxis_title="오류 수",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Malgun Gothic, Arial", size=12)
            )
            
            st.plotly_chart(fig_type_errors, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="chart-container"><div class="chart-title">태그별 문서 수</div>', unsafe_allow_html=True)
            
            # 태그 데이터 처리 (쉼표로 구분된 태그를 분리하여 카운트)
            all_tags = []
            for tag_list in filtered_data["태그"]:
                all_tags.extend([tag.strip() for tag in tag_list.split(",")])
            
            tag_counts = pd.Series(all_tags).value_counts().reset_index()
            tag_counts.columns = ["태그", "문서 수"]
            
            # 상위 8개 태그만 표시
            tag_counts = tag_counts.head(8)
            
            # 태그별 문서 수 차트
            fig_tag_counts = px.bar(
                tag_counts, 
                x="태그", 
                y="문서 수",
                color="태그",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                template="plotly_white"
            )
            
            fig_tag_counts.update_layout(
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                xaxis_title="",
                yaxis_title="문서 수",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Malgun Gothic, Arial", size=12)
            )
            
            st.plotly_chart(fig_tag_counts, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 필터링된 데이터 테이블
        st.markdown('<div class="chart-container"><div class="chart-title">필터링된 문서 목록</div>', unsafe_allow_html=True)
        st.dataframe(filtered_data, use_container_width=True, height=400)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 데이터 다운로드 버튼
        csv = filtered_data.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="CSV 다운로드",
            data=csv,
            file_name="문서교정_데이터.csv",
            mime="text/csv",
        )
    
    # 탭 3: 교정 히스토리 갤러리 (교정 수 및 태그 필터링 추가)
    with tab3:
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        
        # 갤러리 필터 - 첫 번째 행
        col1, col2 = st.columns(2)
        
        with col1:
            gallery_doc_type = st.selectbox("문서 유형 선택", ["전체"] + doc_types, key="gallery_doc_type")
        
        with col2:
            gallery_sort = st.selectbox("정렬 기준", ["최신순", "오류 많은 순", "교정 시간 긴 순"], key="gallery_sort")
        
        # 갤러리 필터 - 두 번째 행 (교정 수 범위)
        st.markdown('<div class="filter-label">교정 수 범위</div>', unsafe_allow_html=True)
        correction_range = st.slider("", 0, 20, (0, 10), key="correction_range")
        
        # 갤러리 필터 - 세 번째 행 (태그 필터)
        st.markdown('<div class="filter-group">', unsafe_allow_html=True)
        st.markdown('<div class="filter-label">태그 필터</div>', unsafe_allow_html=True)
        
        # 태그 목록
        gallery_tags_html = ""
        for tag in tags:
            gallery_tags_html += f'<span class="tag" onclick="this.classList.toggle(\'selected\')">{tag}</span>'
        
        st.markdown(f'<div>{gallery_tags_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 필터 적용 버튼
        gallery_filter_button = st.button("갤러리 필터 적용", use_container_width=True, key="gallery_filter")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 갤러리 아이템 (샘플)
        st.markdown('<h3 style="font-size: 1.2rem; margin: 20px 0 15px 0;">교정 문서 갤러리</h3>', unsafe_allow_html=True)
        
        # 갤러리 아이템 생성 (샘플 데이터)
        gallery_items = []
        for i in range(24):  # 24개 아이템 생성
            doc_id = f"DOC-{i+1:04d}"
            doc_title = f"샘플 문서 {i+1}"
            doc_type = np.random.choice(doc_types)
            doc_status = np.random.choice(["완료", "진행중", "대기중"])
            doc_errors = np.random.randint(1, 15)
            doc_corrections = np.random.randint(1, 15)
            doc_date = (datetime.now() - timedelta(days=np.random.randint(1, 60))).strftime("%Y-%m-%d")
            doc_tags = np.random.choice(tags, size=np.random.randint(1, 4), replace=False)
            
            gallery_items.append({
                "id": doc_id,
                "title": doc_title,
                "type": doc_type,
                "status": doc_status,
                "errors": doc_errors,
                "corrections": doc_corrections,
                "date": doc_date,
                "tags": doc_tags
            })
        
        # 3x4 그리드로 갤러리 아이템 표시 (한 페이지에 12개)
        for i in range(0, 12, 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(gallery_items):  # 아이템이 있는 경우만 표시
                    with cols[j]:
                        item = gallery_items[i + j]
                        
                        # 상태에 따른 색상 설정
                        status_color = "#4CAF50" if item["status"] == "완료" else "#2196F3" if item["status"] == "진행중" else "#FFC107"
                        
                        # 태그 HTML 생성
                        tags_html = ""
                        for tag in item["tags"]:
                            tags_html += f'<span class="card-tag">{tag}</span>'
                        
                        # 카드 HTML 생성
                        st.markdown(f"""
                        <div class="gallery-card">
                            <div class="card-title">{item["title"]}</div>
                            <div class="card-meta">ID: {item["id"]}</div>
                            <div class="card-meta">유형: {item["type"]}</div>
                            <div class="card-meta">교정 수: {item["corrections"]}회</div>
                            <div class="card-tags">{tags_html}</div>
                            <div class="card-footer">
                                <span class="card-status" style="background-color: {status_color};">{item["status"]}</span>
                                <span style="color: #e53935; font-size: 0.85rem;">오류 {item["errors"]}건</span>
                            </div>
                            <div style="color: #888; font-size: 0.8rem; text-align: right; margin-top: 8px;">{item["date"]}</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # 더 보기 버튼
        st.button("더 보기", use_container_width=True, key="load_more")
    
    # 탭 4: 고객 관리 CRM
    with tab4:
        st.markdown('<h2 style="font-size: 1.3rem; margin-bottom: 15px;">고객 관리 CRM</h2>', unsafe_allow_html=True)
        
        # CRM 필터 컨테이너
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        
        # 필터 행
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            crm_doc_type = st.selectbox("문서 유형", ["전체"] + doc_types, key="crm_doc_type")
        
        with col2:
            crm_status = st.selectbox("고객 상태", ["전체", "신규", "진행중", "완료", "대기중", "휴면"], key="crm_status")
        
        with col3:
            crm_sort = st.selectbox("정렬 기준", ["최신순", "마감일순", "금액 높은순", "작업수 많은순"], key="crm_sort")
        
        with col4:
            crm_search = st.text_input("고객명/회사 검색", placeholder="검색어 입력...")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # CRM 대시보드 지표
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">총 고객 수</div>
                <div class="metric-value">127명</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">이번 달 신규 고객</div>
                <div class="metric-value">23명</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">평균 고객 단가</div>
                <div class="metric-value">32.5만원</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-title">이번 달 매출</div>
                <div class="metric-value">1,245만원</div>
            </div>
            """, unsafe_allow_html=True)
        
        # 고객 목록 및 상세 정보
        st.markdown('<div class="chart-container" style="margin-top: 20px;">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">고객 목록</div>', unsafe_allow_html=True)
        
        # 고객 목록 표시 (최대 10명)
        for i, customer in enumerate(customers[:10]):
            # 진행률 계산
            progress_style = f'width: {customer["progress"]}%;'
            progress_color = "#4CAF50" if customer["progress"] >= 80 else "#2196F3" if customer["progress"] >= 40 else "#FFC107"
            
            # 마감일까지 남은 일수 계산
            days_left = (customer["deadline_date"] - datetime.now()).days
            deadline_class = "deadline-tag" if days_left <= 7 else ""
            deadline_text = f"마감 {days_left}일 전" if days_left > 0 else "마감일 지남"
            
            # 가격 포맷팅
            price_formatted = f"{customer['price']:,}원"
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
            
            # 태그 HTML 생성
            tags_html = ""
            for tag in customer["tags"]:
                tags_html += f'<span class="card-tag">{tag}</span>'
            
            # 고객 카드 HTML 생성
            st.markdown(f"""
            <div class="crm-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="customer-name">{customer["name"]} <span style="font-size: 0.8rem; color: #64748b;">({customer["company"]})</span></div>
                    <div>
                        <span class="price-tag">{price_formatted}</span>
                        <span class="{deadline_class}" style="margin-left: 8px;">{deadline_text}</span>
def main():
    # 페이지 선택
    page = st.sidebar.selectbox("페이지 선택", ["PDF 챗봇", "문서 교정 대시보드"])
    
    if page == "PDF 챗봇":
        render_pdf_chatbot(chat_key="pdf_page_chat")
    elif page == "문서 교정 대시보드":
        render_document_correction_dashboard()

# 앱 실행
if __name__ == "__main__":
    main() 