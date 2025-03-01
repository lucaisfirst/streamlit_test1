import os
import streamlit as st
import time
import base64
import uuid
import tempfile
import io
import numpy as np
from PIL import Image

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
    /* 모바일 환경에서의 스타일 조정 */
    @media (max-width: 768px) {
        .main .block-container {
            padding-top: 1rem;
            padding-left: 0.5rem;
            padding-right: 0.5rem;
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