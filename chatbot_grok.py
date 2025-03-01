import os
import streamlit as st
import time
import base64
import uuid
import tempfile
import io
import numpy as np
from PIL import Image
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain

# 페이지 설정 - 모바일 호환성 개선
st.set_page_config(
    page_title="PDF Chatbot",
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
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.llms import OpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError as e:
    st.error(f"필요한 라이브러리가 설치되지 않았습니다: {str(e)}")
    st.warning("다음 명령어로 필요한 라이브러리를 설치하세요:")
    st.code("pip install langchain langchain-community openai pypdf streamlit", language="bash")
    st.stop()

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None

# OpenAI API 키 설정
openai_api_key = os.environ.get("OPENAI_API_KEY", "")
if not openai_api_key:
    openai_api_key = st.sidebar.text_input("OpenAI API 키를 입력하세요", type="password")
    if not openai_api_key:
        st.warning("OpenAI API 키가 필요합니다.")
        st.stop()
os.environ["OPENAI_API_KEY"] = openai_api_key

# 모델 설정
EMBED_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-3.5-turbo"

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None

def display_pdf(file):
    """PDF 파일을 base64로 인코딩하여 iframe으로 표시합니다."""
    try:
        # 파일 포인터 위치 초기화
        file.seek(0)
        
        # 파일 데이터를 읽어옴
        file_data = file.read()
        
        # base64로 인코딩하여 iframe으로 표시
        base64_pdf = base64.b64encode(file_data).decode('utf-8')
        
        # 반응형 디자인 유지하면서 기존 스타일로 복원
        pdf_display = f'''
            <div class="pdf-container">
                <iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" type="application/pdf"></iframe>
            </div>
        '''
        st.markdown(pdf_display, unsafe_allow_html=True)
        
        # 다운로드 버튼 추가
        st.download_button(
            label="PDF 다운로드",
            data=file,
            file_name="document.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"PDF 표시 중 오류가 발생했습니다: {str(e)}")

def initialize_basic_llm():
    if "basic_llm" not in st.session_state:
        st.session_state.basic_llm = OpenAI(
            model=CHAT_MODEL,
            temperature=0.7,
            max_tokens=512,
            stop=["<|im_end|>"],
            repeat_penalty=1.1,
        )
    return st.session_state.basic_llm

def main():
    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "context" not in st.session_state:
        st.session_state.context = None
    
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    
    if "file_processed" not in st.session_state:
        st.session_state.file_processed = False
    
    # 사이드바 설정
    with st.sidebar:
        st.title("Chatbot 옵션")
        
        # 채팅 초기화 버튼
        if st.button("채팅 초기화"):
            reset_chat()
        
        st.header("PDF 업로드 (선택사항)")
        uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")
        
        # 파일이 업로드되었고 이전에 처리된 파일과 다른 경우
        if uploaded_file is not None and (st.session_state.uploaded_file is None or 
                                         uploaded_file.name != st.session_state.uploaded_file.name):
            st.session_state.uploaded_file = uploaded_file
            st.session_state.file_processed = False
            reset_chat()  # 새 파일이 업로드되면 채팅 초기화
            
            with st.spinner("PDF 파일을 처리 중입니다..."):
                try:
                    # 임시 파일로 저장
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # PDF 로더 사용
                    loader = PyPDFLoader(tmp_path)
                    documents = loader.load()
                    
                    # 텍스트 분할
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    splits = text_splitter.split_documents(documents)
                    
                    # 임베딩 및 벡터 저장소 생성
                    if len(splits) > 0:
                        try:
                            # OpenAI 임베딩 모델 사용
                            embeddings = OpenAIEmbeddings(
                                model=EMBED_MODEL
                            )
                            
                            # FAISS 벡터 저장소 생성
                            vectorstore = FAISS.from_documents(
                                documents=splits,
                                embedding=embeddings
                            )
                            
                            # 검색기 생성
                            retriever = vectorstore.as_retriever(
                                search_type="similarity",
                                search_kwargs={"k": 4}
                            )
                            
                            # OpenAI LLM 설정 (매개변수 추가)
                            llm = OpenAI(
                                model=CHAT_MODEL,
                                temperature=0.7,  # 약간의 창의성 허용
                                max_tokens=512,  # 생성할 최대 토큰 수 증가
                                stop=["<|im_end|>"],  # 적절한 중단 토큰 설정
                                repeat_penalty=1.1,  # 반복 방지
                            )
                            
                            # 프롬프트 템플릿 설정
                            prompt = ChatPromptTemplate.from_template("""
                            <context>
                            {context}
                            </context>
                            
                            질문에 대해 위의 컨텍스트 정보를 사용하여 답변해주세요. 
                            컨텍스트에 관련 정보가 없는 경우, "제공된 문서에서 이 질문에 대한 답변을 찾을 수 없습니다."라고 말하세요.
                            답변은 한국어로 제공하고, 가능한 한 자세하게 설명해주세요.
                            
                            질문: {question}
                            """)
                            
                            # 문서 체인 생성
                            document_chain = create_stuff_documents_chain(llm, prompt)
                            
                            # 검색 체인 생성
                            retrieval_chain = create_retrieval_chain(retriever, document_chain)
                            
                            # 컨텍스트 저장
                            st.session_state.context = retrieval_chain
                            st.session_state.file_processed = True
                            
                            # 임시 파일 삭제
                            os.unlink(tmp_path)
                            
                            st.success("PDF가 성공적으로 처리되었습니다!")
                        except Exception as e:
                            st.error(f"벡터 저장소 생성 중 오류: {str(e)}")
                    else:
                        st.error("PDF에서 텍스트를 추출할 수 없습니다.")
                except Exception as e:
                    st.error(f"PDF 처리 중 오류가 발생했습니다: {str(e)}")
    
    # 메인 영역
    st.title("PDF Chatbot")
    
    # 기본 LLM 초기화 (파일 없이도 작동)
    initialize_basic_llm()
    
    # 업로드된 파일이 있으면 표시
    if st.session_state.uploaded_file is not None:
        st.subheader("업로드된 PDF")
        display_pdf(st.session_state.uploaded_file)
    else:
        st.info("PDF 파일을 업로드하면 문서 기반 질의응답이 가능합니다. 파일 업로드는 선택사항입니다.")
    
    # 채팅 인터페이스
    st.subheader("채팅")
    
    # 이전 메시지 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 사용자 입력
    if prompt := st.chat_input("질문을 입력하세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 응답 생성
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # 파일이 처리되었으면 RAG 사용, 아니면 기본 LLM 사용
                if st.session_state.context and st.session_state.file_processed:
                    # RAG를 사용한 응답 생성
                    with st.spinner("답변 생성 중..."):
                        response = st.session_state.context.invoke({
                            "question": prompt
                        })
                        full_response = response["answer"]
                else:
                    # 기본 LLM을 사용한 응답 생성
                    with st.spinner("답변 생성 중..."):
                        system_prompt = """당신은 친절하고 도움이 되는 AI 어시스턴트입니다. 
                        질문에 대해 명확하고 유용한 답변을 제공하세요. 
                        답변은 한국어로 제공하고, 가능한 한 자세하게 설명해주세요."""
                        
                        prompt_template = ChatPromptTemplate.from_messages([
                            ("system", system_prompt),
                            ("human", "{question}")
                        ])
                        
                        chain = prompt_template | st.session_state.basic_llm | StrOutputParser()
                        full_response = chain.invoke({"question": prompt})
                
                # 타이핑 효과
                for chunk in full_response.split():
                    full_response_so_far = full_response[:full_response.find(chunk) + len(chunk)]
                    message_placeholder.markdown(full_response_so_far + "▌")
                    time.sleep(0.01)
                
                message_placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"오류가 발생했습니다: {str(e)}"
                message_placeholder.markdown(full_response)
            
            # 어시스턴트 메시지 추가
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main() 