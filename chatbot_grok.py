import os
import streamlit as st
import time
import base64
import uuid
import tempfile
import io
import numpy as np
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
import requests

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

# 필요한 라이브러리 임포트
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.chains.combine_documents import create_stuff_documents_chain
    
    # Grok API 관련 라이브러리
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from typing import Any, Dict, List, Mapping, Optional
    
    # OpenAI 임베딩 모델 (Grok은 임베딩 API가 없어서 OpenAI 사용)
    try:
        from langchain_openai import OpenAIEmbeddings
        has_openai = True
    except ImportError:
        has_openai = False
        st.warning("OpenAI 라이브러리가 설치되지 않았습니다. 임베딩 기능이 제한될 수 있습니다.")
        
except ImportError as e:
    st.error(f"필요한 라이브러리가 설치되지 않았습니다: {str(e)}")
    st.warning("다음 명령어로 필요한 라이브러리를 설치하세요:")
    st.code("pip install langchain langchain-community langchain-openai faiss-cpu streamlit requests", language="bash")
    st.stop()

# Grok API 클래스 정의
class GrokChatModel(BaseChatModel):
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 1024
    
    @property
    def _llm_type(self) -> str:
        return "grok-chat"
    
    def _generate(
        self,
        messages: List[Any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 메시지 형식 변환
        formatted_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                formatted_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                formatted_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                formatted_messages.append({"role": "assistant", "content": message.content})
        
        payload = {
            "messages": formatted_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        if stop:
            payload["stop"] = stop
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            response_data = response.json()
            
            # 응답 처리
            message_content = response_data["choices"][0]["message"]["content"]
            
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content=message_content),
                        generation_info={"finish_reason": response_data["choices"][0].get("finish_reason")}
                    )
                ]
            )
        except Exception as e:
            raise ValueError(f"Grok API 호출 중 오류 발생: {str(e)}")
    
    async def _agenerate(
        self,
        messages: List[Any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # 비동기 구현은 동기 메서드를 호출
        return self._generate(messages, stop, run_manager, **kwargs)

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None

# 환경 설정
GROK_API_KEY = os.environ.get("GROK_API_KEY", "xai-BZDTPNCSBayfE5PKA7dStIN0mK19IJtmpdiCch9aIWQFNZZKj6WZyBzM8ss9OBZBIPBzisnBDT5nQRgK")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

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
                <iframe src="data:application/pdf;base64,{base64_pdf}" 
                        width="100%"
                        height="500px"
                        type="application/pdf">
                    <p>PDF를 표시할 수 없습니다. <a href="data:application/pdf;base64,{base64_pdf}" download="{file.name}">PDF 다운로드</a></p>
                </iframe>
            </div>
        '''
        st.markdown("### PDF 미리보기")
        st.markdown(pdf_display, unsafe_allow_html=True)
        
        # 원본 파일 다운로드 버튼 제공
        file.seek(0)
        st.download_button(
            label="PDF 다운로드",
            data=file_data,
            file_name=file.name,
            mime="application/pdf"
        )
        
    except Exception as e:
        st.error(f"PDF 표시 중 오류가 발생했습니다: {e}")
    
    # 파일 포인터 위치 다시 초기화
    file.seek(0)

# LLM 초기화 함수
def initialize_basic_llm():
    if "basic_llm" not in st.session_state:
        try:
            st.session_state.basic_llm = GrokChatModel(
                api_key=GROK_API_KEY,
                temperature=0.7,
                max_tokens=1024
            )
        except Exception as e:
            st.error(f"Grok API 연결 오류: {str(e)}")
            return None
    
    return st.session_state.basic_llm

# 임베딩 모델 초기화 함수
def get_embeddings_model():
    if has_openai:
        # OpenAI 임베딩 사용 (Grok은 임베딩 API가 없음)
        if not OPENAI_API_KEY:
            st.sidebar.warning("OpenAI API 키가 설정되지 않았습니다. 사이드바에서 API 키를 입력하세요.")
            return None
        return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    else:
        st.error("임베딩 모델을 초기화할 수 없습니다. OpenAI API 키가 필요합니다.")
        return None

# 사이드바 구성 - 모바일 환경 지원 추가
with st.sidebar:
    st.header(f"Chatbot Options")
    
    # OpenAI API 키 입력 (임베딩용)
    openai_api_key = st.text_input("OpenAI API Key (임베딩용)", value=OPENAI_API_KEY, type="password")
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        OPENAI_API_KEY = openai_api_key
    
    # 채팅 초기화 버튼 추가
    if st.button("Reset Chat"):
        reset_chat()
        st.success("Chat history has been reset.")
    
    st.markdown("---")
    
    # PDF 업로드 섹션 (선택 사항)
    st.header("Optional: Add PDF Document")
    st.write("Upload a PDF to enable document-based Q&A")
    
    # 모바일 환경을 위한 설명 추가
    uploaded_file = st.file_uploader("Choose your `.pdf` file (optional)", type="pdf")
    
    if uploaded_file:
        try:
            file_key = f"{session_id}-{uploaded_file.name}"

            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Indexing your document...")
                
                # 인덱싱 과정에 로딩 상태 표시
                with st.spinner("Processing document..."):
                    if file_key not in st.session_state.get('file_cache', {}):
                        if os.path.exists(temp_dir):
                            loader = PyPDFLoader(file_path)
                        else:    
                            st.error('Could not find the file you uploaded, please check again...')
                            st.stop()
                        
                        pages = loader.load_and_split()
                        
                        # 임베딩 모델 가져오기
                        embeddings = get_embeddings_model()
                        if not embeddings:
                            st.error("임베딩 모델을 초기화할 수 없습니다. OpenAI API 키를 입력하세요.")
                            st.stop()
                        
                        # FAISS 벡터 저장소 생성
                        vectorstore = FAISS.from_documents(
                            documents=pages,
                            embedding=embeddings
                        )
                        
                        # 검색기 설정
                        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
                        
                        # LLM 초기화
                        llm = initialize_basic_llm()
                        if not llm:
                            st.error("LLM을 초기화할 수 없습니다.")
                            st.stop()
                        
                        # 컨텍스트화 프롬프트 설정
                        contextualize_q_system_prompt = """이전 대화 내용과 최신 사용자 질문이 있을 때, 이 질문이 이전 대화 내용과 관련이 있을 수 있습니다. 
                        이런 경우, 대화 내용을 알 필요 없이 독립적으로 이해할 수 있는 질문으로 바꾸세요. 
                        질문에 답할 필요는 없고, 필요하다면 그저 다시 구성하거나 그대로 두세요."""

                        contextualize_q_prompt = ChatPromptTemplate.from_messages(
                            [
                                ("system", contextualize_q_system_prompt),
                                MessagesPlaceholder("chat_history"),
                                ("human", "{input}"),
                            ]
                        )

                        # 대화 기록을 인식하는 검색기 생성
                        history_aware_retriever = create_history_aware_retriever(
                            llm, retriever, contextualize_q_prompt
                        )

                        # 질문-답변 프롬프트 설정
                        qa_system_prompt = """당신은 유용하고 상세한 답변을 제공하는 지식이 풍부한 AI 어시스턴트입니다.
                        사용자 질문에 답변할 때 다음 지침을 따르세요:
                        
                        1. 제공된 문서 내용을 기반으로 상세하고 명확한 답변을 제공하세요.
                        2. 답변은 최소 3-5문장으로 구성하며, 필요한 경우 더 자세한 설명을 제공하세요.
                        3. 문서에서 답변을 찾을 수 없는 경우, 정직하게 모른다고 말하세요.
                        4. 답변 시 핵심 개념을 먼저 간략히 설명한 후, 세부 내용을 제공하는 구조로 작성하세요.
                        5. 가능한 경우 예시나 유사 사례를 포함하여 답변을 강화하세요.
                        
                        ## 답변 형식
                        📍 답변 내용: (상세한 답변을 여기에 작성)
                        
                        📍 참고 자료: (사용한 문서의 관련 부분)
                        
                        {context}"""
                        
                        qa_prompt = ChatPromptTemplate.from_messages(
                            [
                                ("system", qa_system_prompt),
                                MessagesPlaceholder("chat_history"),
                                ("human", "{input}"),
                            ]
                        )

                        # 문서 체인 생성
                        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

                        # 최종 RAG 체인 생성
                        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
                        
                        # 세션 상태에 체인 저장
                        st.session_state.rag_chain = rag_chain

                st.success("PDF loaded successfully! You can now ask questions about the document.")
                display_pdf(uploaded_file)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()     

# 기본 LLM 초기화 (파일 업로드 없이도 사용 가능)
llm = initialize_basic_llm()
if not llm:
    st.warning("Grok API 연결에 문제가 있습니다. API 키를 확인하세요.")

# 웹사이트 제목
st.title("Grok AI Chatbot")

# 모드 표시
if "rag_chain" in st.session_state:
    st.info("📄 Document Q&A mode: Ask questions about the uploaded PDF")
else:
    st.info("💬 General chat mode: Ask me anything!")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 내용을 기록하기 위해 셋업
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
# 프롬프트 비용이 너무 많이 소요되는 것을 방지하기 위해
MAX_MESSAGES_BEFORE_DELETION = 4

# 웹사이트에서 유저의 인풋을 받고 위에서 만든 AI 에이전트 실행시켜서 답변 받기
if prompt := st.chat_input("Ask a question!"):
    # LLM이 초기화되지 않은 경우 처리
    if not llm:
        st.error("Grok API 연결에 문제가 있습니다. API 키를 확인하세요.")
        st.stop()
    
    # 유저가 보낸 질문이면 유저 아이콘과 질문 보여주기
    # 만약 현재 저장된 대화 내용 기록이 4개보다 많으면 자르기
    if len(st.session_state.messages) >= MAX_MESSAGES_BEFORE_DELETION:
        # Remove the first two messages
        del st.session_state.messages[0]
        del st.session_state.messages[0]  
   
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI가 보낸 답변이면 AI 아이콘이랑 LLM 실행시켜서 답변 받고 스트리밍해서 보여주기
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            if "rag_chain" in st.session_state:
                # RAG 체인 사용 (PDF 업로드된 경우)
                result = st.session_state.rag_chain.invoke({"input": prompt, "chat_history": st.session_state.messages})

                # 증거자료 보여주기
                with st.expander("Evidence context"):
                    st.write(result["context"])

                # 답변 표시 (스트리밍 효과)
                for chunk in result["answer"].split(" "):
                    full_response += chunk + " "
                    time.sleep(0.01)  # 스트리밍 속도 조정
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            else:
                # 기본 LLM 사용 (PDF 업로드 없는 경우)
                basic_llm = st.session_state.basic_llm
                
                # 기본 프롬프트 설정
                basic_prompt = ChatPromptTemplate.from_messages([
                    ("system", "당신은 유용하고 상세한 답변을 제공하는 지식이 풍부한 AI 어시스턴트입니다. 사용자의 질문에 친절하고 정확하게 답변해 주세요."),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}")
                ])
                
                # 채팅 체인 생성
                chat_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                
                # Grok API 호출
                messages = [
                    SystemMessage(content="당신은 유용하고 상세한 답변을 제공하는 지식이 풍부한 AI 어시스턴트입니다. 사용자의 질문에 친절하고 정확하게 답변해 주세요.")
                ]
                
                # 이전 대화 내용 추가
                for m in st.session_state.messages[:-1]:  # 마지막 메시지(현재 질문)는 제외
                    if m["role"] == "user":
                        messages.append(HumanMessage(content=m["content"]))
                    elif m["role"] == "assistant":
                        messages.append(AIMessage(content=m["content"]))
                
                # 현재 질문 추가
                messages.append(HumanMessage(content=prompt))
                
                # API 호출
                response = basic_llm.invoke(messages)
                full_response = response.content
                
                # 스트리밍 효과 시뮬레이션
                for i in range(0, len(full_response), 10):
                    chunk = full_response[i:i+10]
                    displayed_text = full_response[:i+10]
                    time.sleep(0.01)
                    message_placeholder.markdown(displayed_text + "▌")
                message_placeholder.markdown(full_response)
                
        except Exception as e:
            error_msg = f"오류가 발생했습니다: {str(e)}"
            st.error(error_msg)
            full_response = error_msg
            
    st.session_state.messages.append({"role": "assistant", "content": full_response})