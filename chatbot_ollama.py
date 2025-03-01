import os
import streamlit as st
import time
import base64
import uuid
import tempfile
import io
import numpy as np
from PIL import Image

# fitz 라이브러리(PyMuPDF) 관련 코드를 제거하고 기본 PDF 표시 방식 사용
try:
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.llms import Ollama
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError as e:
    st.error(f"필요한 라이브러리가 설치되지 않았습니다: {str(e)}")
    st.warning("다음 명령어로 필요한 라이브러리를 설치하세요:")
    st.code("pip install langchain langchain-community faiss-cpu", language="bash")
    st.stop()

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None

# Ollama 서버 URL 설정 (기본값: localhost:11434)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Ollama 모델 설정
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "llama3.2")
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "llama3.2")

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
        pdf_display = f'''
            <iframe src="data:application/pdf;base64,{base64_pdf}" 
                    width="700"
                    height="800"
                    type="application/pdf">
            </iframe>
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

with st.sidebar:
    st.header(f"Add your documents!")
    
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

    if uploaded_file:
        try:
            file_key = f"{session_id}-{uploaded_file.name}"

            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Indexing your document...")

                if file_key not in st.session_state.get('file_cache', {}):
                    if os.path.exists(temp_dir):
                        loader = PyPDFLoader(file_path)
                    else:    
                        st.error('Could not find the file you uploaded, please check again...')
                        st.stop()
                    
                    pages = loader.load_and_split()
                    
                    # Ollama 임베딩 모델 사용
                    embeddings = OllamaEmbeddings(
                        base_url=OLLAMA_BASE_URL,
                        model=OLLAMA_EMBED_MODEL
                    )
                    
                    # FAISS 벡터 저장소 생성 (Chroma 대신 사용)
                    vectorstore = FAISS.from_documents(
                        documents=pages,
                        embedding=embeddings
                    )
                    
                    # 검색기 설정
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
                    
                    # Ollama LLM 설정 (매개변수 추가)
                    llm = Ollama(
                        base_url=OLLAMA_BASE_URL,
                        model=OLLAMA_CHAT_MODEL,
                        temperature=0.7,  # 약간의 창의성 허용
                        num_predict=512,  # 생성할 최대 토큰 수 증가
                        stop=["<|im_end|>"],  # 적절한 중단 토큰 설정
                        repeat_penalty=1.1,  # 반복 방지
                        top_k=40,  # 다양한 단어 선택 허용
                        top_p=0.9  # 다양성 확보
                    )
                    
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

                st.success("Ready to Chat!")
                display_pdf(uploaded_file)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()     

# 웹사이트 제목
st.title("Llama 3.2 LLM Chatbot")

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
        
        if "rag_chain" in st.session_state:
            # RAG 체인 사용
            result = st.session_state.rag_chain.invoke({"input": prompt, "chat_history": st.session_state.messages})

            # 증거자료 보여주기
            with st.expander("Evidence context"):
                st.write(result["context"])

            # 답변 표시 (스트리밍 효과)
            for chunk in result["answer"].split(" "):
                full_response += chunk + " "
                time.sleep(0.05)  # 스트리밍 속도 조정 (Ollama가 더 빠를 수 있음)
                message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
        else:
            # 문서가 로드되지 않은 경우
            full_response = "먼저 PDF 문서를 업로드해주세요."
            message_placeholder.markdown(full_response)
            
    st.session_state.messages.append({"role": "assistant", "content": full_response})