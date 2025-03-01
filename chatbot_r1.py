import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import time
import base64
import uuid
import tempfile
import requests
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, SystemMessage

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None

# Ollama 서버 URL 설정 (기본값: localhost:11434)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Ollama 모델 설정
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "deepseek-r1")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "deepseek-r1")

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None

def display_pdf(file):
    # Opening file from file path
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

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
                    
                    # Ollama LLM 설정
                    llm = Ollama(
                        base_url=OLLAMA_BASE_URL,
                        model=OLLAMA_CHAT_MODEL
                    )
                    
                    from langchain.chains import create_history_aware_retriever
                    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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

                    from langchain.chains import create_retrieval_chain
                    from langchain.chains.combine_documents import create_stuff_documents_chain

                    # 질문-답변 프롬프트 설정
                    qa_system_prompt = """질문-답변 업무를 돕는 보조원입니다. 
                    질문에 답하기 위해 검색된 내용을 사용하세요. 
                    답을 모르면 모른다고 말하세요. 
                    답변은 세 문장 이내로 간결하게 유지하세요.

                    ## 답변 예시
                    📍답변 내용: 
                    📍증거: 

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
st.title("Deepseek-R1 LLM Chatbot")

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