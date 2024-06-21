import os, shutil
import streamlit as st
from function_utils import PdfQA
from constants import *
from tempfile import TemporaryDirectory

# ------------------ Set Streamlit app code ------------------ #
st.set_page_config(
    page_title="Document QnA",
    page_icon="🔖",
    layout="centered",
    initial_sidebar_state="auto"
)

if "pdf_qa_model" not in st.session_state:
    st.session_state["pdf_qa_model"] = PdfQA()


@st.cache_resource
def load_llm(llm, load_in_8bit):
    try:
        if llm == LLM_OPENAI_GPT35:
            pass
        elif llm == LLM_FALCON_SMALL:
            return PdfQA.create_falcon_small(load_in_8bit)
        elif llm == LLM_FASTCHAT_T5_XL:
            return PdfQA.create_fastchat_t5_xl(load_in_8bit)
        elif llm == LLM_FLAN_T5_BASE:
            return PdfQA.create_flan_t5_base(load_in_8bit)
        elif llm == LLM_FLAN_T5_LARGE:
            return PdfQA.create_flan_t5_large(load_in_8bit)
        elif llm == LLM_FLAN_T5_SMALL:
            return PdfQA.create_flan_t5_small(load_in_8bit)
        else:
            raise ValueError("Invalid LLM setting")
    except Exception as e:
        print(f"Error loading LLM: {e}")
        raise
    

@st.cache_resource
def load_embedding(emb):
    try:
        if emb == EMB_OPENAI_ADA:
            print("A")
            return PdfQA.create_openai_emb()
        elif emb == EMB_INSTRUCTOR_XL:
            print("B")
            return PdfQA.create_instructor_xl()
        elif emb == EMB_SBERT_MINILM:
            print("C")
            return PdfQA.create_sbert_mlm()
        elif emb == EMB_SBERT_MPNET_BASE:
            print("D")
            embedding = PdfQA.create_mpnet_base()
            print("Embedding created successfully")
            return embedding
        else:
            raise ValueError('Invalid Embedding setting')
    except Exception as e:
        print(f"Error loading embedding: {e}")
        raise


st.title("PDF Q&A (Self hosted LLMs)")  # PAGE TITLE

with st.sidebar:
    emb = st.radio("**EMBEDDING MODEL**", [EMB_OPENAI_ADA, EMB_SBERT_MINILM, EMB_SBERT_MPNET_BASE, EMB_INSTRUCTOR_XL],index=None)
    llm = st.radio("**LLM**", [LLM_OPENAI_GPT35, LLM_FLAN_T5_SMALL, LLM_FLAN_T5_BASE, LLM_FLAN_T5_LARGE, LLM_FASTCHAT_T5_XL, LLM_FALCON_SMALL],index=None)
    load_in_8bit = st.radio("**LOAD 8 BIT**", [True, False], index=1)
    current_dirtory = os.listdir(".")
    directories = [item for item in current_dirtory if os.path.isdir(os.path.join(".", item))]
    pdf_files_folder_path = st.selectbox("**FOLDER**", directories)
    print(pdf_files_folder_path)
    print("*")

    if st.button("Submit") and pdf_files_folder_path is not None:
        with st.spinner(text="Generating Embeddings.."):
            with TemporaryDirectory() as temp_dir_path:
                if (llm == LLM_OPENAI_GPT35 and OPENAI_API_KEY is None) or (emb == EMB_OPENAI_ADA and OPENAI_API_KEY is None):
                    st.sidebar.success("Update the OpenAI Key and Restart.", icon="❗")
                else:
                    shutil.rmtree(temp_dir_path, ignore_errors=True)
                    shutil.copytree(pdf_files_folder_path, temp_dir_path)
                    st.session_state['pdf_qa_model'].config = {
                        "dir_path": str(temp_dir_path),
                        "embedding": emb,
                        "llm": llm,
                        "load_in_8bit": load_in_8bit
                    }
                    try:
                        print("*"*2, emb)
                        st.session_state['pdf_qa_model'].embedding = load_embedding(emb)
                        print(st.session_state['pdf_qa_model'].embedding)
                        print("*"*3)
                        st.session_state['pdf_qa_model'].llm = load_llm(llm, load_in_8bit)
                        print("*"*4)
                        st.session_state['pdf_qa_model'].init_embeddings()
                        print("*"*5)
                        st.session_state['pdf_qa_model'].init_llm()
                        print("*"*6)
                        st.session_state['pdf_qa_model'].create_vectordb()
                        print("*"*7)
                        st.sidebar.success("Embeddings successfully Created and Store in VectorDB (Chroma)", icon="✔️")
                    except Exception as e:
                        st.sidebar.success(f"Error: {e}")

question = st.text_input('Ask a question')

if st.button("Answer"):
    try:
        st.session_state['pdf_qa_model'].retreival_qa_chain()
        answer = st.session_state['pdf_qa_model'].answer_query(question)
        st.write(f"{answer}")
    except Exception as e:
        st.error(f"Error while answering the question: {str(e)}")