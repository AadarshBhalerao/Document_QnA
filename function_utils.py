from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.prompts import PromptTemplate
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
import torch, os, re
from tqdm import tqdm
from constants import *

class PdfQA:
    def __init__(self, config: dict={}) -> None:
        self.config = config
        self.llm = None
        self.embedding = None
        self.qa = None
        self.retriver = None
        self.vectordb = None

    @classmethod
    def create_openai_emb(cls):
        return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    @classmethod
    def create_instructor_xl(cls):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(device)
        return HuggingFaceInstructEmbeddings(model_name=EMB_INSTRUCTOR_XL, model_kwargs={'device': device})
    
    @classmethod
    def create_mpnet_base(cls):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HuggingFaceInstructEmbeddings(model_name=EMB_SBERT_MPNET_BASE, model_kwargs={'device': device})
    
    @classmethod
    def create_sbert_mlm(cls):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HuggingFaceInstructEmbeddings(model_name=EMB_SBERT_MINILM, model_kwargs={'device': device})
    
    # ------------------ Create LLM Pipelines ------------------ #

    @classmethod
    def create_fastchat_t5_xl(cls, load_in_8bit=False):
        model = LLM_FASTCHAT_T5_XL
        tokenizer = AutoTokenizer.from_pretrained(model)
        return pipeline(
            model=model,
            tokenizer=tokenizer,
            task='text2text-generation',
            max_new_tokens=100,
            model_kwargs={"device_map":"auto", "load_in_8bit": load_in_8bit, "max_length": 512, "temperature": 0.}
        )
    
    @classmethod
    def create_flan_t5_small(cls, load_in_8bit=False):
        model = LLM_FALCON_SMALL
        tokenizer = AutoTokenizer.from_pretrained(model)
        return pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text2text-generation",
            max_new_tokens=100,
            model_kwargs={"device_map":"auto", "load_in_8bit": load_in_8bit, "max_length": 512, "temperature": 0.}
        )
    
    @classmethod
    def create_flan_t5_base(cls, load_in_8bit=False):
        model = LLM_FLAN_T5_BASE
        tokenizer = AutoTokenizer.from_pretrained(model)
        return pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text2text-generation",
            max_new_tokens=100,
            model_kwargs={"device_map":"auto", "load_in_8bit": load_in_8bit, "max_length": 512, "temperature": 0.}
        )
    
    @classmethod
    def create_flan_t5_large(cls, load_in_8bit=False):
        model = LLM_FLAN_T5_LARGE
        tokenizer = AutoTokenizer.from_pretrained(model)
        return pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text2text-generation",
            max_new_tokens=100,
            model_kwargs={"device_map":"auto", "load_in_8bit": load_in_8bit, "max_length": 512, "temperature": 0.}
        )

    @classmethod
    def create_falcon_small(cls, load_in_8bit=False):
        model = LLM_FALCON_SMALL
        tokenizer = AutoTokenizer.from_pretrained(model)
        return pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text2text-generation",
            max_new_tokens=100,
            model_kwargs={"device_map":"auto", "load_in_8bit": load_in_8bit, "max_length": 512, "temperature": 0.}
        )
    
    def init_embeddings(self) -> None:
        try:
            if self.config['embedding'] == EMB_OPENAI_ADA:
                self.embedding = PdfQA.create_openai_emb()
            elif self.config['embedding'] == EMB_INSTRUCTOR_XL:
                self.embedding = PdfQA.create_instructor_xl()
            elif self.config['embedding'] == EMB_SBERT_MINILM:
                self.embedding = PdfQA.create_sbert_mlm()
            elif self.config['embedding'] == EMB_SBERT_MPNET_BASE:
                self.embedding = PdfQA.create_mpnet_base()
            else:
                raise ValueError("Invalid Embedding Value.")
        except Exception as e:
            print(f"Error {e}")


    def init_llm(self) -> None:

        load_in_8bit = self.config.get('load_in_8bit', False)
        if self.config['llm'] == LLM_OPENAI_GPT35:
            self.llm = ChatOpenAI(model=LLM_OPENAI_GPT35, api_key=OPENAI_API_KEY)
        elif self.config['llm'] == LLM_FLAN_T5_SMALL:
            self.llm = PdfQA.create_flan_t5_small(load_in_8bit=load_in_8bit)
        elif self.config['llm'] == LLM_FLAN_T5_BASE:
            self.llm = PdfQA.create_flan_t5_base(load_in_8bit=load_in_8bit)
        elif self.config['llm'] == LLM_FASTCHAT_T5_XL:
            self.llm = PdfQA.create_fastchat_t5_xl(load_in_8bit=load_in_8bit)
        elif self.config['llm'] == LLM_FLAN_T5_LARGE:
            self.llm = PdfQA.create_flan_t5_large(load_in_8bit=load_in_8bit)
        elif self.config['llm'] == LLM_FALCON_SMALL:
            self.llm = PdfQA.create_falcon_small(load_in_8bit=load_in_8bit)
        else:
            raise ValueError("Invalid LLM Value.")
    

    def create_vectordb(self) -> None:
        print("Vector DB Creation Started")
        dir_path = self.config.get('dir_path')
        persist_directory = self.config.get("persist_directory",None)
        if persist_directory and os.path.exists(persist_directory):
            # LOAD FROM VECTOR DB
            self.vectordb = Chroma(persist_directory=persist_directory,
                                   embedding_function=self.embedding)
        elif dir_path and os.path.exists(dir_path):
            # CREATE VECTOR DB
            documents = []

            for pdf_name in tqdm(os.listdir(dir_path)):
                try:
                    # READ THE DOCUMENT
                    print(pdf_name)
                    loader = PDFPlumberLoader(os.path.join(dir_path, pdf_name))
                    doc_loader = loader.load()
                    # SPLIT THE TEXT
                    text_splitter = CharacterTextSplitter(_chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                    documents.extend(text_splitter)
                except Exception as e:
                    print(f"Failed to load file {e}")
            
            if self.embedding is not None:
                self.vectordb = Chroma.from_documents(documents=documents, embedding=self.embedding, persist_directory=persist_directory)

        else:
            print("PDF not found.")


    def retreival_qa_chain(self):
        self.retriver = self.vectordb.as_retriever(search_kwargs={"k":3})

        if self.config['llm'] == LLM_OPENAI_GPT35:
            self.qa = RetrievalQA.from_chain_type(llm=self.llm,
                                                  chain_type="map_rerank",
                                                  retriver=self.retriver)
        else:
            hf_llm = HuggingFacePipeline(pipeline=self.llm, model_id=self.config['llm'])

            self.qa = RetrievalQA.from_chain_type(llm=hf_llm,
                                                  chain_type="stuff",
                                                  retriver=self.retriver)
            
            prompt = """You are an smart QnA bot and you will be provided some context on investment, along with\
            the context you will also be provided a question. You must understand the question and answer\
            using the context only. Remember, you must not provide any additional data.
                
            context: {context}
            question: {question}
            answer:
            """

            PROMPT = PromptTemplate(template=prompt, input_variables=['context', 'question'])
            self.qa.combine_documents_chain.llm_chain.prompt = PROMPT
            self.qa.combine_documents_chain.verbose = True
            self.qa.return_source_documents = True
    
    def answer_query(self, question: str) -> str:
        answer_dict = self.qa({"query": question})
        print(answer_dict)
        answer = answer_dict['result']
        if self.config['llm'] == LLM_FASTCHAT_T5_XL:
            answer = self._clean_fastchat_t5_output(answer)

        return answer
    
    def _clean_fastchat_t5_output(self, answer: str) -> str:
        answer = re.sub(r"<pad>\s+", "", answer)
        answer = re.sub(r"  ", " ", answer)
        answer = re.sub(r"\n$", "", answer)
        return answer
