from langchain_community.vectorstores.chroma import Chroma
from langchain_community.llms.ollama import Ollama

# from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.embeddings import FastEmbedEmbeddings

# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata


class ChatPDF(object):
    vector_store = None
    retriever = None
    chain = None

    def __init__(self, model="mistral", embeddings=FastEmbedEmbeddings()):
        self.model = Ollama(model=model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=100
        )
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
            to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
             maximum and keep the answer concise. [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )
        self.embeddings = embeddings
        """ self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/clip-ViT-B-32-multilingual-v1",
            model_kwargs={"device": "cuda"},
        ) """

    def ingest(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path, extract_images=True).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        vector_store = Chroma.from_documents(
            documents=chunks, embedding=self.embeddings
        )
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )
        print(self.retriever)

        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        )

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF document first."

        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
