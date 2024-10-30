from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.globals import set_debug
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import RetrievalQA


import os
from dotenv import load_dotenv

load_dotenv()
set_debug(True)

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=os.getenv("OPENAI_API_KEY"))

carregador = TextLoader("GTB_gold_Nov23.txt", encoding="utf-8")
documentos = carregador.load()

quebrador = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
textos = quebrador.split_documents(documentos)

embeddings = OpenAiEmbeddings()
db = FAISS.from_documents(textos, embeddings)

qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

question = "Como devo proceder caso tenha um item comprado roubado?"
result = qa_chain.invoke({"query": question})
print(result)