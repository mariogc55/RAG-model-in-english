import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from deep_translator import GoogleTranslator


pdf_path = "libro de automatas.pdf" # Replace the rute to your pdf
model_path = "Lexi-Llama-3-8B-Uncensored_Q4_K_M.gguf" # adjust acording with you model path
persist_directory = "chroma_db"

def crear_base_de_conocimiento():
    # 1. charging documents
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. text fragmentation
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # 3. Embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")

    # 4. Chroma, creation and persist
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    db.persist()
    print("chromadb created with persist in chroma_db")


def consultar_base_de_conocimiento(query):

    # 0. just confirming it is in english(it can be change to any language from here)
    try:
        translated_query = GoogleTranslator(source='en', target='en').translate(query)
        print(f"translated question(english): {translated_query}")
    except Exception as e:
        print(f"Error translating question: {e}")
        translated_query = query # if traduction fails it uses the original version

    # 1. load embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    # 2. initialicing llama
    llm = LlamaCpp(
        model_path = model_path,
        model_kwargs={"n_gpu_layers": 1},
        n_batch=512,
        n_ctx=8192,
        callback_manager=None,
        verbose=False,
    )

    # 3. retrieval chain and consult run with the user question
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
    result_en = qa.run(translated_query)

    # 5. just change it to your lenguage if you need to
    try:
        result_es = GoogleTranslator(source='en', target='en').translate(result_en)
    except Exception as e:
        print(f"Error translating answer: {e}")
        result_es = result_en # if traduction fails it uses the original

    # 6. run consult
    print(f"Original question: {query}")
    print(f"answer in (english): {result_es}")

if __name__ == "__main__":
    if not os.path.exists(persist_directory):
        crear_base_de_conocimiento()
    else:
        print("knowledge base already exist. Omiting it's creation.")

    while True:
        pregunta = input("Insert you question (or 'exit' to end program): ")
        if pregunta.lower() == "exit":
            break
        consultar_base_de_conocimiento(pregunta)