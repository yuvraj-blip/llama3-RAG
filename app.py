from flask import Flask, request
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate



folder_path =  r"db"
embedding = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1024, chunk_overlap = 80, length_function = len, is_separator_regex= False
)


raw_prompt = PromptTemplate.from_template(""" 
    <s>[INST] You are a assistant who specialises in searching documents for answers. If you donot find the answer in document say so. [/INST] </s>
    [INST] {input}
            Context:{context}
            Answer:
    [/INST]
""")

app = Flask(__name__)

cached_llm = Ollama(model = "llama3")


@app.route("/ai", methods=["POST"])
def aiPost():
    print("POST /ai called ") 
    json_content = request.json
    query = json_content.get("query")
    print(f"query: {query}")
    response = cached_llm.invoke(query)
    print(response)
    
    response_answer = {"answer": response}
    return response_answer

@app.route("/ask_pdf", methods=["POST"])
def askpdfpost():
    print("POST /askpdf called ") 
    json_content = request.json
    query = json_content.get("query")
    
    print("Loading vector store")
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

    retriever = vector_store.as_retriever(
        search_type =  "similarity_score_threshold",
        search_kwargs = {
            "k":20,
            "score_threshold": 0.1
        },
    )

    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt )

    chain = create_retrieval_chain(retriever, document_chain)
    result = chain.invoke({"input":query} )
    print(result)

    sources = []   
    for source in result["context"]:
        sources.append(
            {"source":source.metadata["source"], "page_content" : source.page_content}
            )



    response_answer = {"answer": result["answer"], "source":sources}
    return response_answer

@app.route("/pdf", methods=["POST"])
def pdfPost():
    file = request.files["file"]
    file_name = file.filename
    save_file = r"pdf/" + file_name
    file.save(save_file)
    print(f"filename: {file_name}")

    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    print(f"Number of documents: {len(docs)}")

    chunks = text_splitter.split_documents(docs)
    print(f"Number of chunks: {len(chunks)}")

    vector_store = Chroma.from_documents(documents = chunks, embedding = embedding, persist_directory = folder_path)

    vector_store.persist()
    response = {"Statue":"Uploaded", "Filename":file_name , "documents": len(docs), "chunks": len(chunks)}
    return response 




def start_app():
    
    app.run(host = "0.0.0.0", port = 8080, debug = True)
if __name__ == "__main__":
    start_app();

