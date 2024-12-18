import gradio as gr
import PyPDF2
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama.llms import OllamaLLM
from pymongo import MongoClient
import numpy as np

# --- Funciones auxiliares ---
def extract_text_from_pdf(pdf_file):
    """Extrae texto de un archivo PDF cargado por el usuario."""
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()


def extract_text_from_url(url):
    """Extrae texto desde una URL usando BeautifulSoup."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        text = "\n".join([para.get_text(strip=True) for para in paragraphs if para.get_text(strip=True)])
        return text if text else "No se encontró texto significativo en la página."
    except Exception as e:
        return f"Error al acceder a la página: {e}"


def process_text_and_create_db(text, collection_name):
    """Divide el texto en fragmentos y crea la base de datos vectorial."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)

    # Crear embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Crear base de datos vectorial con Chroma
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory="chroma_db",
    )
    vectorstore.persist()

    # Guardar embeddings en MongoDB
    save_embeddings_to_mongo(zip(embeddings.embed_documents(chunks), chunks), collection_name)

    return vectorstore, chunks


def query_model(query, vectorstore):
    """Realiza una consulta al modelo RAG."""
    llm = OllamaLLM(model="llama3.2", format="json", server_url="http://localhost:11434")
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    response = qa_chain.invoke({"query": query})
    return response


def save_embeddings_to_mongo(embeddings_with_metadata, collection_name):
    """Guarda los embeddings en MongoDB."""
    db = client["rag_db"]
    collection = db[collection_name]
    data = [{"embedding": emb.tolist(), "metadata": meta} for emb, meta in embeddings_with_metadata]
    collection.insert_many(data)
    print("Embeddings guardados en MongoDB.")


def load_embeddings_from_mongo(collection_name):
    """Carga los embeddings desde MongoDB."""
    db = client["rag_db"]
    collection = db[collection_name]
    cursor = collection.find()
    embeddings = [(np.array(doc["embedding"]), doc["metadata"]) for doc in cursor]
    return embeddings

# --- Funciones principales para Gradio ---
def handle_url_input(url):
    """Procesa la URL ingresada, crea la base de datos y permite consultas."""
    text = extract_text_from_url(url)
    if "Error" in text:
        return text, None, None

    vectorstore, chunks = process_text_and_create_db(text, collection_name="web_facts")
    return (
        f"Texto extraído de la URL:\n\n{text[:500]}...\n\n(Base de datos creada con {len(chunks)} fragmentos)",
        vectorstore,
        "Base de datos lista para consultas. Escribe tu pregunta abajo.",
    )


def handle_pdf_input(pdf_file):
    """Procesa el PDF cargado, crea la base de datos y permite consultas."""
    text = extract_text_from_pdf(pdf_file)
    if not text:
        return "Error: No se pudo extraer texto del archivo PDF.", None, None

    vectorstore, chunks = process_text_and_create_db(text, collection_name="pdf_facts")
    return (
        f"Texto extraído del PDF:\n\n{text[:500]}...\n\n(Base de datos creada con {len(chunks)} fragmentos)",
        vectorstore,
        "Base de datos lista para consultas. Escribe tu pregunta abajo.",
    )


def handle_query(query, vectorstore):
    """Realiza la consulta al modelo usando la base de datos."""
    if vectorstore is None:
        return "Primero debes procesar una URL o un archivo PDF."
    return query_model(query, vectorstore)

# --- Interfaz Gradio ---
with gr.Blocks() as app:
    gr.Markdown("# 📚 RAG GUI: Consulta datos desde Web o PDFs con modelos LLM")
    gr.Markdown("**Selecciona una fuente de datos, procesa el texto y realiza preguntas al sistema RAG.**")

    with gr.Tabs():
        # --- Tab para URL ---
        with gr.Tab("URL (Web)"):
            url_input = gr.Textbox(label="Ingresa una URL")
            url_output_text = gr.Textbox(label="Texto extraído y detalles de la base de datos", interactive=False)
            url_output_status = gr.Textbox(label="Estado de la base de datos", interactive=False)
            url_vectorstore = gr.State(None)  # Almacena el vectorstore para consultas
            url_query_input = gr.Textbox(label="Haz una pregunta", placeholder="Escribe tu consulta aquí...")
            url_query_output = gr.Textbox(label="Respuesta del modelo", interactive=False)

            # Botones
            process_url_button = gr.Button("Procesar URL")
            process_url_button.click(
                handle_url_input, inputs=[url_input], outputs=[url_output_text, url_vectorstore, url_output_status]
            )

            query_url_button = gr.Button("Consultar")
            query_url_button.click(
                handle_query, inputs=[url_query_input, url_vectorstore], outputs=[url_query_output]
            )

        # --- Tab para PDF ---
        with gr.Tab("PDF"):
            pdf_input = gr.File(label="Carga un archivo PDF", type="filepath")
            pdf_output_text = gr.Textbox(label="Texto extraído y detalles de la base de datos", interactive=False)
            pdf_output_status = gr.Textbox(label="Estado de la base de datos", interactive=False)
            pdf_vectorstore = gr.State(None)  # Almacena el vectorstore para consultas
            pdf_query_input = gr.Textbox(label="Haz una pregunta", placeholder="Escribe tu consulta aquí...")
            pdf_query_output = gr.Textbox(label="Respuesta del modelo", interactive=False)

            # Botones
            process_pdf_button = gr.Button("Procesar PDF")
            process_pdf_button.click(
                handle_pdf_input, inputs=[pdf_input], outputs=[pdf_output_text, pdf_vectorstore, pdf_output_status]
            )

            query_pdf_button = gr.Button("Consultar")
            query_pdf_button.click(
                handle_query, inputs=[pdf_query_input, pdf_vectorstore], outputs=[pdf_query_output]
            )

# Mongodb
mongo_uri = "mongodb+srv://<raguser>:<changeme>@cluster0.xcjpc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Conexión al cliente
client = MongoClient(mongo_uri)

# Acceso a la base de datos y colecciones
db = client["rag_db"]
web_collection = db["web_embeddings"]
pdf_collection = db["pdf_embeddings"]

print("Conexión exitosa a MongoDB Atlas")

# --- Ejecutar la aplicación ---
app.launch()
