# Proyecto RAG: Recuperación de Información con LLM y MongoDB Atlas

Este proyecto construye diferentes sistemas RAG (Retrieval-Augmented Generation) utilizando componentes como bases de datos vectoriales, modelos de lenguaje y MongoDB Atlas. Se divide en varios apartados para facilitar su implementación y despliegue.

---

## Requisitos previos

Antes de comenzar, asegúrate de tener instalado lo siguiente:

1. **Python**: Versión >= 3.9
2. **Bibliotecas necesarias**: Se encuentran especificadas en un archivo `requirements.txt`.
3. **Cuenta en MongoDB Atlas**: Para almacenar embeddings y realizar consultas.
4. **Docker**: Para el apartado 4 (Dockerización).
5. **Modelo Ollama (Opcional)**: Para usar LLM localmente (si eliges esta opción).

Para instalar las bibliotecas necesarias:
```bash
pip install -r requirements.txt
```

Asegúrate también de configurar tu cuenta de MongoDB Atlas y obtener tu cadena de conexión.

---

## Apartados del Proyecto

### 1. RAG en inglés que crea vector store a partir de datos dunha páxina web

**Archivos relacionados:**
- `rag_web_en.ipynb`
- `gui_rag.ipynb`

#### Descripción
Este apartado se centra en extraer texto de una URL en inglés, generar embeddings y almacenarlos en una base de datos vectorial. Además, guarda estos embeddings en una colección de MongoDB Atlas.

#### Funcionamiento
1. Ejecuta el archivo `rag_web_en.ipynb`.
2. Ingresa la URL desde la cual deseas extraer texto.
3. El script dividirá el texto, generará embeddings utilizando el modelo `sentence-transformers/all-MiniLM-L6-v2` y creará un vector store con Chroma.
4. Los embeddings se guardarán también en MongoDB Atlas.

#### Implementación
- Configura tu cadena de conexión a MongoDB Atlas en el archivo.
- Asegúrate de tener la biblioteca `BeautifulSoup` instalada para extraer texto de la web.

Comando para ejecutar:
```bash
jupyter notebook rag_web_en.ipynb
```

---

### 2. RAG en castellano que crea vector store a partir de un ou varios ficheiros PDF

**Archivos relacionados:**
- `rag_pdf_es.ipynb`
- `gui_rag.ipynb`

#### Descripción
Este apartado procesa uno o varios archivos PDF en castellano, genera embeddings y los guarda en una base de datos vectorial, además de almacenarlos en MongoDB Atlas.

#### Funcionamiento
1. Ejecuta el archivo `rag_pdf_es.ipynb`.
2. Sube uno o varios archivos PDF al script.
3. El script extraerá texto del PDF, lo dividirá en fragmentos y generará embeddings con el mismo modelo que en el apartado anterior.
4. Los embeddings se guardarán en Chroma y en MongoDB Atlas.

#### Implementación
- Configura tu cadena de conexión a MongoDB Atlas en el archivo.
- Asegúrate de tener la biblioteca `PyPDF2` instalada para procesar PDFs.

Comando para ejecutar:
```bash
jupyter notebook rag_pdf_es.ipynb
```

---

### 3. Crear una GUI para uno de los RAGs anteriores

**Archivos relacionados:**
- `gui_rag.ipynb`

#### Descripción
Este apartado implementa una interfaz gráfica (GUI) utilizando Gradio para interactuar con los sistemas RAG desarrollados anteriormente. Permite procesar URLs o PDFs y realizar consultas al sistema.

#### Funcionamiento
1. Ejecuta el archivo `gui_rag.ipynb`.
2. La interfaz permite:
   - Ingresar una URL o cargar un archivo PDF.
   - Procesar el texto y generar una base de datos vectorial.
   - Realizar preguntas al sistema utilizando el modelo Ollama o cualquier otro configurado.
3. Los embeddings generados se almacenan en MongoDB Atlas.

#### Implementación
- Configura tu cadena de conexión a MongoDB Atlas dentro del archivo `gui_rag.py`.
- Asegúrate de que el servidor Ollama (si lo usas) esté en ejecución y accesible en `http://localhost:11434`.

Comando para ejecutar:
```bash
python gui_rag.ipynb
```

---

### 4. RAG dockerizado contra Mongo Atlas

**Archivos relacionados:**
- `Dockerfile`
- `docker-compose.yml`
- `gui_rag_mongo.ipynb`
- Scripts de los apartados anteriores

#### Descripción
Este apartado dockeriza todo el sistema, incluyendo la conexión a MongoDB Atlas, para facilitar su despliegue.

#### Funcionamiento
1. Construye la imagen Docker utilizando el archivo `Dockerfile`.
2. Ejecuta el contenedor y accede al sistema a través del puerto especificado.
3. La configuración del contenedor incluye todas las dependencias necesarias para los scripts y la conexión a MongoDB Atlas.

#### Implementación
- Modifica las variables necesarias (como credenciales de MongoDB Atlas) en el archivo `docker-compose.yml`.
- Construye y ejecuta la imagen Docker:

Comandos:
```bash
docker-compose build
docker-compose up
```

---


