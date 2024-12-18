# Base image with Python
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Ollama CLI
RUN apt-get update && apt-get install -y wget && \
    wget https://ollama.com/download/latest/linux -O /ollama-cli && \
    chmod +x /ollama-cli && mv /ollama-cli /usr/local/bin/ollama

# Copy the application code
COPY . .

# Expose Gradio default port
EXPOSE 7860

# Run the app when the container starts
CMD ["python", "rag_gui.py"]
