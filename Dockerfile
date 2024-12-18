# Base image with Python
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose Gradio default port
EXPOSE 7860

# Run the app when the container starts
CMD ["python", "rag_gui_mongo.ipynb"]
