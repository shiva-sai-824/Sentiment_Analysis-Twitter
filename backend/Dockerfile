# Use Python 3.10
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend files
COPY . .

# Hugging Face Spaces use port 7860
EXPOSE 7860

# Start the server
CMD ["python", "server.py"]
