FROM python:3.10

# Install Git and Git LFS
RUN apt-get update && apt-get install -y git git-lfs && git lfs install

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else
COPY . .

# IMPORTANT: Force pull the real LFS files into the container
RUN git lfs pull

EXPOSE 7860

CMD ["python", "server.py"]
