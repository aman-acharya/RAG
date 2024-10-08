FROM python:3.11.0-slim
WORKDIR /app
COPY . /app
RUN apt update && apt install -y gcc
RUN pip install --no-cache-dir -r requirements.txt
RUN git clone https://github.com/ayushach007/RAG
ENTRYPOINT [ "streamlit", "run" , "app.py" , "--server.port=8080" , "--server.address=0.0.0.0" ]