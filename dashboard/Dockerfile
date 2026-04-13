FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir \
    streamlit==1.35.0 \
    plotly==5.22.0 \
    pandas==2.2.2 \
    sqlalchemy==2.0.30 \
    httpx==0.27.0

COPY dashboard/ ./dashboard/
COPY app/ ./app/
COPY data/ ./data/

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", \
    "--server.port=8501", "--server.address=0.0.0.0", \
    "--server.headless=true"]
