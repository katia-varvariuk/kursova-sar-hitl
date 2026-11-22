# Оркестратор (CPU-версія torch)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
COPY docker/requirements-orchestrator.txt /tmp/req.txt
RUN pip install --no-cache-dir -r /tmp/req.txt

ENV PYTHONUNBUFFERED=1 MLFLOW_DISABLE_AGENT_HINT=1
EXPOSE 8000
CMD ["uvicorn", "orchestrator:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
