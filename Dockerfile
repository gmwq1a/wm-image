FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir iopaint fastapi "uvicorn[standard]" python-multipart
RUN python - <<'PY'
from iopaint.helper import download_model
download_model("https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt")
PY
WORKDIR /srv
COPY app.py /srv/app.py
ENV PORT=10000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
