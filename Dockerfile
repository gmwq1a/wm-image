FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir simple-lama-inpainting fastapi "uvicorn[standard]" python-multipart
RUN python -c "from simple_lama_inpainting import SimpleLama; SimpleLama()"
WORKDIR /srv
COPY app.py /srv/app.py
ENV PORT=10000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
