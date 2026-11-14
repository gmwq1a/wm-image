FROM python:3.11-slim
RUN pip install --no-cache-dir opencv-python-headless numpy fastapi "uvicorn[standard]" python-multipart
WORKDIR /srv
COPY app.py /srv/app.py
ENV PORT=10000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
