FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV MODEL_REPO=/models PORT=5006
EXPOSE 5006
CMD ["python", "server.py"]
