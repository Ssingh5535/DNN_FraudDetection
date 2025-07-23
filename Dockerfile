# 1. Base image: lightweight Python
FROM python:3.9-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your FastAPI app and the trained-model folder
COPY app.py .
COPY ../model ./model

# 5. Expose port 80 (inside container)
EXPOSE 80

# 6. Start the Uvicorn server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
