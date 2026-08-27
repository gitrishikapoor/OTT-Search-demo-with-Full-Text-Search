# Use lightweight python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency definition
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and frontend
COPY app.py .
COPY index.html .

# Expose port
EXPOSE 8080

# Run FastAPI app using uvicorn on Cloud Run's dynamic $PORT env variable
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
