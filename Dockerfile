# Dockerfile (Multi-stage build for a lightweight and secure image)

# Stage 1: Build stage
FROM python:3.10-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.10-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY . .
# Expose port if needed (e.g., for a web service)
# EXPOSE 8000
CMD ["python", "app/main.py"]
