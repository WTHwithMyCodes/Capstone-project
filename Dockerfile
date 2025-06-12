# Use an official lightweight Python image
FROM python:3.12-slim

# Environment variables to reduce Docker image size and enable real-time logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Expose the port Railway will use
EXPOSE 5000

# Default command — modify if your app uses Flask, FastAPI, etc.
CMD ["flask", "run", "host=0.0.0.0", "--port=8000"]

