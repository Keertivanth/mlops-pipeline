# Pull the Python image from Docker Hub
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy dependencies before application code so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and model artifacts
COPY . .

EXPOSE 8000

# Start the API server when the container is ready
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]