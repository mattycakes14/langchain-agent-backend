# This tells Railway how to run your app

# Use Python 3.11
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python packages
RUN pip install -r requirements.txt

# Copy your source code
COPY src/ ./src/

# Expose port 8000 (where your FastAPI runs)
EXPOSE 8000

# Command to start your app
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
