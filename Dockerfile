FROM python:3.11-slim

# Install only essential system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements_optimized.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_optimized.txt

# Copy application code
COPY main_optimized.py .

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "main_optimized.py"]