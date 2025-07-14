FROM python:3.8-slim

# Install essential system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app
WORKDIR /app

# Expose port for Streamlit
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "app/main.py"]
