FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install ESP-IDF
RUN mkdir -p /esp && cd /esp \
    && git clone -b v5.1 --recursive https://github.com/espressif/esp-idf.git \
    && cd esp-idf \
    && ./install.sh

# Set environment variables
ENV PATH=$PATH:/esp/esp-idf/tools
ENV IDF_PATH=/esp/esp-idf

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app
WORKDIR /app

# Build the firmware
RUN . $IDF_PATH/export.sh && idf.py build

# Run the application
CMD ["idf.py", "monitor"]
