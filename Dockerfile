FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create cache directory
RUN mkdir -p ./temp_indexes

# Set environment variables
ENV PORT=10000
ENV HOST=0.0.0.0
ENV DEBUG=False

# Expose the port
EXPOSE 10000

# Command to run the application
CMD ["uvicorn", "api_app:app", "--host", "0.0.0.0", "--port", "10000"]