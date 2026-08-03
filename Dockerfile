FROM python:3.10-slim

WORKDIR /code

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Hugging Face Spaces require Port 7860
EXPOSE 7860

# Run Flask backend on 0.0.0.0:7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
