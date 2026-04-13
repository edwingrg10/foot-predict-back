FROM python:3.11-slim

# Herramientas base necesarias para playwright install --with-deps
RUN apt-get update && apt-get install -y \
    curl wget gnupg sudo \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Instala Chromium + TODAS sus dependencias de sistema automáticamente
RUN playwright install --with-deps chromium

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
