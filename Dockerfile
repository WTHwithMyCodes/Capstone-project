# Gunakan image Python ringan
FROM python:3.12-slim

# Environment untuk mengurangi ukuran image & enable log real-time
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set direktori kerja
WORKDIR /app

# Upgrade pip dan alat bantu build
RUN pip install --upgrade pip setuptools wheel

# Salin file requirements terlebih dahulu lalu install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file lainnya ke image
COPY . .

# Expose port yang akan digunakan
EXPOSE 8000

# Jalankan Flask
CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]

