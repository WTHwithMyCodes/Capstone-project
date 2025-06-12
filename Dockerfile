# Dockerfile (Versi Final dengan Semua Dependensi Sistem)

# Gunakan base image Python versi 3.9 yang ringan
FROM python:3.9-slim

# Set direktori kerja di dalam container
WORKDIR /app

# Perbarui package manager (apt) dan instal SEMUA dependensi sistem yang dibutuhkan
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    # Untuk OpenCV (memperbaiki error libGL.so.1)
    libgl1-mesa-glx \
    # Untuk pyzbar (memperbaiki error zbar shared library)
    libzbar0 \
    # Untuk error libgthread-2.0.so.0 (dependensi Glib)
    libglib2.0-0 \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Salin file requirements.txt terlebih dahulu untuk caching
COPY requirements.txt .

# Install semua pustaka Python yang dibutuhkan
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua sisa kode proyek Anda ke dalam container
COPY . .

# Beritahu Docker port mana yang akan diekspos oleh aplikasi
EXPOSE $PORT

# Perintah untuk menjalankan aplikasi saat container dimulai
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "--workers", "1", "app:app"]
