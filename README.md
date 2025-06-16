#  Sistem Presensi Digital Cerdas Berbasis Pengenalan Wajah dan Kode QR

## Latar Belakang

Proyek ini berpusat pada pengembangan sebuah Sistem Presensi Digital Cerdas, sebuah inovasi yang dirancang untuk menjawab tantangan fundamental dalam sistem absensi konvensional: inefisiensi dan kerawanan terhadap kecurangan. Latar belakang utama dari proyek ini adalah pengamatan terhadap praktik "titip absen" yang masih marak di lingkungan akademik dan korporat, yang tidak hanya merusak integritas data tetapi juga mendemotivasi individu yang jujur. Selain itu, proses rekapitulasi data yang dilakukan secara manual sangat memakan waktu, rentan terhadap kesalahan input (human error), dan memperlambat proses administrasi penting seperti penggajian atau pelaporan akademik.

Untuk mengatasi masalah ini, rencana proyek berfokus pada implementasi solusi multimodal yang menyediakan dua jalur otentikasi fleksibel. Pertama, verifikasi biometrik melalui pengenalan wajah real-time yang menawarkan tingkat keamanan tertinggi dengan memastikan kehadiran fisik individu secara unik. Kedua, pemindaian Kode QR yang dirancang untuk memberikan alternatif presensi yang sangat cepat dan efisien, ideal untuk situasi yang membutuhkan kecepatan proses. Keunggulan utama dari solusi yang diusulkan adalah aksesibilitas dan efisiensi biaya; dengan hanya mengandalkan webcam standar yang umum tersedia, sistem ini meniadakan kebutuhan akan investasi pada perangkat keras biometrik khusus yang mahal, menjadikannya pilihan yang realistis bagi institusi pendidikan dan usaha kecil menengah.

## Meet the Team

| Team ID     | CC25-CR427                              |
|-------------|-----------------------------------------|


Tim kami terdiri dari sekelompok kreator, pengembang, dan ahli strategi yang beragam. Kenali kami!


| Learning Path |  ID       | Name                    | University                         | Status   |
|------|------------------|-------------------------|------------------------------------|----------|
| Machine Learning   | MC858D5Y1366     | Firman Fitrah Ramadhan         | Universitas Brawijaya   | Active   |
| Machine Learning   | MC149D5Y0528     | Kamal Abdurrohman         | STT Pati   | Active   |



## Tech Stack

Proyek ini menggunakan berbagai alat dan teknologi untuk mewujudkan ide:
- **Languange:** Python (versi 3.8.10), JavaScript, CSS.
- **Framework:** Flask, Supabase
- **Library:** Opencv, DeepFace, TensorFlow, pyzbar.

---

### Installation

1. Clone the repo 
   ```sh
   https://github.com/ProJustitia/Capstone-project.git
   cd Capstone-project
   ```
2. Membuat virtual env  dan installasi depedensi
    ```
    python -m venv analisis_data
    source analisis_data/bin/activate # linux
    analisis_data/Scripts/activate    # Windows
    pip install -r requirements.txt
    ```
3. Jalankan kode utama
   ```
   python app.py
   #atau
   flask run
   ```
