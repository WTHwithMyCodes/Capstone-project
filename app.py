from flask import Flask, render_template, request, session, redirect, url_for, Response, jsonify, flash
import mysql.connector
import cv2
from PIL import Image
import os
import time
from datetime import date
from pyzbar.pyzbar import decode
import numpy as np
from deepface import DeepFace
import pickle
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWD = os.getenv("DB_PASSWD") 
DB_NAME = os.getenv("DB_NAME", "flask_db")
DB_PORT = int(os.getenv("DB_PORT", 3306))

if not DB_PASSWD:
    raise ValueError("Error: Password database (DB_PASSWD) tidak ditemukan di file .env")


HAAR_CASCADE_PATH = os.path.join(os.path.dirname(__file__), "resources", "haarcascade_frontalface_default.xml")
if not os.path.exists(HAAR_CASCADE_PATH):
    HAAR_CASCADE_PATH = "/home/firman/ESP32cam-Flask/resources/haarcascade_frontalface_default.xml"
    if not os.path.exists(HAAR_CASCADE_PATH):
        print(f"ERROR: File Haar Cascade tidak ditemukan di {HAAR_CASCADE_PATH} atau di direktori 'resources'. Harap perbaiki path.")

DATASET_DIR = "./dataset"
EMBEDDINGS_FILE = "known_faces_embeddings.pkl"

app = Flask(__name__)
app.secret_key = "kunci_rahasia_"

cnt = 0
pause_cnt = 0
justscanned = False

facenet_model = None
known_face_embeddings = {}
prs_nbr_to_name_skill = {}

def get_db_connection():
    """Establishes a connection to the MySQL database."""
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            passwd=DB_PASSWD,
            database=DB_NAME,
            port=DB_PORT,
            auth_plugin='mysql_native_password'
        )
        return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to database: {err}")
        return None

# --- FaceNet & Embeddings Functions ---
def load_facenet_resources():
    """
    Loads the FaceNet model and known face embeddings from a pickle file.
    Also populates the prs_nbr_to_name_skill map from the database.
    """
    global facenet_model, known_face_embeddings, prs_nbr_to_name_skill
    try:
        facenet_model = DeepFace.build_model("Facenet")
        # Load known face embeddings
        if os.path.exists(EMBEDDINGS_FILE):
            with open(EMBEDDINGS_FILE, 'rb') as f:
                known_face_embeddings = pickle.load(f)
            print(f"Embeddings wajah yang diketahui dimuat dari {EMBEDDINGS_FILE}")
        else:
            print(f"File embeddings {EMBEDDINGS_FILE} tidak ditemukan. Perlu di-generate.")

        # Load mapping prs_nbr to name and skill for quick lookup
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT prs_nbr, prs_name, prs_skill FROM prs_mstr")
            for prs_nbr, prs_name, prs_skill in cursor.fetchall():
                prs_nbr_to_name_skill[str(prs_nbr)] = (prs_name, prs_skill)
            cursor.close()
            conn.close()

    except Exception as e:
        print(f"Error loading FaceNet resources: {e}")
        print("Pastikan deepface terinstal dan model dapat diunduh jika belum ada.")

# Load FaceNet resources when the app context is available
with app.app_context():
    load_facenet_resources()
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        print(f"Direktori dataset dibuat di {DATASET_DIR}")


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Generate dataset (REVISED) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Generate dataset (REVISED with Haar Cascade) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def generate_dataset(nbr):
    """
    REVISED: Captures 5 face images using Haar Cascade for detection to ensure
    consistency with the real-time recognition method.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak bisa membuka kamera.")
        yield b"Error: Camera not accessible"
        return

    # 1. Muat Haar Cascade Classifier di sini
    face_classifier = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

    conn = get_db_connection()
    if not conn:
        print("Error: Tidak bisa terhubung ke database untuk generate_dataset.")
        yield b"Error: DB connection failed"
        cap.release()
        return
    
    # Get the starting image ID from the database
    cursor = conn.cursor()
    cursor.execute("SELECT IFNULL(MAX(CAST(img_id AS UNSIGNED)), 0) FROM img_dataset")
    lastid = cursor.fetchone()[0]
    cursor.close()

    img_id = lastid
    count_img = 0
    max_images = 5
    
    capture_delay_seconds = 2.0 
    last_capture_time = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    while count_img < max_images:
        ret, img = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        
        display_frame = img.copy()
        
        # Tampilkan progress di layar
        progress_text = f"Progress: {count_img} / {max_images}"
        cv2.putText(display_frame, progress_text, (10, 30), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        time_since_last_capture = time.time() - last_capture_time
        
        # 2. Lakukan deteksi wajah menggunakan Haar Cascade
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.1, 4)

        # 3. Cek apakah wajah ditemukan
        if len(faces) > 0:
            # Ambil wajah pertama yang ditemukan (asumsikan hanya ada satu orang)
            (x, y, w, h) = faces[0]

            # Gambar kotak di sekitar wajah yang terdeteksi
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Kondisi untuk mengambil gambar: waktu jeda sudah cukup
            if time_since_last_capture >= capture_delay_seconds:
                count_img += 1
                img_id += 1
                
                # Potong dan ubah ukuran wajah untuk disimpan
                cropped_face = img[y:y + h, x:x + w]
                face_to_save = cv2.resize(cropped_face, (160, 160))
                
                # Simpan file gambar
                file_name_path = os.path.join(DATASET_DIR, f"{nbr}.{img_id}.jpg")
                cv2.imwrite(file_name_path, face_to_save)
                
                # Update database
                cursor = conn.cursor()
                try:
                    cursor.execute("INSERT INTO img_dataset (img_id, img_person) VALUES (%s, %s)", (img_id, nbr))
                    conn.commit()
                except mysql.connector.Error as err:
                    print(f"DB Error during dataset generation: {err}")
                finally:
                    cursor.close()
                
                # Update waktu pengambilan terakhir
                last_capture_time = time.time()
                
                cv2.putText(display_frame, f"Gambar {count_img} TERSIMPAN!", (x, y - 10), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            else:
                # Tampilkan pesan untuk menunggu dan bergerak
                wait_time = int(capture_delay_seconds - time_since_last_capture) + 1
                instruction = "Gerakkan sedikit kepala Anda"
                cv2.putText(display_frame, instruction, (10, 60), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(display_frame, f"Ambil gambar dalam: {wait_time}", (x, y - 10), font, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

        else: # Jika tidak ada wajah yang terdeteksi
            cv2.putText(display_frame, "Arahkan wajah ke kamera", (10, 60), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            last_capture_time = 0 # Reset timer jika wajah hilang
            
        # Stream frame ke browser
        frame_bytes = cv2.imencode('.jpg', display_frame)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        if cv2.waitKey(1) == 27:
            break

    # Tampilkan pesan selesai setelah loop berakhir
    final_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(final_frame, "Pengambilan data selesai!", (100, 240), font, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
    frame_bytes = cv2.imencode('.jpg', final_frame)[1].tobytes()
    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    time.sleep(2)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if conn.is_connected():
        conn.close()
    print("Proses generate dataset selesai.")

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Enroll Faces (Process Dataset for FaceNet) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route('/enroll_faces')
def enroll_faces_route():
    """
    Processes images in the DATASET_DIR, generates FaceNet embeddings,
    and stores them in known_face_embeddings.pkl and in memory.
    """
    global known_face_embeddings, prs_nbr_to_name_skill

    if not os.path.exists(DATASET_DIR) or not os.listdir(DATASET_DIR):
        flash("Direktori dataset kosong. Silakan tambahkan data wajah terlebih dahulu.", "error")
        return redirect(url_for('home'))

    # Reset in-memory data before re-enrollment
    known_face_embeddings_new = {}
    prs_nbr_to_name_skill_new = {}

    # Fetch person data from the database
    conn = get_db_connection()
    if not conn:
        flash("Koneksi database gagal.", "error")
        return redirect(url_for('home'))

    cursor = conn.cursor(dictionary=True) # Use dictionary=True for column access by name
    cursor.execute("SELECT prs_nbr, prs_name, prs_skill FROM prs_mstr")
    persons_in_db = {str(row['prs_nbr']): (row['prs_name'], row['prs_skill']) for row in cursor.fetchall()}
    cursor.close()
    conn.close()

    enrolled_count = 0
    img_processed_count = 0

    print("Memulai proses enrollment wajah...")
    for image_name in os.listdir(DATASET_DIR):
        if image_name.endswith(".jpg") or image_name.endswith(".png"):
            try:
                prs_nbr = str(image_name.split(".")[0]) # Extract person number from filename
                image_path = os.path.join(DATASET_DIR, image_name)

                if prs_nbr not in persons_in_db:
                    print(f"Skipping {image_name}: Person ID {prs_nbr} tidak ditemukan di database prs_mstr.")
                    continue

                # Generate embedding using DeepFace
                # enforce_detection=False because images in DATASET_DIR should already be cropped faces
                # Kode yang BENAR
                embedding_objs = DeepFace.represent(img_path=cropped_face, model_name='Facenet', enforce_detection=False)

                if embedding_objs and isinstance(embedding_objs, list) and 'embedding' in embedding_objs[0]:
                    embedding = embedding_objs[0]['embedding']
                    if prs_nbr not in known_face_embeddings_new:
                        known_face_embeddings_new[prs_nbr] = []
                    known_face_embeddings_new[prs_nbr].append(np.array(embedding))
                    img_processed_count +=1

                    # Update name and skill from DB if not already present for this person
                    if prs_nbr not in prs_nbr_to_name_skill_new:
                        prs_nbr_to_name_skill_new[prs_nbr] = persons_in_db[prs_nbr]

                    # Count unique persons enrolled
                    if prs_nbr in known_face_embeddings_new and len(known_face_embeddings_new[prs_nbr]) == 1:
                        enrolled_count +=1
                else:
                    print(f"Tidak ada wajah terdeteksi atau embedding tidak bisa dihasilkan dari {image_name}")

            except Exception as e:
                print(f"Error processing {image_name} untuk enrollment: {e}")

    if not known_face_embeddings_new:
        flash("Tidak ada wajah yang berhasil di-enroll. Pastikan gambar dataset valid dan wajah terdeteksi.", "warning")
        return redirect(url_for('home'))

    # Save embeddings to file
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(known_face_embeddings_new, f)

    # Update global variables with new data
    known_face_embeddings = known_face_embeddings_new
    prs_nbr_to_name_skill = prs_nbr_to_name_skill_new

    flash(f"{enrolled_count} person dengan total {img_processed_count} gambar wajah berhasil di-enroll dan embeddings disimpan.", "success")
    print(f"Enrollment selesai. {len(known_face_embeddings)} persons enrolled.")
    return redirect(url_for('home'))


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Face Recognition with FaceNet >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Face Recognition with FaceNet (REVISED) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Face Recognition with FaceNet (STABLE HYBRID) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Face Recognition (STABLE HYBRID + DB CHECK) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Face Recognition (FIXED OVERLAPPING TEXT) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Face Recognition (FIXED DATABASE CONFLICT) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Face Recognition (FINAL FIX) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Face Recognition (FINAL FIX + VALIDITY CHECK) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Face Recognition (FIXED HANGING PAGE) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def face_recognition_facenet():
    """
    Final version with a robust, non-conflicting database logic, and fixes
    the hanging page bug when the model file is not generated yet.
    """
    global cnt, pause_cnt, justscanned
    global known_face_embeddings, prs_nbr_to_name_skill, facenet_model

    face_classifier = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    wCam, hCam = 640, 480
    
    # --- PERBAIKAN UTAMA DIMULAI DI SINI ---
    # Cek model di awal SEBELUM membuka kamera
    if not facenet_model or not known_face_embeddings:
        print("Peringatan: Model wajah belum di-enroll. Menampilkan pesan error.")
        img_err = np.zeros((hCam, wCam, 3), dtype=np.uint8)
        
        # Pesan error yang lebih informatif
        message = "Model Wajah Belum Di-Enroll!"
        message2 = "Silakan ke Dashboard dan Enroll Wajah."
        cv2.putText(img_err, message, (50, int(hCam/2) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.putText(img_err, message2, (50, int(hCam/2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        
        frame_bytes = cv2.imencode('.jpg', img_err)[1].tobytes()
        
        # Kirim frame error SATU KALI saja
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Hentikan generator. Ini akan menutup koneksi dan mencegah 'hang'.
        return 
    # --- PERBAIKAN UTAMA SELESAI ---

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak bisa membuka kamera untuk face recognition.")
        yield b"Error: Camera not accessible"
        return
    cap.set(3, wCam)
    cap.set(4, hCam)

    DISTANCE_THRESHOLD = 0.4
    
    current_person = {"identity": None, "name": None, "status_text": None, "status_color": None}
    
    # ... (SISA KODE DI BAWAH INI SAMA PERSIS SEPERTI SEBELUMNYA, TIDAK PERLU DIUBAH) ...
    while True:
        ret, img = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
            
        if justscanned:
            pause_cnt += 1
            if pause_cnt > 40:
                justscanned = False
                pause_cnt = 0
                cnt = 0
                current_person = {"identity": None, "name": None, "status_text": None, "status_color": None}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.1, 4)
        
        identified_person_this_frame = False

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

            try:
                if justscanned:
                    if current_person["status_text"]:
                         cv2.putText(img, current_person["status_text"], (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, current_person["status_color"], 2)
                    continue

                cropped_face = img[y:y + h, x:x + w]
                if cropped_face.size == 0: continue

                embedding_objs = DeepFace.represent(img_path=cropped_face, model_name='Facenet', enforce_detection=False)
                current_embedding = np.array(embedding_objs[0]['embedding'])

                min_distance = float('inf')
                identity = "UNKNOWN"

                for prs_nbr_known, embeddings_list in known_face_embeddings.items():
                    for known_emb in embeddings_list:
                        dot = np.dot(current_embedding, known_emb)
                        norm_current = np.linalg.norm(current_embedding)
                        norm_known = np.linalg.norm(known_emb)
                        if norm_current == 0 or norm_known == 0: continue
                        similarity = dot / (norm_current * norm_known)
                        distance = 1 - similarity

                        if distance < min_distance and distance < DISTANCE_THRESHOLD:
                            min_distance = distance
                            identity = prs_nbr_known
                
                if identity != "UNKNOWN" and identity in prs_nbr_to_name_skill:
                    identified_person_this_frame = True
                    pname, pskill = prs_nbr_to_name_skill[identity]

                    if current_person["identity"] != identity:
                        cnt = 0
                        current_person["identity"] = identity
                        current_person["name"] = pname

                    if int(cnt) < 15:
                        cnt += 1
                        n_percent = (100 / 15) * cnt
                        w_filled = (cnt / 15) * w
                        cv2.putText(img, f'{pname} ({int(n_percent)}%)', (x + 5, y + h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (153, 255, 255), 2)
                        cv2.rectangle(img, (x, y + h + 40), (x + w, y + h + 50), (255, 255, 0), 2)
                        cv2.rectangle(img, (x, y + h + 40), (x + int(w_filled), y + h + 50), (153, 255, 255), cv2.FILLED)
                    else: 
                        if not justscanned:
                            conn = get_db_connection()
                            if conn:
                                try:
                                    cursor = conn.cursor(buffered=True)
                                    cursor.execute("SELECT accs_id FROM accs_hist WHERE accs_prsn = %s AND accs_date = CURDATE()", (identity,))
                                    attendance_record = cursor.fetchone()

                                    if attendance_record:
                                        current_person["status_text"] = f"Sudah Absen: {pname}"
                                        current_person["status_color"] = (0, 255, 255)
                                        print(f"DUPLIKAT: Wajah {pname} ({identity}) terdeteksi.")
                                    else:
                                        cursor.execute("INSERT INTO accs_hist (accs_date, accs_prsn) VALUES (%s, %s)", (str(date.today()), identity))
                                        cursor.execute("UPDATE prs_mstr SET prs_active = 'HADIR' WHERE prs_nbr = %s", (identity,))
                                        conn.commit()
                                        current_person["status_text"] = f"Berhasil: {pname}"
                                        current_person["status_color"] = (50, 255, 50)
                                        print(f"Absensi untuk {pname} ({identity}) berhasil dicatat.")
                                
                                except mysql.connector.Error as err:
                                    print(f"DB Error saat mencatat absensi: {err}")
                                finally:
                                    if conn.is_connected():
                                        cursor.close()
                                        conn.close()
                            
                            justscanned = True 

                        if current_person["status_text"]:
                            cv2.putText(img, current_person["status_text"], (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, current_person["status_color"], 2)
                else:
                    if identity != "UNKNOWN":
                        print(f"Peringatan: Wajah dengan ID '{identity}' dikenali dari file embedding, tetapi tidak ditemukan di database prs_mstr. Harap lakukan enroll ulang.")
                    cv2.putText(img, 'UNKNOWN', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            except Exception as e:
                print(f"Error pada DeepFace.represent: {e}")
                cv2.putText(img, 'Analyzing...', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if not identified_person_this_frame:
            cnt = 0
            current_person["identity"] = None

        frame_bytes = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

        if cv2.waitKey(1) == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Stream face recognition dihentikan.")
    
def qrcode_reader():
    """
    REVISED (FINAL): Reads QR codes, prevents multiple attendance for the same continuous scan,
    and provides clear, persistent feedback.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        yield b"Error: Camera for QR not accessible"
        return

    wCam, hCam = 400, 400
    cap.set(3, wCam)
    cap.set(4, hCam)
    font = cv2.FONT_HERSHEY_SIMPLEX


    while True:
        ret, img = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        current_time = time.time()
        qr_found_in_frame = False

        try:
            for code in decode(img):
                qr_found_in_frame = True
                decoded_data = code.data.decode("utf-8").strip()
                rect_pts = code.rect
                pts = np.array([code.polygon], np.int32)
                

                # --- This is a NEW QR code, process it ---
                cv2.polylines(img, [pts], True, (0, 255, 0), 3) # Green box for new scan
                conn = get_db_connection()
                if not conn:
                    cv2.putText(img, "DB Error", (rect_pts[0], rect_pts[1] - 20), font, 0.8, (0,0,255), 2)
                    break

                cursor = conn.cursor(dictionary=True)
                cursor.execute("SELECT prs_name FROM prs_mstr WHERE prs_nbr = %s", (decoded_data,))
                result = cursor.fetchone()

                if result:
                    pname = result['prs_name']
                    
                    # --- PERUBAHAN DIMULAI DI SINI ---
                    # 1. Cek dulu ke database apakah sudah ada rekor absensi untuk hari ini
                    cursor.execute("SELECT accs_id FROM accs_hist WHERE accs_prsn = %s AND accs_date = CURDATE()", (decoded_data,))
                    attendance_record = cursor.fetchone()

                    if attendance_record:
                        # 2. Jika sudah ada, tampilkan pesan dan jangan lakukan apa-apa lagi
                        print(f"DUPLIKAT: {pname} ({decoded_data}) mencoba scan lagi, status sudah HADIR.")
                        cv2.putText(img, f"Sudah Absen: {pname}", (rect_pts[0], rect_pts[1] - 10), font, 0.7, (0, 255, 255), 2)

                    else:
                        # 3. Jika belum ada, baru lakukan proses absensi seperti biasa
                        try:
                            cursor.execute("INSERT INTO accs_hist (accs_date, accs_prsn) VALUES (%s, %s)", (str(date.today()), decoded_data))
                            cursor.execute("UPDATE prs_mstr SET prs_active = 'HADIR' WHERE prs_nbr = %s", (decoded_data,))
                            conn.commit()
                            
                            print(f"Absensi BERHASIL untuk: {pname} ({decoded_data})")
                            cv2.putText(img, f"Berhasil: {pname}", (rect_pts[0], rect_pts[1] - 10), font, 0.7, (50, 255, 50), 2)
                            
                            session['qr_isSuccess'] = "true"
                            session['qr_username'] = pname

                        except mysql.connector.Error as db_err:
                            print(f"DB Error saat QR absen: {db_err}")
                            cv2.putText(img, "DB Save Error", (rect_pts[0], rect_pts[1] - 20), font, 0.7, (0,0,255), 2)
                    # --- PERUBAHAN SELESAI DI SINI ---
                    
                else:
                    cv2.putText(img, "QR TIDAK DIKENAL", (rect_pts[0], rect_pts[1] - 10), font, 0.7, (0, 0, 255), 2)
                
                cursor.close()
                conn.close()
                break # Process only one QR code per frame

        except Exception as e:
            # print(f"Error during QR processing: {e}") # for debugging
            pass


        # Stream the final composed frame
        frame_bytes = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/getstatus', methods=['GET'])
def getstatus():
    """Returns the status of the last QR code scan (success/failure and username)."""
    status = session.pop('qr_isSuccess', "false")
    name = session.pop('qr_username', "")
    return jsonify(isSuccess=status, username=name), 200

@app.route('/')
def home():
    """Renders the home page with a list of registered persons."""
    conn = get_db_connection()
    if not conn:
        flash("Koneksi database gagal.", "error")
        return render_template('index.html', data=[])

    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT prs_nbr, prs_name, prs_skill, prs_active, DATE_FORMAT(prs_added, '%Y-%m-%d %H:%i:%s') as prs_added FROM prs_mstr")
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('index.html', data=data)


@app.route('/addprsn')
def addprsn():
    """Renders the add person page, generating a new person number."""
    conn = get_db_connection()
    if not conn:
        flash("Koneksi database gagal.", "error")
        return redirect(url_for('home'))

    cursor = conn.cursor()
    # Get the next available person number
    cursor.execute("SELECT IFNULL(MAX(CAST(prs_nbr AS UNSIGNED)), 2) + 1 FROM prs_mstr")
    row = cursor.fetchone()
    nbr = row[0] if row else 3 # Default starting number if no records exist
    cursor.close()
    conn.close()
    return render_template('addprsn.html', newnbr=int(nbr))


@app.route('/addprsn_submit', methods=['POST'])
def addprsn_submit():
    """Handles the submission of new person data and saves it to the database."""
    prsnbr = request.form.get('txtnbr')
    prsname = request.form.get('txtname')
    prsskill = request.form.get('optskill')

    if not all([prsnbr, prsname, prsskill]):
        flash("Semua field harus diisi!", "error")
        return redirect(url_for('addprsn'))

    conn = get_db_connection()
    if not conn:
        flash("Koneksi database gagal.", "error")
        return redirect(url_for('addprsn'))

    cursor = conn.cursor()
    try:
        # Insert new person into prs_mstr
        cursor.execute("INSERT INTO prs_mstr (prs_nbr, prs_name, prs_skill) VALUES (%s, %s, %s)",
                       (prsnbr, prsname, prsskill))
        conn.commit()
        flash(f"Person {prsname} berhasil ditambahkan.", "success")
        # Update in-memory map for quick lookup
        prs_nbr_to_name_skill[str(prsnbr)] = (prsname, prsskill)
    except mysql.connector.Error as err:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"DATABASE INSERT ERROR: {err}")
        print(f"Data yang gagal dimasukkan: NBR={prsnbr}, NAMA={prsname}, SKILL={prsskill}")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        flash(f"Error adding person: {err}", "error")
        # Re-raise the exception to show Flask debug page in debug mode
        raise err
    finally:
        cursor.close()
        conn.close()

    return redirect(url_for('vfdataset_page', prs=prsnbr))


@app.route('/vfdataset_page/<prs>')
def vfdataset_page(prs):
    """Renders the dataset generation page for a specific person."""
    return render_template('gendataset.html', prs=prs)


@app.route('/vidfeed_dataset/<nbr>')
def vidfeed_dataset(nbr):
    """Streams video feed for dataset generation."""
    return Response(generate_dataset(nbr), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed')
def video_feed():
    """Streams video feed for real-time face recognition."""
    return Response(face_recognition_facenet(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/fr_page')
def fr_page():
    """
    Renders the Face Recognition page, displaying today's access history.
    This route now explicitly selects b.prs_nbr as 'nim_from_prs_mstr'.
    """
    conn = get_db_connection()
    if not conn:
        flash("Koneksi database gagal.", "error")
        return render_template('fr_page.html', data=[])

    cursor = conn.cursor(dictionary=True)
    # MODIFIED: Added b.prs_nbr to the SELECT statement
    cursor.execute("SELECT a.accs_id, a.accs_prsn, b.prs_nbr AS nim_from_prs_mstr, b.prs_name, b.prs_skill, DATE_FORMAT(a.accs_added, '%Y-%m-%d %H:%i:%s') as accs_added "
                   "FROM accs_hist a "
                   "LEFT JOIN prs_mstr b ON a.accs_prsn = b.prs_nbr "
                   "WHERE a.accs_date = CURDATE() "
                   "ORDER BY a.accs_id DESC")
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('fr_page.html', data=data)

# AJAX Endpoint to count today's scans
@app.route('/countTodayScan')
def countTodayScan():
    """Returns the count of access history records for today."""
    conn = get_db_connection()
    if not conn: return jsonify({'rowcount': 0, 'error': 'DB connection failed'})

    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM accs_hist WHERE accs_date = CURDATE()")
        rowcount = cursor.fetchone()[0]
        return jsonify({'rowcount': rowcount})
    except mysql.connector.Error as err:
        return jsonify({'rowcount': 0, 'error': str(err)})
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()


@app.route('/loadData', methods=['GET'])
def loadData():
    """
    AJAX endpoint to load today's access history data.
    This route now explicitly selects b.prs_nbr as 'nim_from_prs_mstr'.
    """
    conn = get_db_connection()
    if not conn: return jsonify(response=[], error='DB connection failed')

    cursor = conn.cursor(dictionary=True)
    try:
        # MODIFIED: Added b.prs_nbr to the SELECT statement
        cursor.execute("SELECT a.accs_id, a.accs_prsn, b.prs_nbr AS nim_from_prs_mstr, b.prs_name, b.prs_skill, DATE_FORMAT(a.accs_added, '%H:%i:%s') as time_added "
                       "FROM accs_hist a "
                       "LEFT JOIN prs_mstr b ON a.accs_prsn = b.prs_nbr "
                       "WHERE a.accs_date = CURDATE() "
                       "ORDER BY a.accs_id DESC")
        data = cursor.fetchall()
        return jsonify(response=data)
    except mysql.connector.Error as err:
        return jsonify(response=[], error=str(err))
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

@app.route('/qrcode')
def qrcode():
    """Renders the QR code scanning page."""
    return render_template('qrcode_page.html')


@app.route('/qrcode_video')
def qrcode_video():
    """Streams video feed for QR code reading."""
    return Response(qrcode_reader(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    # Ensure Haar Cascade file exists
    if not os.path.exists(HAAR_CASCADE_PATH):
        print(f"KRITIS: File Haar Cascade '{HAAR_CASCADE_PATH}' tidak ditemukan. Aplikasi mungkin tidak berfungsi dengan benar.")

    # Ensure dataset directory exists
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        print(f"Direktori dataset dibuat: {DATASET_DIR}")

    app.run(host='0.0.0.0', port=5000, debug=True)
