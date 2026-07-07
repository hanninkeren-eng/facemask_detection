from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import sqlite3
import pygame
import pandas as pd
import threading
import time
from datetime import datetime
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ==============================================================================
# AUDIO INITIALIZATION & CHANNELS (Anti Gema Total)
# ==============================================================================
pygame.mixer.pre_init(44100, -16, 2, 512)
pygame.mixer.init()

alarm_sound = pygame.mixer.Sound("alarm.mp3")
tolak_sound = pygame.mixer.Sound("akses_ditolak.mp3")
masuk_sound = pygame.mixer.Sound("silahkan_masuk.mp3")

channel_bicara = pygame.mixer.Channel(1)
channel_alarm = pygame.mixer.Channel(2)

# =========================
# MACHINE LEARNING MODEL
# =========================
model = load_model("mask_detector.h5")

# =========================
# FACE DETECTOR (HAAR CASCADE)
# =========================
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# =========================
# HARDWARE VIDEO CAPTURE
# =========================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# ==============================================================================
# RUNTIME GLOBAL VARIABLES & TIMERS
# ==============================================================================
door_status = "none"
system_active = True

# Cooldown simpan database agar tidak menimbun data dalam 1 detik (5 detik per input)
last_mask_save_time = 0
last_nomask_save_time = 0

# Waktu penanda kapan pintu mulai dibuka
door_open_until = 0

# Penanda status audio sedang aktif
audio_sedang_berbicara = False

# =========================
# DATABASE OPERATIONS
# =========================
def save_database(status, akses):
    now = datetime.now()
    try:
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT,
                akses TEXT,
                hari TEXT,
                tanggal TEXT,
                jam TEXT
            )
        ''')

        cursor.execute(
            """
            INSERT INTO detections (status, akses, hari, tanggal, jam)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                status,
                akses,
                now.strftime("%A"),
                now.strftime("%d-%m-%Y"),
                now.strftime("%H:%M:%S")
            )
        )
        conn.commit()
        print(f"[SUCCESS] Berhasil menginput data: {status} ke database.")
    except Exception as e:
        print(f"Database Error: {e}")
    finally:
        conn.close()

# ==============================================================================
# AUDIO WORKER (Mengunci Gema Menggunakan Status Thread)
# ==============================================================================
def putar_suara_aman(status_terdeteksi):
    global audio_sedang_berbicara
    
    if channel_bicara.get_busy() or audio_sedang_berbicara:
        return

    audio_sedang_berbicara = True
    try:
        if status_terdeteksi == "MASK":
            channel_alarm.stop() 
            channel_bicara.play(masuk_sound)
            time.sleep(3.0) 
            
        elif status_terdeteksi == "NO MASK":
            channel_bicara.play(tolak_sound)
            time.sleep(2.5)
            # Jalankan sirine jika kondisi masih close dan tidak sedang berbunyi
            if door_status == "close" and system_active and not channel_alarm.get_busy():
                channel_alarm.play(alarm_sound, loops=-1)
    except Exception as e:
        print(f"Audio Error: {e}")
    finally:
        audio_sedang_berbicara = False

# ==============================================================================
# MAIN VIDEO GENERATOR STREAM
# ==============================================================================
def generate_frames():
    global door_status, system_active, last_mask_save_time, last_nomask_save_time, door_open_until

    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)

        if not system_active:
            channel_bicara.stop()
            channel_alarm.stop()
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (25, 20, 20), -1)
            cv2.putText(
                frame, "SISTEM NONAKTIF", (80, int(frame.shape[0]/2)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3
            )
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Turunkan skala deteksi ke 1.2 agar pembacaan wajah jauh lebih sensitif
        faces = face_detector.detectMultiScale(gray, 1.2, 5)

        current_time = time.time()

        # JIKA PINTU SEDANG DALAM MASA DELAY TERBUKA, JANGAN RESET STATUS KE "NONE"
        if current_time < door_open_until:
            door_status = "open"
        else:
            # Jika masa delay 5 detik sudah habis, kembalikan status ke standby/close
            if len(faces) == 0:
                door_status = "none"
                if not channel_bicara.get_busy():
                    channel_alarm.stop()

        wajah_terpilih = None
        max_area = 0
        for (x, y, w, h) in faces:
            area = w * h
            if area > max_area:
                max_area = area
                wajah_terpilih = (x, y, w, h)

        if wajah_terpilih is not None:
            x, y, w, h = wajah_terpilih
            face_img = frame[y:y+h, x:x+w]
            
            img = cv2.resize(face_img, (224, 224))
            img = img / 255.0
            img = np.reshape(img, [1, 224, 224, 3])

            pred = model.predict(img, verbose=0)
            
            # Membaca output probability tunggal model h5
            confidence_score = float(pred[0][0])

            # CATATAN: Jika terbalik (pakai masker malah NO MASK), ubah >= menjadi <
            if confidence_score >= 0.5:
                label = "MASK"
                akses = "DITERIMA"
                color = (74, 78, 19)   # Navy-Emerald Green
                
                # AKTIFKAN DELAY PINTU TERBUKA SELAMA 5 DETIK KEDEPAN
                door_status = "open"
                door_open_until = current_time + 5.0 

                # Putar suara pengingat masuk
                threading.Thread(target=putar_suara_aman, args=("MASK",), daemon=True).start()
                
                # SISTEM INPUT DATABASE PASTI MASUK (Cooldown per 5 detik agar data tidak menumpuk)
                if current_time - last_mask_save_time > 5:
                    threading.Thread(target=save_database, args=(label, akses), daemon=True).start()
                    last_mask_save_time = current_time
            else:
                label = "NO MASK"
                akses = "DITOLAK"
                color = (0, 0, 255)    # Merah Terang
                
                # Jika tidak pakai masker dan tidak dalam masa delay pintu terbuka, kunci pintu
                if current_time >= door_open_until:
                    door_status = "close"

                threading.Thread(target=putar_suara_aman, args=("NO MASK",), daemon=True).start()
                
                if current_time - last_nomask_save_time > 5:
                    threading.Thread(target=save_database, args=(label, akses), daemon=True).start()
                    last_nomask_save_time = current_time

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            cv2.putText(frame, f"TARGET: {label}", (x, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# =========================
# FLASK CONTROLLER ROUTES
# =========================
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_system')
def toggle_system():
    global system_active, door_status, door_open_until
    system_active = not system_active
    if not system_active:
        door_status = "none"
        door_open_until = 0
    return jsonify({"active": system_active})

@app.route('/realtime')
def realtime():
    total_mask = 0
    total_nomask = 0
    table = "<table class='table'><tr><td>Belum ada record data terdeteksi.</td></tr></table>"
    
    try:
        conn = sqlite3.connect("database.db")
        df = pd.read_sql_query("SELECT id, status, akses, hari, tanggal, jam FROM detections ORDER BY id DESC", conn)
        conn.close()

        if not df.empty:
            total_mask = len(df[df["status"] == "MASK"])
            total_nomask = len(df[df["status"] == "NO MASK"])
            table = df.to_html(classes="table", index=False)
    except Exception as e:
        print(f"Gagal memuat database: {e}")

    return jsonify({
        "mask": total_mask,
        "nomask": total_nomask,
        "table": table,
        "door": door_status,
        "system": system_active
    })

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)