import sqlite3

conn = sqlite3.connect(
    "database.db"
)

cursor = conn.cursor()

cursor.execute(

    """
    CREATE TABLE IF NOT EXISTS detections (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        status TEXT,

        akses TEXT,

        hari TEXT,

        tanggal TEXT,

        jam TEXT

    )
    """

)

conn.commit()

conn.close()

print("DATABASE BERHASIL DIBUAT")