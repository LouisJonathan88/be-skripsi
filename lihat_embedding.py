# """
# Script bantu untuk BAB 3 — mengambil contoh nilai vektor embedding
# dari database guna ditampilkan sebagai bukti implementasi pada subbab
# Metode Ekstraksi Fitur FaceNet.

# Cara pakai:
#     python lihat_embedding.py

# Pastikan file .env (DB_HOST/DB_USER/DB_PASS/DB_NAME) sudah terisi
# sesuai konfigurasi server kamu, dan jalankan script ini dari folder
# yang sama dengan db.py (sisi server, di mana koneksi MySQL berada).
# """

# import numpy as np
# from db import load_semua_embedding


# def main():
#     embeddings = load_semua_embedding()

#     if not embeddings:
#         print("Tidak ada data embedding di database.")
#         print("Pastikan minimal sudah ada satu pengguna yang melakukan registrasi.")
#         return

#     # Ambil satu contoh saja — pengguna pertama dalam dictionary
#     nomor_identitas = list(embeddings.keys())[0]
#     vector = embeddings[nomor_identitas]

#     print("=" * 60)
#     print("CONTOH VEKTOR EMBEDDING UNTUK BAB 3")
#     print("=" * 60)
#     print(f"Nomor Identitas      : {nomor_identitas}")
#     print(f"Dimensi Embedding    : {vector.shape[0]}")
#     print(f"Tipe Data            : {vector.dtype}")
#     print()
#     print("10 Nilai Pertama:")
#     print(np.array2string(vector, precision=4, separator=', '))
#     print()
#     print("Format siap-tempel untuk narasi skripsi:")
#     nilai_str = ", ".join(f"{v:.4f}" for v in vector[:10])
#     print(f"[{nilai_str}, ...]")
#     print("=" * 60)


# if __name__ == "__main__":
#     main()




"""
Script bantu untuk BAB 3 — mengambil contoh nilai vektor embedding
dari database (embedding referensi/rata-rata) maupun dari folder
faces_aligned_resized (embedding individual per citra, 1 per jarak).

Cara pakai:
    python lihat_embedding.py

Pastikan file .env sudah terisi dan jalankan dari folder yang sama
dengan db.py dan folder faces_aligned_resized.
"""

import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from flask.cli import load_dotenv
from db import load_semua_embedding

load_dotenv()

# ════════════════════════════════
#  HELPER FORMAT VEKTOR
# ════════════════════════════════

def format_vektor(vector: np.ndarray, n_awal: int = 5) -> str:
    """Format vektor embedding: [v1, v2, v3, v4, v5, ..., v_last]"""
    awal  = ", ".join(f"{v:.5f}" for v in vector[:n_awal])
    akhir = f"{vector[-1]:.5f}"
    return f"[ {awal}, ..., {akhir} ]"


# ════════════════════════════════
#  EMBEDDING INDIVIDUAL (1 PER JARAK)
# ════════════════════════════════

def ambil_embedding_per_jarak(folder_pengguna: str) -> list:
    """
    Ambil 1 embedding dari tiap subfolder jarak (1m, 2m, 3m).
    Mengembalikan list of (label, vector).
    """
    device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    jalur_pengguna = os.path.join("faces_aligned_resized", folder_pengguna)
    if not os.path.exists(jalur_pengguna):
        print(f"Folder tidak ditemukan: {jalur_pengguna}")
        return []

    hasil = []

    for jarak in sorted(os.listdir(jalur_pengguna)):
        jalur_jarak = os.path.join(jalur_pengguna, jarak)
        if not os.path.isdir(jalur_jarak):
            continue

        # Ambil citra pertama saja dari setiap folder jarak
        for nama_gambar in sorted(os.listdir(jalur_jarak)):
            if not nama_gambar.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            jalur_gambar = os.path.join(jalur_jarak, nama_gambar)
            img          = cv2.imread(jalur_gambar)
            if img is None:
                continue

            img      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img      = cv2.resize(img, (160, 160))
            img_norm = (img / 255.0 - 0.5) * 2.0

            img_tensor = torch.tensor(
                img_norm.transpose(2, 0, 1),
                dtype=torch.float32
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = model_facenet(img_tensor)

            vector = embedding.cpu().numpy().flatten()
            label  = f"Citra jarak {jarak} ({nama_gambar})"
            hasil.append((label, vector))
            break  # Hanya ambil 1 citra per jarak

    return hasil


# ════════════════════════════════
#  MAIN
# ════════════════════════════════

def main():
    # ── 1. Embedding referensi dari database ──
    print("=" * 65)
    print("EMBEDDING REFERENSI (HASIL RATA-RATA) — dari Database")
    print("=" * 65)

    embeddings = load_semua_embedding()
    if not embeddings:
        print("Tidak ada data embedding di database.")
        return

    nomor_identitas  = list(embeddings.keys())[0]
    vector_referensi = embeddings[nomor_identitas]

    print(f"Nomor Identitas   : {nomor_identitas}")
    print(f"Dimensi Embedding : {vector_referensi.shape[0]}")

    # ── 2. Embedding individual per jarak ──
    print()
    print("=" * 65)
    print("EMBEDDING INDIVIDUAL (PER JARAK) — dari Folder")
    print("=" * 65)

    folder_pengguna = None
    if os.path.exists("faces_aligned_resized"):
        for folder in os.listdir("faces_aligned_resized"):
            if folder.startswith(nomor_identitas):
                folder_pengguna = folder
                break

    if folder_pengguna is None:
        print(f"Folder faces_aligned_resized/{nomor_identitas}_* tidak ditemukan.")
        return

    print(f"Folder pengguna   : {folder_pengguna}")
    print()

    contoh_individual = ambil_embedding_per_jarak(folder_pengguna)

    if not contoh_individual:
        print("Tidak ada citra yang berhasil dibaca.")
        return

    for label, vector in contoh_individual:
        print(f"{label}:")
        print(format_vektor(vector))
        print()

    # ── 3. Contoh perhitungan rata-rata (3 citra pertama) ──
    print("=" * 65)
    print("CONTOH PERHITUNGAN RATA-RATA (5 NILAI PERTAMA)")
    print("(Ilustrasi dari 3 citra, bukan keseluruhan 225)")
    print("=" * 65)

    vectors = [v for _, v in contoh_individual]
    print("Nilai pertama dari tiap embedding individual:")
    for i, (label, vector) in enumerate(contoh_individual):
        nilai = ", ".join(f"{v:.5f}" for v in vector[:5])
        print(f"  E{i+1}: [ {nilai}, ... ]")

    print()
    rata = np.mean(np.stack(vectors), axis=0)
    nilai_rata = ", ".join(f"{v:.5f}" for v in rata[:5])
    print(f"  Rata-rata E1..E{len(vectors)}: [ {nilai_rata}, ... ]")
    print()
    print("  (Dalam sistem sesungguhnya, rata-rata dihitung dari")
    print(f"   seluruh 225 embedding, bukan hanya {len(vectors)} citra ini)")

    # ── 4. Ringkasan siap-tempel ──
    print()
    print("=" * 65)
    print("RINGKASAN SIAP-TEMPEL UNTUK NARASI SKRIPSI")
    print("=" * 65)
    print()
    for label, vector in contoh_individual:
        print(f"  {label}:")
        print(f"  {format_vektor(vector)}")
        print()
    print(f"  Embedding Referensi (hasil rata-rata 225 citra):")
    print(f"  {format_vektor(vector_referensi)}")
    print("=" * 65)


if __name__ == "__main__":
    main()