"""
Script bantu untuk BAB 3 — mensimulasikan proses pencocokan wajah
menggunakan Cosine Similarity antara embedding input (satu citra)
dengan embedding referensi dari database.

Cara pakai:
    python hitung_cosine.py

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

THRESHOLD = 0.55


# ════════════════════════════════
#  HELPER
# ════════════════════════════════

def format_vektor(vector: np.ndarray, n_awal: int = 5) -> str:
    awal  = ", ".join(f"{v:.5f}" for v in vector[:n_awal])
    akhir = f"{vector[-1]:.5f}"
    return f"[ {awal}, ..., {akhir} ]"


def hitung_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Hitung Cosine Similarity antara dua vektor embedding."""
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return float(np.dot(a_norm, b_norm))


# ════════════════════════════════
#  EKSTRAK EMBEDDING SATU CITRA
# ════════════════════════════════

def ekstrak_embedding_citra(jalur_gambar: str, device, model) -> np.ndarray:
    """Ekstrak embedding dari satu citra."""
    img = cv2.imread(jalur_gambar)
    if img is None:
        return None

    img      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img      = cv2.resize(img, (160, 160))
    img_norm = (img / 255.0 - 0.5) * 2.0

    img_tensor = torch.tensor(
        img_norm.transpose(2, 0, 1),
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(img_tensor)

    return embedding.cpu().numpy().flatten()


# ════════════════════════════════
#  MAIN
# ════════════════════════════════

def main():
    device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # ── 1. Muat embedding referensi dari database ──
    embeddings_db = load_semua_embedding()
    if not embeddings_db:
        print("Tidak ada embedding di database.")
        return

    # ── 2. Ambil satu citra input (simulasi presensi) ──
    # Cari folder pengguna pertama di faces_aligned_resized
    nomor_identitas_db = list(embeddings_db.keys())[0]
    folder_pengguna    = None

    if os.path.exists("faces_aligned_resized"):
        for folder in os.listdir("faces_aligned_resized"):
            if folder.startswith(nomor_identitas_db):
                folder_pengguna = folder
                break

    if folder_pengguna is None:
        print("Folder faces_aligned_resized tidak ditemukan.")
        return

    # Ambil satu citra dari jarak 2m sebagai simulasi input presensi
    jalur_citra_input = None
    nama_citra_input  = None
    jalur_2m          = os.path.join("faces_aligned_resized", folder_pengguna, "2m")

    if os.path.exists(jalur_2m):
        for nama_gambar in sorted(os.listdir(jalur_2m)):
            if nama_gambar.lower().endswith((".jpg", ".jpeg", ".png")):
                jalur_citra_input = os.path.join(jalur_2m, nama_gambar)
                nama_citra_input  = nama_gambar
                break

    if jalur_citra_input is None:
        print("Citra input tidak ditemukan di folder 2m.")
        return

    # ── 3. Ekstrak embedding input ──
    embedding_input = ekstrak_embedding_citra(jalur_citra_input, device, model_facenet)
    if embedding_input is None:
        print("Gagal mengekstrak embedding dari citra input.")
        return

    # ── 4. Hitung Cosine Similarity dengan semua referensi ──
    print("=" * 65)
    print("SIMULASI PENCOCOKAN WAJAH — Cosine Similarity")
    print("=" * 65)
    print(f"Citra input       : {folder_pengguna}/2m/{nama_citra_input}")
    print(f"Embedding input   : {format_vektor(embedding_input)}")
    print()

    hasil_cocok = []
    for nomor_identitas, embedding_ref in embeddings_db.items():
        similarity = hitung_cosine_similarity(embedding_input, embedding_ref)
        hasil_cocok.append((nomor_identitas, similarity, embedding_ref))

    # Urutkan dari similarity tertinggi
    hasil_cocok.sort(key=lambda x: x[1], reverse=True)
    nomor_terbaik, similarity_terbaik, embedding_ref_terbaik = hasil_cocok[0]

    # ── 5. Tampilkan perhitungan konkret ──
    print("PERHITUNGAN COSINE SIMILARITY (Berdasarkan Rumus 2.9)")
    print("-" * 65)
    print(f"Embedding Input (A)   : {format_vektor(embedding_input)}")
    print(f"Embedding Referensi (B): {format_vektor(embedding_ref_terbaik)}")
    print()

    # Hitung komponen rumus secara terpisah
    a_norm      = embedding_input / np.linalg.norm(embedding_input)
    b_norm      = embedding_ref_terbaik / np.linalg.norm(embedding_ref_terbaik)
    dot_product = float(np.dot(a_norm, b_norm))
    norm_a      = float(np.linalg.norm(embedding_input))
    norm_b      = float(np.linalg.norm(embedding_ref_terbaik))

    print(f"‖A‖ (magnitudo A)     : {norm_a:.5f}")
    print(f"‖B‖ (magnitudo B)     : {norm_b:.5f}")
    print(f"A · B (dot product A dan B yang sudah dinormalisasi): {dot_product:.5f}")
    print()
    print(f"Cosine Similarity = (A · B) / (‖A‖ × ‖B‖)")
    print(f"                  = {dot_product:.5f}")
    print(f"                    (setelah normalisasi L2, dot product = similarity)")
    print()

    # ── 6. Keputusan threshold ──
    print("KEPUTUSAN THRESHOLD")
    print("-" * 65)
    print(f"Nilai Similarity Tertinggi : {similarity_terbaik:.5f}")
    print(f"Threshold                  : {THRESHOLD}")
    print()

    if similarity_terbaik >= THRESHOLD:
        print(f"Similarity ({similarity_terbaik:.5f}) >= Threshold ({THRESHOLD})")
        print(f"→ Wajah DIKENALI sebagai nomor identitas: {nomor_terbaik}")
    else:
        print(f"Similarity ({similarity_terbaik:.5f}) < Threshold ({THRESHOLD})")
        print(f"→ Wajah TIDAK DIKENALI (UNKNOWN)")

    # ── 7. Ringkasan siap-tempel ──
    print()
    print("=" * 65)
    print("RINGKASAN SIAP-TEMPEL UNTUK NARASI SKRIPSI")
    print("=" * 65)
    print(f"  Embedding Input (A)    : {format_vektor(embedding_input)}")
    print(f"  Embedding Referensi (B): {format_vektor(embedding_ref_terbaik)}")
    print(f"  ‖A‖                    : {norm_a:.5f}")
    print(f"  ‖B‖                    : {norm_b:.5f}")
    print(f"  Cosine Similarity      : {similarity_terbaik:.5f}")
    print(f"  Threshold              : {THRESHOLD}")
    if similarity_terbaik >= THRESHOLD:
        print(f"  Keputusan              : DIKENALI ({nomor_terbaik})")
    else:
        print(f"  Keputusan              : UNKNOWN")
    print("=" * 65)


if __name__ == "__main__":
    main()
