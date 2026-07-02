# ## Memakai API

# from flask import Flask, render_template, request, jsonify, Response
# from db import get_kehadiran, simpan_mahasiswa, simpan_embedding, simpan_presensi, load_semua_embedding
# import json
# import time
# import numpy as np
# import pandas as pd
# from io import BytesIO
# from flask import send_file

# app = Flask(__name__)


# # ════════════════════════════════
# #  HALAMAN
# # ════════════════════════════════

# @app.route('/')
# @app.route('/kehadiran')
# def kehadiran():
#     return render_template('data_kehadiran.html')


# # ════════════════════════════════
# #  API DATA WEB
# # ════════════════════════════════

# @app.route('/kehadiran/data')
# def kehadiran_data():
#     keyword = request.args.get('keyword', '').strip()
#     tanggal = request.args.get('tanggal', '').strip()
#     try:
#         data = get_kehadiran(keyword, tanggal)
#         return jsonify(data)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# # ════════════════════════════════
# #  SSE STREAM (REALTIME)
# # ═══════════════════════════════
# @app.route('/kehadiran/stream')
# def kehadiran_stream():
#     keyword = request.args.get('keyword', '').strip()
#     tanggal_dari = request.args.get('tanggal_dari', '').strip()
#     tanggal_sampai = request.args.get('tanggal_sampai', '').strip()

#     def event_stream():
#         last_data = None
#         while True:
#             data = get_kehadiran(keyword, tanggal_dari, tanggal_sampai)
#             if data != last_data:
#                 yield f"data: {json.dumps(data)}\n\n"
#                 last_data = data
#             time.sleep(2)

#     return Response(event_stream(), mimetype="text/event-stream")


# @app.route('/kehadiran/export')
# def kehadiran_export():
#     keyword = request.args.get('keyword', '').strip()
#     tanggal_dari = request.args.get('tanggal_dari', '').strip()
#     tanggal_sampai = request.args.get('tanggal_sampai', '').strip()

#     # Proteksi backend dari tanggal terbalik
#     if tanggal_dari and tanggal_sampai and tanggal_dari > tanggal_sampai:
#         return "Rentang tanggal tidak valid", 400

#     data = get_kehadiran(keyword, tanggal_dari, tanggal_sampai)

#     # Antisipasi data kosong (tetap print header excel)
#     if not data:
#         df = pd.DataFrame(columns=["ID", "NIM", "Nama Mahasiswa", "Waktu Presensi", "Status"])
#     else:
#         df = pd.DataFrame(data)
#         # Merapikan nama kolom untuk Excel
#         df.rename(columns={
#             "nim": "NIM",
#             "nama": "Nama Mahasiswa",
#             "waktu_presensi": "Waktu Presensi",
#             "status": "Status"
#         }, inplace=True)

#     output = BytesIO()
#     with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
#         df.to_excel(writer, index=False, sheet_name='Data Presensi')
        
#         # Opsional: melebarkan kolom otomatis di Excel
#         worksheet = writer.sheets['Data Presensi']
#         worksheet.set_column('B:B', 15) # Lebar kolom NIM
#         worksheet.set_column('C:C', 30) # Lebar kolom Nama
#         worksheet.set_column('D:D', 20) # Lebar kolom Waktu

#     output.seek(0)
    
#     # Penamaan file dinamis
#     if tanggal_dari or tanggal_sampai:
#         filename = f"Presensi_{tanggal_dari}_sd_{tanggal_sampai}.xlsx"
#     else:
#         filename = "Presensi_Semua_Data.xlsx"

#     return send_file(output, download_name=filename, as_attachment=True, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# # ════════════════════════════════
# #  API ENDPOINT UNTUK RPi (EDGE)
# # ════════════════════════════════

# @app.route('/api/mahasiswa', methods=['POST'])
# def api_simpan_mahasiswa():
#     """Endpoint untuk RPi mengirim data mahasiswa"""
#     try:
#         data = request.get_json()
#         nim  = data.get('nim')
#         nama = data.get('nama')

#         if not nim or not nama:
#             return jsonify({"error": "NIM dan nama wajib diisi"}), 400

#         simpan_mahasiswa(nim, nama)
#         return jsonify({"message": f"Mahasiswa {nama} ({nim}) berhasil disimpan"}), 200

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# @app.route('/api/embedding', methods=['POST'])
# def api_simpan_embedding():
#     """Endpoint untuk RPi mengirim data embedding wajah"""
#     try:
#         data         = request.get_json()
#         nim          = data.get('nim')
#         vector_hex   = data.get('vector')

#         if not nim or not vector_hex:
#             return jsonify({"error": "NIM dan vector wajib diisi"}), 400

#         vector = np.frombuffer(bytes.fromhex(vector_hex), dtype=np.float32)
#         simpan_embedding(nim, vector)
#         return jsonify({"message": f"Embedding NIM {nim} berhasil disimpan"}), 200

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# @app.route('/api/presensi', methods=['POST'])
# def api_simpan_presensi():
#     """Endpoint untuk RPi mengirim data presensi"""
#     try:
#         data = request.get_json()
#         nim  = data.get('nim')

#         if not nim:
#             return jsonify({"error": "NIM wajib diisi"}), 400

#         simpan_presensi(nim)
#         return jsonify({"message": f"Presensi NIM {nim} berhasil dicatat"}), 200

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# @app.route('/api/embeddings', methods=['GET'])
# def api_get_embeddings():
#     """Endpoint untuk RPi mengambil semua embedding dari server"""
#     try:
#         embeddings = load_semua_embedding()
#         result     = {}

#         for nim, vector in embeddings.items():
#             result[nim] = vector.tobytes().hex()

#         return jsonify(result), 200

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
    
# @app.route('/api/mahasiswa/<nim>', methods=['GET'])
# def api_get_mahasiswa(nim):
#     """Endpoint untuk cek NIM dan ambil nama mahasiswa"""
#     try:
#         from db import get_nama, nim_sudah_terdaftar
#         terdaftar = nim_sudah_terdaftar(nim)
#         if terdaftar:
#             nama = get_nama(nim)
#             return jsonify({"terdaftar": True, "nama": nama}), 200
#         return jsonify({"terdaftar": False}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# @app.route('/api/presensi/hari-ini', methods=['GET'])
# def api_get_presensi_hari_ini():
#     """Endpoint untuk RPi mengambil NIM yang sudah presensi hari ini"""
#     try:
#         from db import get_nim_presensi_hari_ini
#         nims = get_nim_presensi_hari_ini()
#         return jsonify({"nims": list(nims)}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# # ════════════════════════════════
# #  MAIN
# # ════════════════════════════════

# if __name__ == '__main__':
#     app.run(debug=True, threaded=True, use_reloader=False)




# -----------------------------------
# fix


## Memakai API

from flask import Flask, render_template, request, jsonify
from db import get_kehadiran, simpan_pengguna, simpan_embedding, simpan_presensi, load_semua_embedding
import numpy as np
import pandas as pd
from io import BytesIO
from flask import send_file

app = Flask(__name__)


# ════════════════════════════════
#  HALAMAN
# ════════════════════════════════

@app.route('/')
@app.route('/kehadiran')
def kehadiran():
    return render_template('data_kehadiran.html')


# ════════════════════════════════
#  API DATA WEB
# ════════════════════════════════

@app.route('/kehadiran/data')
def kehadiran_data():
    keyword = request.args.get('keyword', '').strip()
    tanggal_dari = request.args.get('tanggal_dari', '').strip()
    tanggal_sampai = request.args.get('tanggal_sampai', '').strip()
    try:
        data = get_kehadiran(keyword, tanggal_dari, tanggal_sampai)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/kehadiran/export')
def kehadiran_export():
    keyword = request.args.get('keyword', '').strip()
    tanggal_dari = request.args.get('tanggal_dari', '').strip()
    tanggal_sampai = request.args.get('tanggal_sampai', '').strip()

    # Proteksi backend dari tanggal terbalik
    if tanggal_dari and tanggal_sampai and tanggal_dari > tanggal_sampai:
        return "Rentang tanggal tidak valid", 400

    data = get_kehadiran(keyword, tanggal_dari, tanggal_sampai)

    # Antisipasi data kosong (tetap print header excel)
    if not data:
        df = pd.DataFrame(columns=["Nomor Identitas", "Nama Pengguna", "Waktu Presensi", "Status"])
    else:
        df = pd.DataFrame(data)
        # Merapikan nama kolom untuk Excel
        df.rename(columns={
            "nomor_identitas": "Nomor Identitas",
            "nama": "Nama Pengguna",
            "waktu_presensi": "Waktu Presensi",
            "status": "Status"
        }, inplace=True)

    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Data Presensi')

        # Opsional: melebarkan kolom otomatis di Excel
        worksheet = writer.sheets['Data Presensi']
        worksheet.set_column('B:B', 18)  # Lebar kolom Nomor Identitas
        worksheet.set_column('C:C', 30)  # Lebar kolom Nama
        worksheet.set_column('D:D', 20)  # Lebar kolom Waktu

    output.seek(0)

    # Penamaan file dinamis
    if tanggal_dari or tanggal_sampai:
        filename = f"Presensi_{tanggal_dari}_sd_{tanggal_sampai}.xlsx"
    else:
        filename = "Presensi_Semua_Data.xlsx"

    return send_file(output, download_name=filename, as_attachment=True, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ════════════════════════════════
#  API ENDPOINT UNTUK RPi (EDGE)
# ════════════════════════════════

@app.route('/api/pengguna', methods=['POST'])
def api_simpan_pengguna():
    """Endpoint untuk RPi mengirim data pengguna"""
    try:
        data = request.get_json()
        nomor_identitas = data.get('nomor_identitas')
        nama = data.get('nama')

        if not nomor_identitas or not nama:
            return jsonify({"error": "Nomor identitas dan nama wajib diisi"}), 400

        simpan_pengguna(nomor_identitas, nama)
        return jsonify({"message": f"Pengguna {nama} ({nomor_identitas}) berhasil disimpan"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/embedding', methods=['POST'])
def api_simpan_embedding():
    """Endpoint untuk RPi mengirim data embedding wajah"""
    try:
        data = request.get_json()
        nomor_identitas = data.get('nomor_identitas')
        vector_hex = data.get('vector')

        if not nomor_identitas or not vector_hex:
            return jsonify({"error": "Nomor identitas dan vector wajib diisi"}), 400

        vector = np.frombuffer(bytes.fromhex(vector_hex), dtype=np.float32)
        simpan_embedding(nomor_identitas, vector)
        return jsonify({"message": f"Embedding nomor identitas {nomor_identitas} berhasil disimpan"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/presensi', methods=['POST'])
def api_simpan_presensi():
    """Endpoint untuk RPi mengirim data presensi"""
    try:
        data = request.get_json()
        nomor_identitas = data.get('nomor_identitas')

        if not nomor_identitas:
            return jsonify({"error": "Nomor identitas wajib diisi"}), 400

        simpan_presensi(nomor_identitas)
        return jsonify({"message": f"Presensi nomor identitas {nomor_identitas} berhasil dicatat"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/embeddings', methods=['GET'])
def api_get_embeddings():
    """Endpoint untuk RPi mengambil semua embedding dari server"""
    try:
        embeddings = load_semua_embedding()
        result = {}

        for nomor_identitas, vector in embeddings.items():
            result[nomor_identitas] = vector.tobytes().hex()

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/pengguna/<nomor_identitas>', methods=['GET'])
def api_get_pengguna(nomor_identitas):
    """Endpoint untuk cek nomor identitas dan ambil nama pengguna"""
    try:
        from db import get_nama, nomor_identitas_sudah_terdaftar
        terdaftar = nomor_identitas_sudah_terdaftar(nomor_identitas)
        if terdaftar:
            nama = get_nama(nomor_identitas)
            return jsonify({"terdaftar": True, "nama": nama}), 200
        return jsonify({"terdaftar": False}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/presensi/hari-ini', methods=['GET'])
def api_get_presensi_hari_ini():
    """Endpoint untuk RPi mengambil nomor identitas yang sudah presensi hari ini"""
    try:
        from db import get_nomor_identitas_presensi_hari_ini
        daftar_nomor_identitas = get_nomor_identitas_presensi_hari_ini()
        return jsonify({"daftar_nomor_identitas": list(daftar_nomor_identitas)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════
#  MAIN
# ════════════════════════════════

if __name__ == '__main__':
    app.run(debug=True, threaded=True, use_reloader=False)