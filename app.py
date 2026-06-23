from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os

# Mengatur backend Matplotlib agar aman running di Flask/Server tanpa GUI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

app = Flask(__name__)

# =====================================================================
# LOKASI FILE & INISIALISASI VARIABEL GLOBAL
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH = os.path.join(BASE_DIR, 'Dataset Pengaruh TPT dan RLS terhadap presentase penduduk miskin2.xlsx')

IMAGE_DIR = os.path.join(BASE_DIR, 'static', 'images')
os.makedirs(IMAGE_DIR, exist_ok=True)

model_trained = False
metrics = {}
df_sample_json = []
chart_data = []
top10_tinggi_json = []
top10_rendah_json = []
missing_values_info = {}
regr = None

# =====================================================================
# FUNCTION: GENERATE & SAVE REGRESSION CHARTS
# =====================================================================
def generate_and_save_plots(df_clean, col_x1, col_x2, col_y, col_provinsi, konstanta, koef_tpt, koef_rls):
    """
    Membuat dan menyimpan grafik laporan secara langsung di server backend.
    """
    try:
        plt.rcParams['axes.facecolor'] = '#ffffff'
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.color'] = '#f1f5f9'
        plt.rcParams['axes.edgecolor'] = '#cbd5e1'

        # --- 1. GRAFIK STATIS: SEBARAN DATA ---
        plt.figure(figsize=(8, 5))
        plt.gca().set_facecolor('#ffffff')
        plt.grid(True, color='#f1f5f9', linestyle='-', linewidth=1.5)
        plt.scatter(df_clean[col_x1], df_clean[col_y],
                    color='#4f46e5', edgecolor='#4338ca',
                    linewidth=1.5, s=80, alpha=0.6, label='Titik Koordinat Provinsi')
        plt.title('Hubungan Tingkat Pengangguran Terbuka (X1) vs Kemiskinan (Y)', fontsize=12, fontweight='bold', color='#1e293b', pad=15)
        plt.xlabel('Tingkat Pengangguran Terbuka (%)')
        plt.ylabel('Persentase Penduduk Miskin (%)')
        for spine in plt.gca().spines.values():
            spine.set_color('#cbd5e1')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGE_DIR, 'sebaran_data.png'), dpi=300)
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.scatter(df_clean[col_x1], df_clean[col_y], color='#4f46e5', alpha=0.8, label='Data Provinsi')

        # Menghitung garis regresi linear sederhana
        coef_garis_x1 = np.polyfit(df_clean[col_x1], df_clean[col_y], 1)
        x1_line = np.linspace(df_clean[col_x1].min(), df_clean[col_x1].max(), 100)
        y1_line = np.polyval(coef_garis_x1, x1_line)

        # Plot garis regresi
        plt.plot(x1_line, y1_line, color='#4338ca', linewidth=2, label='Garis Regresi')

        plt.title('Hubungan Tingkat Pengangguran (X1) vs Kemiskinan (Y)', fontsize=12, fontweight='bold')
        plt.xlabel('Tingkat Pengangguran Terbuka (%)')
        plt.ylabel('Persentase Penduduk Miskin (%)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGE_DIR, 'scatter_pengangguran.png'), dpi=300)
        plt.close()

        # --- 2. GRAFIK STATIS: PENDIDIKAN (X2) VS KEMISKINAN (Y) ---
        # Pengganti sns.regplot: scatter + garis regresi manual pakai numpy
        plt.figure(figsize=(8, 6))
        plt.scatter(df_clean[col_x2], df_clean[col_y], color='teal', alpha=0.8)
        coef_garis = np.polyfit(df_clean[col_x2], df_clean[col_y], 1)
        x_line = np.linspace(df_clean[col_x2].min(), df_clean[col_x2].max(), 100)
        y_line = np.polyval(coef_garis, x_line)
        plt.plot(x_line, y_line, color='teal', linewidth=2)
        plt.title('Hubungan Rata-Rata Lama Sekolah (X2) vs Kemiskinan (Y)', fontsize=12, fontweight='bold')
        plt.xlabel('Rata-Rata Lama Sekolah (Tahun)')
        plt.ylabel('Persentase Penduduk Miskin (%)')
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGE_DIR, 'scatter_pendidikan.png'), dpi=300)
        plt.close()

        # --- 3. GRAFIK STATIS: BIDANG REGRESI 3D ---
        fig = plt.figure(figsize=(10, 8))
        ax_3d = plt.axes(projection='3d')
        x1_range = np.linspace(df_clean[col_x1].min(), df_clean[col_x1].max(), 20)
        x2_range = np.linspace(df_clean[col_x2].min(), df_clean[col_x2].max(), 20)
        X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)
        Y_grid = konstanta + koef_tpt * X1_grid + koef_rls * X2_grid
        ax_3d.plot_surface(X1_grid, X2_grid, Y_grid, cmap='viridis', alpha=0.4, edgecolor='none')
        ax_3d.scatter3D(df_clean[col_x1], df_clean[col_x2], df_clean[col_y], color='red', s=50, depthshade=True, label='Data Riil Provinsi')
        ax_3d.set_title('Visualisasi Bidang Regresi 3D (UAS Statistika)', fontsize=14, fontweight='bold', pad=20)
        ax_3d.set_xlabel('Tingkat Pengangguran (X1) %')
        ax_3d.set_ylabel('Rata-Rata Lama Sekolah (X2) Tahun')
        ax_3d.set_zlabel('Kemiskinan (Y) %')
        ax_3d.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGE_DIR, 'regresi_3d.png'), dpi=300)
        plt.close()

        # --- 4. GRAFIK STATIS: 10 PROVINSI KEMISKINAN TERTINGGI ---
        plt.figure(figsize=(10, 5))
        df_t = df_clean.sort_values(by=col_y, ascending=False).head(10)
        plt.barh(df_t[col_provinsi][::-1], df_t[col_y][::-1], color='#ef4444')
        plt.title('Top 10 Provinsi dengan Persentase Kemiskinan Tertinggi', fontsize=12, fontweight='bold', pad=15)
        plt.xlabel('Persentase Penduduk Miskin (%)')
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGE_DIR, 'top10_tertinggi.png'), dpi=300)
        plt.close()

        # --- 5. GRAFIK STATIS: 10 PROVINSI KEMISKINAN TERENDAH ---
        plt.figure(figsize=(10, 5))
        df_r = df_clean.sort_values(by=col_y, ascending=True).head(10)
        plt.barh(df_r[col_provinsi][::-1], df_r[col_y][::-1], color='#10b981')
        plt.title('Top 10 Provinsi dengan Persentase Kemiskinan Terendah', fontsize=12, fontweight='bold', pad=15)
        plt.xlabel('Persentase Penduduk Miskin (%)')
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGE_DIR, 'top10_terendah.png'), dpi=300)
        plt.close()

        print("[INFO] Semua grafik statis (.png) sukses diproduksi di folder static/images/!")
    except Exception as e:
        print(f"[PERINGATAN] Gagal memproduksi gambar grafik: {e}")

# =====================================================================
# PREPROCESSING & TRAINING MODEL (JALAN SAAT START)
# =====================================================================
if os.path.exists(EXCEL_PATH):
    try:
        xl = pd.ExcelFile(EXCEL_PATH)
        df = None
        for sheet in xl.sheet_names:
            temp_df = pd.read_excel(EXCEL_PATH, sheet_name=sheet)
            temp_df.columns = temp_df.columns.str.strip()
            if any('Kemiskinan' in col or 'Miskin' in col for col in temp_df.columns):
                df = temp_df
                break
        if df is None:
            df = pd.read_excel(EXCEL_PATH)
            df.columns = df.columns.str.strip()

        possible_y    = ['Persentase Penduduk Miskin (Persen)', 'Persentase Penduduk Miskin']
        possible_x1   = ['Tingkat Pengangguran Terbuka (Persen)', 'Tingkat Pengangguran Terbuka']
        possible_x2   = ['Rata-Rata Lama Sekolah Penduduk Umur 15 Tahun ke Atas', 'Rata-Rata Lama Sekolah']
        possible_prov = ['Provinsi', 'Nama Provinsi', 'Wilayah']

        col_y        = next((c for c in possible_y    if c in df.columns), df.columns[-1])
        col_x1       = next((c for c in possible_x1   if c in df.columns), df.columns[1])
        col_x2       = next((c for c in possible_x2   if c in df.columns), df.columns[2])
        col_provinsi = next((c for c in possible_prov if c in df.columns), df.columns[0])

        missing_values_info = df[[col_y, col_x1, col_x2]].isnull().sum().to_dict()

        for col in [col_y, col_x1, col_x2]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df_clean = df.dropna(subset=[col_y, col_x1, col_x2]).copy()

        # --- REGRESI LINEAR BERGANDA (OLS) PAKAI NUMPY — TANPA SCIKIT-LEARN ---
        X_arr = np.column_stack([
            np.ones(len(df_clean)),
            df_clean[col_x1].values,
            df_clean[col_x2].values
        ])
        y_arr = df_clean[col_y].values

        coeffs, _, _, _ = np.linalg.lstsq(X_arr, y_arr, rcond=None)
        konstanta, koef_tpt, koef_rls = coeffs

        y_pred  = X_arr @ coeffs
        ss_res  = np.sum((y_arr - y_pred) ** 2)
        ss_tot  = np.sum((y_arr - np.mean(y_arr)) ** 2)
        r2      = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        regr = True  # penanda model siap

        df_sample_json = []
        for _, row in df_clean.head(10).iterrows():
            df_sample_json.append({
                'provinsi': str(row[col_provinsi]),
                'tpt':      float(row[col_x1]),
                'rls':      float(row[col_x2]),
                'p0':       float(row[col_y])
            })

        chart_data = []
        for _, row in df_clean.iterrows():
            chart_data.append({
                'x': float(row[col_x1]),
                'y': float(row[col_y])
            })

        df_sorted_desc = df_clean.sort_values(by=col_y, ascending=False)
        df_sorted_asc  = df_clean.sort_values(by=col_y, ascending=True)

        top10_tinggi_json = []
        for _, row in df_sorted_desc.head(10).iterrows():
            top10_tinggi_json.append({
                'provinsi': str(row[col_provinsi]),
                'p0':       float(row[col_y])
            })

        top10_rendah_json = []
        for _, row in df_sorted_asc.head(10).iterrows():
            top10_rendah_json.append({
                'provinsi': str(row[col_provinsi]),
                'p0':       float(row[col_y])
            })

        metrics = {
            'konstanta':    float(konstanta),
            'koef_tpt':     float(koef_tpt),
            'koef_rls':     float(koef_rls),
            'r2':           float(r2),
            'total_sampel': len(df_clean),
            'label_y':      col_y,
            'label_x1':     col_x1,
            'label_x2':     col_x2,
            'persamaan':    f"Y = {konstanta:.4f} + ({koef_tpt:.4f} * TPT) + ({koef_rls:.4f} * RLS)"
        }
        model_trained = True

        generate_and_save_plots(df_clean, col_x1, col_x2, col_y, col_provinsi, konstanta, koef_tpt, koef_rls)

    except Exception as e:
        print(f"[EROR] Gagal memproses data: {e}")

# =====================================================================
# ROUTE CONTROLLER
# =====================================================================
@app.route('/')
def index():
    return render_template(
        'dashboard.html',
        model_trained=model_trained,
        metrics=metrics,
        df_sample=df_sample_json,
        chart_data=chart_data,
        missing_values=missing_values_info,
        top10_tinggi=top10_tinggi_json,
        top10_rendah=top10_rendah_json
    )

@app.route('/predict', methods=['POST'])
def predict():
    if not model_trained or regr is None:
        return jsonify({'status': 'error', 'message': 'Model belum siap.'}), 500
    try:
        data = request.get_json()
        tpt_val = float(data.get('tpt', 0))
        rls_val = float(data.get('rls', 0))

        prediksi_mentah = float(metrics['konstanta'] + (metrics['koef_tpt'] * tpt_val) + (metrics['koef_rls'] * rls_val))
        prediksi_aman   = max(0.0, min(100.0, prediksi_mentah))

        return jsonify({
            'status':              'success',
            'prediksi_mentah':     prediksi_mentah,
            'prediksi_aman':       prediksi_aman,
            'safety_guard_aktif':  prediksi_mentah < 0
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)