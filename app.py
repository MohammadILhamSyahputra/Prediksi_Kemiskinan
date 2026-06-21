from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
import os

# Mengatur backend Matplotlib agar aman running di Flask/Server tanpa GUI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits import mplot3d

app = Flask(__name__)

# =====================================================================
# LOKASI FILE & INISIALISASI VARIABEL GLOBAL
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH = os.path.join(BASE_DIR, 'Dataset Pengaruh TPT dan RLS terhadap presentase penduduk miskin.xlsx')

IMAGE_DIR = os.path.join(BASE_DIR, 'static', 'images')
os.makedirs(IMAGE_DIR, exist_ok=True)

model_trained = False
metrics = {}
df_sample_json = []
chart_data = []
top10_tinggi_json = []  # Menyimpan data 10 tertinggi untuk dikirim ke web
top10_rendah_json = []  # Menyimpan data 10 terendah untuk dikirim ke web
missing_values_info = {}
regr = None

# =====================================================================
# FUNCTION: GENERATE & SAVE REGRESSION CHARTS (PURE PYTHON STATIC)
# =====================================================================
def generate_and_save_plots(df_clean, col_x1, col_x2, col_y, col_provinsi, konstanta, koef_tpt, koef_rls):
    """
    Membuat dan menyimpan grafik laporan secara langsung di server backend.
    """
    try:
        sns.set_theme(style="whitegrid")
        
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

        # --- 2. GRAFIK STATIS: PENDIDIKAN (X2) VS KEMISKINAN (Y) ---
        plt.figure(figsize=(8, 6))
        sns.regplot(data=df_clean, x=col_x2, y=col_y, color='teal', scatter_kws={'alpha':0.8})
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

        # --- 4. BARU - GRAFIK STATIS: 10 PROVINSI KEMISKINAN TERTINGGI (Horizontal Bar) ---
        plt.figure(figsize=(10, 5))
        df_t = df_clean.sort_values(by=col_y, ascending=False).head(10)
        plt.barh(df_t[col_provinsi][::-1], df_t[col_y][::-1], color='#ef4444')  # Red
        plt.title('Top 10 Provinsi dengan Persentase Kemiskinan Tertinggi', fontsize=12, fontweight='bold', pad=15)
        plt.xlabel('Persentase Penduduk Miskin (%)')
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGE_DIR, 'top10_tertinggi.png'), dpi=300)
        plt.close()

        # --- 5. BARU - GRAFIK STATIS: 10 PROVINSI KEMISKINAN TERENDAH (Horizontal Bar) ---
        plt.figure(figsize=(10, 5))
        df_r = df_clean.sort_values(by=col_y, ascending=True).head(10)
        plt.barh(df_r[col_provinsi][::-1], df_r[col_y][::-1], color='#10b981')  # Emerald Green
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

        possible_y = ['Persentase Penduduk Miskin (Persen)', 'Persentase Penduduk Miskin']
        possible_x1 = ['Tingkat Pengangguran Terbuka (Persen)', 'Tingkat Pengangguran Terbuka']
        possible_x2 = ['Rata-Rata Lama Sekolah Penduduk Umur 15 Tahun ke Atas', 'Rata-Rata Lama Sekolah']
        possible_prov = ['Provinsi', 'Nama Provinsi', 'Wilayah']

        col_y = next((c for c in possible_y if c in df.columns), df.columns[-1])
        col_x1 = next((c for c in possible_x1 if c in df.columns), df.columns[1])
        col_x2 = next((c for c in possible_x2 if c in df.columns), df.columns[2])
        col_provinsi = next((c for c in possible_prov if c in df.columns), df.columns[0])

        missing_values_info = df[[col_y, col_x1, col_x2]].isnull().sum().to_dict()
        
        for col in [col_y, col_x1, col_x2]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df_clean = df.dropna(subset=[col_y, col_x1, col_x2]).copy()
        
        X = df_clean[[col_x1, col_x2]]
        y = df_clean[col_y]
        
        regr = linear_model.LinearRegression()
        regr.fit(X, y)
        
        konstanta = regr.intercept_
        koef_tpt = regr.coef_[0]
        koef_rls = regr.coef_[1]
        
        y_pred = regr.predict(X)
        r2 = r2_score(y, y_pred)
        
        df_sample_json = []
        for _, row in df_clean.head(10).iterrows():
            df_sample_json.append({
                'provinsi': str(row[col_provinsi]),
                'tpt': float(row[col_x1]),
                'rls': float(row[col_x2]),
                'p0': float(row[col_y])
            })
            
        chart_data = []
        for _, row in df_clean.iterrows():
            chart_data.append({
                'x': float(row[col_x1]),
                'y': float(row[col_y])
            })

        # --- BARU: URUTKAN UNTUK EXTRACT TOP 10 TERTINGGI & TERENDAH ---
        df_sorted_desc = df_clean.sort_values(by=col_y, ascending=False)
        df_sorted_asc = df_clean.sort_values(by=col_y, ascending=True)
        
        top10_tinggi_json = []
        for _, row in df_sorted_desc.head(10).iterrows():
            top10_tinggi_json.append({
                'provinsi': str(row[col_provinsi]),
                'p0': float(row[col_y])
            })
            
        top10_rendah_json = []
        for _, row in df_sorted_asc.head(10).iterrows():
            top10_rendah_json.append({
                'provinsi': str(row[col_provinsi]),
                'p0': float(row[col_y])
            })
        
        metrics = {
            'konstanta': float(konstanta),
            'koef_tpt': float(koef_tpt),
            'koef_rls': float(koef_rls),
            'r2': float(r2),
            'total_sampel': len(df_clean),
            'label_y': col_y,
            'label_x1': col_x1,
            'label_x2': col_x2,
            'persamaan': f"Y = {konstanta:.4f} + ({koef_tpt:.4f} * TPT) + ({koef_rls:.4f} * RLS)"
        }
        model_trained = True
        
        # Panggil fungsi generate grafik agar semua aset PNG terbuat dari backend
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
        top10_tinggi=top10_tinggi_json,  # DIKIRIM KE FRONTEND
        top10_rendah=top10_rendah_json   # DIKIRIM KE FRONTEND
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
        prediksi_aman = max(0.0, min(100.0, prediksi_mentah))
        
        return jsonify({
            'status': 'success',
            'prediksi_mentah': prediksi_mentah,
            'prediksi_aman': prediksi_aman,
            'safety_guard_aktif': prediksi_mentah < 0
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)