import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats

# =====================================================================
# 1. MEMBACA & PREPROSES DATA
# =====================================================================
# Membaca dataset padi Sumatera
df = pd.read_csv('Data_Tanaman_Padi_Sumatera_version_1.csv')

# Membersihkan nama kolom dari spasi berlebih
df.columns = df.columns.str.strip()

# Menentukan variabel independen (X) dan dependen (y)
X_vars = ['Luas Panen', 'Curah hujan']
y_var = 'Produksi'

X = df[X_vars]
y = df[y_var]

print("=" * 80)
print("             ANALISIS REGRESI LINEAR BERGANDA UNTUK UAS ILMU DATA")
print("=" * 80)
print(f"Dataset      : Data Tanaman Padi Sumatera")
print(f"Jumlah Sampel: {len(df)} data")
print(f"Variabel X1  : Luas Panen (Hektar)")
print(f"Variabel X2  : Curah Hujan (mm)")
print(f"Variabel Y   : Produksi Padi (Ton)")
print("-" * 80)

# =====================================================================
# 2. ANALISIS KORELASI AWAL
# =====================================================================
matrix_korelasi = df[X_vars + [y_var]].corr(method='pearson')
print("\n[1] MATRIKS KORELASI PEARSON:")
print(matrix_korelasi)
print("-" * 80)

# =====================================================================
# 3. PEMODELAN REGRESI MENGGUNAKAN STATSMODELS OLS
# =====================================================================
# Menambahkan konstanta (intercept) pada variabel X untuk statsmodels
X_with_const = sm.add_constant(X)

# Membuat dan melatih model OLS (Ordinary Least Squares)
model = sm.OLS(y, X_with_const).fit()

# Menyimpan parameter model
konstanta = model.params['const']
koef_luas = model.params['Luas Panen']
koef_curah = model.params['Curah hujan']
residuals = model.resid
fitted_values = model.fittedvalues

print("\n[2] RINGKASAN MODEL OLS (ORDINARY LEAST SQUARES):")
print(model.summary())
print("-" * 80)

# =====================================================================
# 4. UJI ASUMSI KLASIK
# =====================================================================
print("\n[3] UJI ASUMSI KLASIK MODEL REGRESI:")

# A. Uji Multikolinearitas menggunakan VIF (Variance Inflation Factor)
# VIF < 10 berarti tidak terjadi gejala multikolinearitas serius.
vif_data = pd.DataFrame()
vif_data["Variabel"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print("\nA. Uji Multikolinearitas (VIF):")
print(vif_data)
has_multikol = any(vif_data["VIF"] > 10)
print(f"Kesimpulan VIF: {'Terjadi' if has_multikol else 'Aman dari'} Gejala Multikolinearitas (Batas Toleransi VIF < 10)")

# B. Uji Normalitas Residual (Shapiro-Wilk & Jarque-Bera)
# p-value > 0.05 berarti residual berdistribusi normal.
stat_jb, p_jb, _, _ = sm.stats.jarque_bera(residuals)
print(f"\nB. Uji Normalitas Residual (Jarque-Bera Test):")
print(f"   - JB Statistic: {stat_jb:.4f}")
print(f"   - p-value     : {p_jb:.4f}")
print(f"   Kesimpulan    : Residual {'BERDISTRIBUSI NORMAL' if p_jb > 0.05 else 'TIDAK BERDISTRIBUSI NORMAL'} (alpha=5%)")

# C. Uji Heteroskedastisitas (Breusch-Pagan Test)
# p-value > 0.05 berarti varians residual homoskedastis (konstan/tidak ada heteroskedastisitas).
bp_test = het_breuschpagan(residuals, X_with_const)
p_bp = bp_test[1]
print(f"\nC. Uji Heteroskedastisitas (Breusch-Pagan Test):")
print(f"   - LM Statistic: {bp_test[0]:.4f}")
print(f"   - p-value     : {p_bp:.4f}")
print(f"   Kesimpulan    : Model {'Bebas dari' if p_bp > 0.05 else 'Mengalami Gejala'} Heteroskedastisitas (alpha=5%)")

# D. Uji Autokorelasi (Durbin-Watson Test)
# Nilai DW di sekitar 1.5 - 2.5 mengindikasikan tidak adanya autokorelasi serius.
dw_stat = sm.stats.stattools.durbin_watson(residuals)
print(f"\nD. Uji Autokorelasi (Durbin-Watson):")
print(f"   - DW Statistic: {dw_stat:.4f}")
if 1.5 <= dw_stat <= 2.5:
    dw_conclusion = "Tidak ada indikasi autokorelasi yang kuat (Aman)"
else:
    dw_conclusion = "Terindikasi adanya autokorelasi pada residual"
print(f"   Kesimpulan    : {dw_conclusion}")
print("-" * 80)

# =====================================================================
# 5. FORMULASI & INTERPRETASI AKHIR UNTUK UAS
# =====================================================================
print("\n[4] PERSAMAAN REGRESI AKHIR:")
print(f"Y = {konstanta:.4f} + ({koef_luas:.4f} * Luas_Panen) + ({koef_curah:.4f} * Curah_Hujan)")
print("\nInterpretasi Koefisien:")
print(f"1. Konstanta ({konstanta:.4f}): Jika Luas Panen dan Curah Hujan bernilai 0, estimasi produksi padi bernilai {konstanta:.2f} Ton.")
print(f"   (Nilai negatif aman secara matematis karena data latih berada jauh dari titik nol).")
print(f"2. Koefisien Luas Panen ({koef_luas:.4f}): Hubungan POSITIF (+).")
print(f"   Setiap kenaikan Luas Panen sebesar 1 Hektar akan meningkatkan Produksi Padi sebesar {koef_luas:.4f} Ton, dengan asumsi Curah Hujan konstan.")
print(f"3. Koefisien Curah Hujan ({koef_curah:.4f}): Hubungan POSITIF (+).")
print(f"   Setiap kenaikan Curah Hujan sebesar 1 mm akan meningkatkan Produksi Padi sebesar {koef_curah:.4f} Ton, dengan asumsi Luas Panen konstan.")
print("=" * 80)

# =====================================================================
# 6. VISUALISASI GRAFIK DIAGNOSTIK UAS (MULTI-PLOT)
# =====================================================================
# Mengatur tema estetik plot
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('INFOGRAFIS DIAGNOSTIK MODEL REGRESI - UAS ILMU DATA', fontsize=16, fontweight='bold')

# Plot 1: Heatmap Korelasi Pearson
sns.heatmap(matrix_korelasi, annot=True, cmap="YlGnBu", fmt=".4f", linewidths=.5, ax=axes[0,0])
axes[0,0].set_title("A. Matriks Korelasi Pearson (Kekuatan Hubungan)", fontsize=12, fontweight='bold')

# Plot 2: Residual vs Fitted Values (Uji Gejala Heteroskedastisitas)
axes[0,1].scatter(fitted_values, residuals, color='coral', alpha=0.7, edgecolors='k')
axes[0,1].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
axes[0,1].set_xlabel('Fitted Values (Prediksi Produksi)')
axes[0,1].set_ylabel('Residuals (Error)')
axes[0,1].set_title('B. Residuals vs Fitted Plot (Deteksi Heteroskedastisitas)', fontsize=12, fontweight='bold')

# Plot 3: Distribusi Residual dengan Kurva KDE (Uji Normalitas)
sns.histplot(residuals, kde=True, color='teal', ax=axes[1,0], bins=20)
axes[1,0].set_title('C. Distribusi Frekuensi Residual (Deteksi Normalitas)', fontsize=12, fontweight='bold')
axes[1,0].set_xlabel('Residual (Error)')

# Plot 4: Q-Q Plot Residual
sm.qqplot(residuals, line='45', fit=True, ax=axes[1,1])
axes[1,1].get_lines()[1].set_color('red') # Mengubah warna garis referensi 45 derajat
axes[1,1].set_title('D. Normal Q-Q Plot (Validasi Sebaran Residual)', fontsize=12, fontweight='bold')

plt.tight_layout()
print("\n[INFO] Menampilkan visualisasi diagnostik 4-panel untuk laporan UAS...")
plt.show()