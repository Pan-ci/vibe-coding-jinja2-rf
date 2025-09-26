import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker  # tambahkan di awal file
from pathlib import Path
import sys

# Tambahkan root project ke sys.path
root = Path(__file__).parent.parent.resolve()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from app.dependencies.mapping import load_and_clean_data, map_jurusan

# BASE_DIR = Path(__file__).resolve().parent
csv_path = root / 'data/Student Mental health.csv'

df = load_and_clean_data(csv_path)
df_jurusan = load_and_clean_data(csv_path, to_replace=map_jurusan)

tahun_map ={
    'year 1': 'Tahun 1',
    'year 2': 'Tahun 2',
    'year 3': 'Tahun 3',
    'year 4': 'Tahun 4'
}

# 2. Binerisasi target (Depression)
df['Depression'] = df['Do you have Depression?'].map({'Yes': "Ya", 'No': "Tidak"})
df['Depresi'] = df['Do you have Depression?'].map({'Yes': "Ya", 'No': "Tidak"})
df['Do you have Depression?'] = df['Do you have Depression?'].map({'Yes': 1, 'No': 0})
df['Choose your gender'] = df['Choose your gender'].map({'Female': "Perempuan", 'Male': "Laki-laki"})
df['Jenis Kelamin'] = df['Choose your gender']
df['Tahun Perkuliahan'] = df['Your current year of Study'].str.strip().str.lower().map(tahun_map)
df['IPK'] = df['What is your CGPA?'].str.strip().str.lower()
df['Status Pernikahan'] = df['Marital status'].map({'Yes': 'Menikah', 'No': 'Lajang'})
df['Jurusan'] = df_jurusan['What is your course?']
df['Usia'] = df['Age']

# ============================
# 1. Pie chart distribusi target
# ============================
plt.figure(figsize=(5, 5))
counts = df['Depression'].value_counts()

# Pie chart dengan label kategori Ya dan Tidak
counts.plot.pie(
    labels=counts.index,  # Menampilkan "Yes" dan "No" # autopct='%1.1f%%',
    autopct=lambda p: '{:.0f}%'.format(p),
    colors=['#66b3ff', '#ff9999'],
    startangle=90  # (opsional) rotasi agar lebih simetris
)
plt.title('Distribusi Mahasiswa dengan/ tanpa Depresi')
plt.ylabel('')  # Hilangkan label sumbu y

png_path = root / 'outputs/figures/Distribusi depresi mahasiswa.png'
plt.savefig(png_path, dpi=300)
plt.show()

# ============================
# 2. Bar chart per fitur kategorikal
# ============================
def barplot_by_depression(column_name, title, order=None, tahun=False):
    plt.figure(figsize=(8, 4))
    ax = sns.countplot(data=df, x=column_name, hue='Depresi', palette='Set2', order=order)

    # Tambahkan label jumlah di atas bar (tanpa koma)
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', label_type='edge', padding=3)
    
    # Format sumbu Y agar tampil sebagai integer tanpa koma
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))

    plt.title(title)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.ylabel('Jumlah')  # Hilangkan label sumbu y

    # Tambahkan setelah sns.countplot() dan sebelum plt.show()
    max_height = max([bar.get_height() for bar in ax.patches])
    ax.set_ylim(0, max_height * 1.15)  # 15% lebih tinggi dari bar tertinggi

    if tahun:
        ticks = ax.get_xticks()
        labels = [f'{label.get_text()} tahun' for label in ax.get_xticklabels()]
        ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(labels))

    png_path = root / f'outputs/figures/{title}.png'
    plt.savefig(png_path, dpi=300)
    plt.show()

barplot_by_depression('Jenis Kelamin', 'Jenis Kelamin vs Depresi')
barplot_by_depression('Tahun Perkuliahan', 'Tahun Perkuliahan vs Depresi')

cgpa_order = ['0 - 1.99', '2.00 - 2.49', '2.50 - 2.99', '3.00 - 3.49', '3.50 - 4.00']
barplot_by_depression('IPK', 'Indeks Prestasi Kumulatif (IPK) vs Depresi', order=cgpa_order)

marital_order = ['Menikah', 'Lajang']

barplot_by_depression('Status Pernikahan', 'Status Pernikahan vs Depresi', order=marital_order)

barplot_by_depression('Usia', 'Usia vs Depresi', tahun=True)

# Threshold minimal frekuensi jurusan untuk dipertahankan
threshold = 3

# Hitung frekuensi tiap jurusan
course_counts = df['Jurusan'].value_counts()

# Jurusan dengan jumlah di bawah threshold → akan jadi 'Other'
minor_courses = course_counts[course_counts < threshold].index

# Ganti jurusan minoritas dengan 'Other'
df['Jurusan'] = df['Jurusan'].replace(minor_courses, 'lain-lain')

course_depression = df[df['Depresi'] == 'Ya']['Jurusan'].value_counts()
course_total = df['Jurusan'].value_counts()

course_df = pd.DataFrame({
    'Total': course_total,
    'Depressed': course_depression
}).fillna(0).sort_values(by='Depressed', ascending=False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Depressed', y=course_df.index, data=course_df, palette='Reds_r', hue=course_df.index, legend=False)

# Label jumlah di ujung bar
for container in ax.containers:
    ax.bar_label(container, fmt='%d', padding=3, label_type='edge')

ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))

max_width = max([bar.get_width() for bar in ax.patches])
ax.set_xlim(0, max_width * 1.15)

plt.xlabel('Jumlah Mahasiswa dengan Depresi')
plt.title('Jumlah Mahasiswa dengan Depresi per Jurusan (Gabungan Minor → Lain-lain)')
plt.tight_layout()
png_path = root / f'outputs/figures/Jumlah Mahasiswa dengan Depresi.png'
plt.savefig(png_path, dpi=300)
plt.show()

course_ratio = (course_depression / course_total).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x=course_ratio.values, y=course_ratio.index, palette='coolwarm', hue=course_ratio.index, legend=False)

# Tambahkan label persentase di ujung bar
for bar in ax.patches:
    width = bar.get_width()
    ax.text(width + 0.01,                   # Sedikit di kanan bar
            bar.get_y() + bar.get_height()/2, 
            f'{width:.0%}',                # Format sebagai persen bulat
            va='center')

# plt.subplots_adjust(right=0.85)  # Memberi ruang di sisi kanan
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.0%}'))
max_width = max([bar.get_width() for bar in ax.patches])
ax.set_xlim(0, max_width * 1.075)
plt.xlabel('Rasio Mahasiswa Depresi')
plt.title('Rasio Depresi per Jurusan (Gabungan Minor → Lain-lain)')
plt.tight_layout()
png_path = root / f'outputs/figures/Rasio Mahasiswa Depresi.png'
plt.savefig(png_path, dpi=300)
plt.show()

'''
# tidak akan dieksekusi jika hanya diimpor, harus dijalankan langsung
if __name__ == '__barplot_by_depression__':
    barplot_by_depression()
'''