from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import PowerTransformer, StandardScaler, OneHotEncoder
import pandas as pd
import joblib
# 6. Gunakan cross_val_score untuk evaluasi (5-fold cross-validation)
from sklearn.model_selection import StratifiedKFold
import sys
from pathlib import Path

root = Path(__file__).parent.parent.resolve()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from app.dependencies.mapping import load_and_clean_data #, map_jurusan

csv_path = root / 'data' / 'Student Mental health.csv'

# 1. Buat contoh dataset (kecil)
df = load_and_clean_data(csv_path)

# 2. Pisahkan fitur dan target
X = df.drop(columns='Do you have Depression?')
y = df['Do you have Depression?']

le = LabelEncoder()
y_encoded = le.fit_transform(y)
# y_decoded = le.inverse_transform(pred) # pred dari prediksi

# 3. Tentukan fitur numerik & kategorikal
numerical_features = ['Age']
# categorical_balanced = ['Choose your gender', 'Marital status']
under_ten = ['What is your course?', 'Your current year of Study',
            'What is your CGPA?']
under_twenty = ['Choose your gender', 'Marital status']

numerical_transformer = Pipeline(steps=[
    ('power', PowerTransformer(method='yeo-johnson')),  # bisa atasi skew
    ('scaler', StandardScaler())
])

# 4. Buat preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat_imb_ten', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=10), under_ten),
        ('cat_imb_twen', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=20), under_twenty),
    ]
)

# 5. Buat pipeline lengkap
pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])

param_grid = {
    'model__n_estimators': [50, 100],
    'model__max_depth': [5, 10],
    'model__max_features': ['sqrt', 'log2']
}

grid = GridSearchCV(pipeline, param_grid, cv=3, error_score='raise')

fold = 5
cv = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42)
scores_accuracy = cross_val_score(grid, X, y_encoded, cv=cv, scoring='accuracy')
scores_precision = cross_val_score(grid, X, y_encoded, cv=cv, scoring='precision')
scores_recall = cross_val_score(grid, X, y_encoded, cv=cv, scoring='recall')
scores_f1 = cross_val_score(grid, X, y_encoded, cv=cv, scoring='f1')

# print(y_encoded)  # Output: array([1, 0, 1, 1, 0])
# 7. Cetak hasil
print("\n===== Estimasi Skor Evaluasi Performa Model dan Pencarian Hyperparameter =====")
print("Jumlah fold Stratified K-Fold: ", fold)
print("Rata-rata akurasi Nested CV:", scores_accuracy.mean())
print("Rata-rata presisi (Yes) Nested CV:", scores_precision.mean())
print("Rata-rata recall (Yes) Nested CV:", scores_recall.mean())
print("Rata-rata f1 (Yes) Nested CV:", scores_f1.mean())

grid.fit(X, y_encoded)

model_path = root / 'pkl' / 'nested_cv.pkl'
joblib.dump(grid, model_path)

# Contoh input manual
data_baru = pd.DataFrame([{
    'What is your course?': 'engineering',
    'Your current year of Study': 'year 3',
    'What is your CGPA?': '3.00 - 3.49',
    'Choose your gender': 'Male',
    'Marital status': 'Yes',
    'Age': 24
}])

# Prediksi
print("\n===== Prediksi input dengan Nested CV =====")
print("Input Pengguna:")
for k, v in data_baru.items():
    print(f"{k}: {v}")

prediksi = grid.predict(data_baru)
print("\nPrediksi:", prediksi[0])
y_decoded = le.inverse_transform(prediksi) # pred dari prediksi
print("Prediksi:", y_decoded)

# Ambil model terbaik
best_model = grid.best_estimator_

scores_accuracy = cross_val_score(best_model, X, y_encoded, cv=cv, scoring='accuracy')
scores_precision = cross_val_score(best_model, X, y_encoded, cv=cv, scoring='precision')
scores_recall = cross_val_score(best_model, X, y_encoded, cv=cv, scoring='recall')
scores_f1 = cross_val_score(best_model, X, y_encoded, cv=cv, scoring='f1')

# print(y_encoded)  # Output: array([1, 0, 1, 1, 0])
# 7. Cetak hasil
print("\n===== Estimasi Skor Evaluasi Performa Model Untuk Deployment =====")
print("Jumlah fold Stratified K-Fold: ", fold)
print("Rata-rata akurasi Best Model GridSearchCV:", scores_accuracy.mean())
print("Rata-rata presisi (Yes) Best Model GridSearchCV:", scores_precision.mean())
print("Rata-rata recall (Yes) Best Model GridSearchCV:", scores_recall.mean())
print("Rata-rata f1 (Yes) Best Model GridSearchCV:", scores_f1.mean())

best_model.fit(X, y_encoded)

model_path = root / 'pkl' / 'best_model.pkl'
joblib.dump(best_model, model_path)

# Prediksi
print("\n===== Prediksi input dengan Best Model GridSearchCV =====")
print("Input Pengguna:")
for k, v in data_baru.items():
    print(f"{k}: {v}")

prediksi = best_model.predict(data_baru)
print("\nPrediksi:", prediksi[0])
y_decoded = le.inverse_transform(prediksi) # pred dari prediksi
print("Prediksi:", y_decoded)
