import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os
import warnings

from models.preprocessing import preprocess_data  # Pastikan file ini ada dan benar

# Pengaturan
warnings.filterwarnings("ignore")
sns.set()

# === 1. BACA DATASET DAN PREPROCESS ===
df = pd.read_csv('./data breast cancer.csv')
X_scaled, y = preprocess_data(df)

# === 2. SELEKSI FITUR MENGGUNAKAN RFE (10 FITUR) ===
svc_estimator = SVC(kernel='linear', random_state=42)
rfe = RFE(estimator=svc_estimator, n_features_to_select=10)
rfe.fit(X_scaled, y)
selected_features = X_scaled.columns[rfe.support_]

# Ambil koefisien dari model yang telah di-fit oleh RFE
coef = rfe.estimator_.coef_[0]  # pastikan ini hanya untuk kernel='linear'


# Tampilkan fitur terpilih
print("Fitur-fitur terpilih oleh RFE:")
print(selected_features.tolist())

# Buat DataFrame koefisien
coef_df = pd.DataFrame({
    'Fitur': selected_features,
    'Koefisien': coef
}).sort_values(by='Koefisien', ascending=False)

# Plot koefisien fitur
plt.figure(figsize=(10, 6))
sns.barplot(x='Koefisien', y='Fitur', data=coef_df, palette='Blues_r')
plt.axvline(0, color='gray', linestyle='--')
plt.title('🔍 Fitur Terpilih Berdasarkan RFE (SVC Linear)')
plt.xlabel('Koefisien Pentingnya Fitur')
plt.ylabel('Nama Fitur')

# Tambahkan label nilai koefisien
for i, (value, name) in enumerate(zip(coef_df['Koefisien'], coef_df['Fitur'])):
    plt.text(value, i, f'{value:.2f}', va='center',
             ha='left' if value > 0 else 'right', color='black')

plt.tight_layout()
os.makedirs('models/svc_rfe', exist_ok=True)
plt.savefig('models/svc_rfe/fitur_terpilih_rfecv_SVC.png', dpi=300)
plt.show()

# === 3. VISUALISASI RANKING FITUR ===
ranking_df = pd.DataFrame({
    'Fitur': X_scaled.columns,
    'Ranking': rfe.ranking_
}).sort_values(by='Ranking')

plt.figure(figsize=(10, 6))
sns.barplot(x='Ranking', y='Fitur', data=ranking_df,
            palette=sns.color_palette("Blues", n_colors=len(ranking_df)))
plt.title("Ranking Fitur RFE (SVC Linear)")
plt.xlabel("Ranking (1 = paling penting)")
plt.tight_layout()
plt.savefig("models/svc_rfe/feature_ranking.png", dpi=300)
plt.show()

# === 4. SPLIT DATA DAN GRIDSEARCH ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled[selected_features], y, test_size=0.3, random_state=42
)

stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {'C': [0.01, 0.1, 1, 10]}
grid = GridSearchCV(
    SVC(kernel='linear', class_weight='balanced'),
    param_grid,
    cv=stratified_kfold,
    scoring='accuracy',
    n_jobs=-1
)
grid.fit(X_train, y_train)

# === 5. EVALUASI MODEL TERBAIK ===
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi SVC + RFE: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# === 6. CONFUSION MATRIX ===
def plot_conf_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.title("Confusion Matrix: SVC + RFE")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

plot_conf_matrix(y_test, y_pred, save_path="models/svc_rfe/conf_matrix.png")

# === 7. SIMPAN MODEL DAN DATA ===
with open('./models/svc_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open("models/svc_rfe/model_svc_rfe.pkl", 'wb') as f:
    pickle.dump(best_model, f)

with open("models/svc_rfe/accuracy.pkl", 'wb') as f:
    pickle.dump(accuracy, f)

with open("models/svc_rfe/classification_report.pkl", 'wb') as f:
    pickle.dump(classification_report(y_test, y_pred), f)

with open("models/svc_rfe/data_splits.pkl", 'wb') as f:
    pickle.dump({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }, f)
