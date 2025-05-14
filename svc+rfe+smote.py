import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
import pickle
import warnings
import os

# Abaikan peringatan
warnings.filterwarnings('ignore')
sns.set()

# ========== BACA DATASET ==========

df = pd.read_csv('./data breast cancer.csv')

# Pastikan kolom numerik dalam tipe float
numerik_kolom = [
    'concavity_mean', 'smoothness_worst', 'compactness_worst',
    'concavity_worst', 'concave points_worst', 'symmetry_worst'
]
for col in numerik_kolom:
    df[col] = df[col].astype('float')

# Isi nilai diagnosis jika ada yang kosong
if df['diagnosis'].isnull().sum() > 0:
    df['diagnosis'].fillna(df['diagnosis'].mode()[0], inplace=True)

# Label encoding diagnosis
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
y = df['diagnosis']
X = df.drop('diagnosis', axis=1)

# ========== SPLIT DATA ==========

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========== STANDARDISASI & SMOTE ==========

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE hanya untuk data latih
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train_scaled, y_train)

# Cek distribusi kelas
class_dist_before = Counter(y_train)
class_dist_after = Counter(y_res)

print("Distribusi kelas sebelum SMOTE:")
print(f"Benign: {class_dist_before[0]}, Malignant: {class_dist_before[1]}")

print("\nDistribusi kelas setelah SMOTE:")
print(f"Benign: {class_dist_after[0]}, Malignant: {class_dist_after[1]}")

# ========== VISUALISASI DISTRIBUSI KELAS ==========

# Barplot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.countplot(x=y_train, palette="Blues")
plt.title("Distribusi Kelas Sebelum SMOTE")
plt.xticks([0, 1], ['Benign', 'Malignant'])

plt.subplot(1, 2, 2)
sns.countplot(x=y_res, palette="Greens")
plt.title("Distribusi Kelas Setelah SMOTE")
plt.xticks([0, 1], ['Benign', 'Malignant'])

plt.tight_layout()
os.makedirs("models/svc+rfe+smote", exist_ok=True)
plt.savefig("models/svc+rfe+smote/distribusi_barplot.png", dpi=300)
plt.show()

# Pie chart
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.pie(
    [class_dist_before[0], class_dist_before[1]],
    labels=['Benign', 'Malignant'],
    autopct='%1.1f%%',
    colors=['#4C72B0', '#55A868']
)
plt.title('Distribusi Sebelum SMOTE')

plt.subplot(1, 2, 2)
plt.pie(
    [class_dist_after[0], class_dist_after[1]],
    labels=['Benign', 'Malignant'],
    autopct='%1.1f%%',
    colors=['#8C564B', '#E377C2']
)
plt.title('Distribusi Setelah SMOTE')

plt.tight_layout()
plt.savefig("models/svc+rfe+smote/distribusi_piechart.png", dpi=300)
plt.show()

# ========== RFECV DENGAN LinearSVC ==========

svc_linear = LinearSVC(max_iter=10000, dual=False)
rfecv = RFECV(estimator=svc_linear, step=1, cv=5, scoring='accuracy')
rfecv.fit(X_res, y_res)

fitur_names = X.columns
ranking = rfecv.ranking_

# Visualisasi ranking fitur
ranking_df = pd.DataFrame({
    'Fitur': fitur_names,
    'Ranking': ranking
}).sort_values(by='Ranking', ascending=True)

colors = sns.color_palette("Blues", n_colors=ranking_df['Ranking'].nunique())[::-1]
palette_dict = dict(zip(sorted(ranking_df['Ranking'].unique()), colors))
ranking_df['Color'] = ranking_df['Ranking'].map(palette_dict)

plt.figure(figsize=(10, 12))
sns.barplot(
    x='Ranking',
    y='Fitur',
    data=ranking_df,
    palette=ranking_df['Color'].tolist()
)
plt.title("📊 Ranking Fitur Berdasarkan RFECV (LinearSVC)", fontsize=14)
plt.xlabel("Ranking (1 = paling penting)")
plt.ylabel("Nama Fitur")
plt.tight_layout()
plt.savefig("models/svc+rfe+smote/rfecv_svc_all_feature_ranking.png", dpi=300)
plt.show()

# ========== VISUALISASI FITUR TERPILIH + KOEFISIEN ==========

selected_features = fitur_names[rfecv.support_]
coef_values = rfecv.estimator_.coef_[0]

coef_df = pd.DataFrame({
    'Fitur': selected_features,
    'Koefisien': coef_values
}).sort_values(by='Koefisien', ascending=True)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Koefisien', y='Fitur', data=coef_df, palette='Blues_r')

for i, row in coef_df.iterrows():
    ax.text(
        row.Koefisien + 0.05 if row.Koefisien > 0 else row.Koefisien - 0.3,
        i,
        f"{row.Koefisien:.2f}",
        va='center'
    )

plt.title("🎯 Fitur Terpilih Berdasarkan RFECV (LinearSVC)", fontsize=14)
plt.xlabel("Koefisien Pentingnya Fitur")
plt.ylabel("Nama Fitur")
plt.tight_layout()
plt.savefig("models/svc+rfe+smote/rfecv_svc_selected_feature.png", dpi=300)
plt.show()

# ========== TRAIN FINAL MODEL & EVALUASI ==========

X_train_rfe = X_res[:, rfecv.support_]
X_test_rfe = X_test_scaled[:, rfecv.support_]

svc_model = SVC(kernel='linear', random_state=42)
svc_model.fit(X_train_rfe, y_res)
y_pred = svc_model.predict(X_test_rfe)

acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Akurasi SVC + RFECV (Setelah SMOTE): {acc:.4f}")
print(classification_report(y_test, y_pred))

# Confusion matrix
def plot_confusion_matrix(y_true, y_pred, title, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.title(f'Confusion Matrix: {title}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

plot_confusion_matrix(y_test, y_pred, "SVC + RFE (Setelah SMOTE)",
                      save_path="models/svc+rfe+smote/conmat-svc+rfe+smote.png")

# Simpan model & data
pickle_data = {
    'model': svc_model,
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test
}
with open('models/svc+rfe+smote/data_file_gab_svc.pkl', 'wb') as f:
    pickle.dump(pickle_data, f)

with open('models/svc+rfe+smote/akurasiakhir-svcrfe+smote.pkl', 'wb') as f1:
    pickle.dump(acc, f1)

with open('models/svc+rfe+smote/cfreport-svcrfe+smote.pkl', 'wb') as f2:
    pickle.dump(classification_report(y_test, y_pred), f2)

with open('models/svc+rfe+smote/svcrfe+smote.pkl', 'wb') as f3:
    pickle.dump(y_pred, f3)

print("\n✅ Semua model, visualisasi, dan evaluasi berhasil disimpan di folder ./models/svc+rfe+smote")
