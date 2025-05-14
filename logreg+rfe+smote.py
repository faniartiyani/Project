# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import pickle

# ========== 1. LOAD DAN PREPROCESS DATA ==========
dataset = pd.read_csv('./data breast cancer.csv')

for col in ['concavity_mean', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst']:
    dataset[col] = dataset[col].astype('float')

df = dataset.copy()
df['diagnosis'] = df['diagnosis'].fillna(df['diagnosis'].mode()[0])
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# ========== 2. SPLIT DATA ==========
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ========== 3. SCALING DAN FEATURE SELECTION SEBELUM SMOTE ==========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

estimator = LogisticRegression(max_iter=10000, solver='liblinear')
rfecv = RFECV(estimator=estimator, step=1, cv=5, scoring='accuracy')
rfecv.fit(X_train_scaled, y_train)
selected_features = X.columns[rfecv.support_]

# ========== 4. SMOTE SETELAH FITUR TERPILIH ==========
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_selected, y_train)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(x=y_train, palette='Blues')
plt.title("Distribusi Sebelum SMOTE")
plt.xlabel("Kelas")
plt.ylabel("Jumlah")

plt.subplot(1, 2, 2)
sns.countplot(x=y_train_smote, hue=y_train_smote, palette='Blues', legend=False)
plt.title("Distribusi Setelah SMOTE")
plt.xlabel("Kelas")
plt.ylabel("Jumlah")

plt.tight_layout()
os.makedirs('models/logreg+rfe+smote', exist_ok=True)
plt.savefig('models/logreg+rfe+smote/distribusi_kelas_smote.png', dpi=300)
plt.show()

# Scaling ulang setelah SMOTE
X_train_smote_scaled = scaler.fit_transform(X_train_smote)
X_test_selected_scaled = scaler.transform(X_test_selected)

# ========== 5. VISUALISASI RANKING FITUR ==========
ranking_df = pd.DataFrame({
    'Fitur': X.columns,
    'Dipilih': rfecv.support_,
    'Ranking': rfecv.ranking_
})
ranking_df = ranking_df[ranking_df['Dipilih']]
ranking_df['Koefisien'] = rfecv.estimator_.coef_[0]
ranking_df = ranking_df.sort_values(by='Ranking').reset_index(drop=True)

plt.figure(figsize=(10, 6))
sns.barplot(
    x='Koefisien',
    y='Fitur',
    data=ranking_df,
    palette='Blues_r'
)
for i, row in ranking_df.iterrows():
    plt.text(row.Koefisien + 0.05 if row.Koefisien > 0 else row.Koefisien - 0.3, i, f"{row.Koefisien:.2f}", va='center')

plt.title('Fitur Terpilih Berdasarkan RFECV (Logreg)', fontsize=14)
plt.xlabel('Koefisien Pentingnya Fitur', fontsize=12)
plt.ylabel('Fitur', fontsize=12)
plt.tight_layout()
plt.savefig('models/logreg+rfe+smote/fitur_terpilih_rfecv_logreg.png', dpi=300)
plt.show()

# ========== 6. GRIDSEARCH DAN EVALUASI ==========
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'penalty': ['l1'],
    'solver': ['liblinear'],
    'max_iter': [100, 200, 500]
}

grid_search = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5, scoring='balanced_accuracy', n_jobs=-1)
grid_search.fit(X_train_smote_scaled, y_train_smote)
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test_selected_scaled)
acc = accuracy_score(y_test, y_pred)
print("\nBest Parameters:", grid_search.best_params_)
print(f"Akurasi: {acc:.4f}")
print(classification_report(y_test, y_pred))

# ========== 7. CONFUSION MATRIX ==========
def plot_confusion_matrix(y_true, y_pred, title, filename=None):
    cmatrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cmatrix, annot=True, fmt='g', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.title(f'Confusion Matrix: {title}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    if filename:
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
    plt.show()

plot_confusion_matrix(y_test, y_pred, "Logistic Regression (Setelah SMOTE)", 
                      filename='models/logreg+rfe+smote/conmat-logreg+rfe+smote.png')

# ========== 8. SIMPAN MODEL DAN DATA ==========
data_yang_ingin_disimpan = {
    'X_train': X_train_smote_scaled,
    'y_train': y_train_smote,
    'X_test': X_test_selected_scaled,
    'y_test': y_test,
    'best_logreg_after': best_model,
    'selected_features': selected_features
}

with open('models/logreg+rfe+smote/data_file_gab_logreg.pkl', 'wb') as f:
    pickle.dump(data_yang_ingin_disimpan, f)

with open('models/logreg+rfe+smote/akurasiakhir-logreg+rfe+smote.pkl', 'wb') as f1:
    pickle.dump(acc, f1)

with open('models/logreg+rfe+smote/cfreport-logreg+rfe+smote.pkl', 'wb') as f2:
    pickle.dump(classification_report(y_test, y_pred), f2)

with open('models/logreg+rfe+smote/logreg+rfe+smote.pkl', 'wb') as f3:
    pickle.dump(y_pred, f3)