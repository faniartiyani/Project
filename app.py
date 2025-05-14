from flask import Flask, render_template, request, jsonify
import pickle
import os
import pandas as pd
import warnings
from flask import send_from_directory


# Nonaktifkan peringatan FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# ==== Load Accuracy & Classification Report ====

# Logistic Regression + RFE
with open('./models/logreg+rfe/akurasiakhir-logreg+rfe.pkl', 'rb') as f:
    akurasi_logreg_rfe = pickle.load(f)

with open('./models/logreg+rfe/cfreport-logreg+rfe.pkl', 'rb') as f:
    classification_report_logreg_rfe = pickle.load(f)

# Logistic Regression + RFE + SMOTE
with open('./models/logreg+rfe+smote/akurasiakhir-logreg+rfe+smote.pkl', 'rb') as f:
    accuracy_logreg_rfe_smote = pickle.load(f)

with open('./models/logreg+rfe+smote/cfreport-logreg+rfe+smote.pkl', 'rb') as f:
    classification_report_logreg_rfe_smote = pickle.load(f)

# SVC + RFE
with open('./models/svc_rfe/accuracy.pkl', 'rb') as f:
    akurasi_svc_rfe = pickle.load(f)

with open('./models/svc_rfe/classification_report.pkl', 'rb') as f:
    classification_report_svc_rfe = pickle.load(f)

# SVC + RFE + SMOTE
with open('./models/svc+rfe+smote/akurasiakhir-svcrfe+smote.pkl', 'rb') as f:
    accuracy_svc_rfe_smote = pickle.load(f)

with open('./models/svc+rfe+smote/cfreport-svcrfe+smote.pkl', 'rb') as f:
    classification_report_svc_rfe_smote = pickle.load(f)

# Classification report terbaik
with open('./models/logreg+rfe/cfreport-logreg+rfe.pkl', 'rb') as f:
    classification_report_logreg_rfe = pickle.load(f)

with open('./models/logreg+rfe+smote/cfreport-logreg+rfe+smote.pkl', 'rb') as f:
    classification_report_str = pickle.load(f)

with open('./models/svc_rfe/classification_report.pkl', 'rb') as f:
    classification_report_svc_rfe = pickle.load(f)

with open('./models/svc+rfe+smote/cfreport-svcrfe+smote.pkl', 'rb') as f:
    classification_report_svc_rfe_smote = pickle.load(f)

# ==== Load Preprocessing dan Model ====
scaler = pickle.load(open('./models/scaler.pkl', 'rb'))
label_encoder = pickle.load(open('./models/label_encoder.pkl', 'rb'))

# Logistic Regression
logrfe = pickle.load(open('./models/logrfe.pkl', 'rb'))
logrfe_model = pickle.load(open('./models/logrfe_model.pkl', 'rb'))

logrfesmote = pickle.load(open('./models/logrfesmote.pkl', 'rb'))
logrfesmote_model = pickle.load(open('./models/logrfesmote_model.pkl', 'rb'))

# SVC
svcrfe = pickle.load(open('./models/svcrfe.pkl', 'rb'))
svcrfe_model = pickle.load(open('./models/svcrfe_model.pkl', 'rb'))

svcrfesmote = pickle.load(open('./models/svcrfesmote.pkl', 'rb'))
svcrfesmote_model = pickle.load(open('./models/svcrfesmote_model.pkl', 'rb'))

# ==== Flask App Initialization ====

app = Flask(__name__, template_folder='frontend', static_folder='static')

# ==== Routes ====

@app.route('/models/<path:filename>')
def model_image(filename):
    models_dir = os.path.join(app.root_path, 'models')
    return send_from_directory(models_dir, filename)

@app.route('/')
def home():
    return render_template('index.html', css_file='css/style.css')

@app.route('/about-us')
def about():
    return render_template('about-us.html', css_file='css/about-us.css')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', css_file='css/bootstrap.min.css')

@app.route('/pengujian')
def pengujian():
    return render_template('pengujian.html', css_file='css/bootstrap.min.css')

@app.route('/dataset')
def dataset():
    # Load dataset asli dan hasil standarisasi
    df_asli = pd.read_csv('data breast cancer.csv')
    df_standar = pd.read_csv('hasil_standarisasi.csv')

    return render_template('dataset.html', df_databreastcancer=df_asli,
                           df_databreastcancer_baru=df_standar,
                           css_file='css/bootstrap.min.css')


# Mendefinisikan fitur_dict di awal
fitur_dict = {
    'logreg_rfe': ['concavity_worst' , 'radius_worst' , 'compactness_worst', 'texture_se', 'symmetry_worst', 'concavity_mean', 'concave points_worst', 'compactness_mean', 'smoothness_worst' , 'radius_mean' ],
    'logreg_rfecv': ['texture_mean', 'area_mean', 'concave points_mean', 'radius_se', 'area_se', 'compactness_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'perimeter_mean'],
    'svc_rfecv1': ['concavity_worst','area_worst', 'texture_worst', 'area_se', 'radius_worst', 'radius_se', 'smoothness_worst', 'concave points_se', 'texture_se', 'compactness_se'],
    'svc_rfecv2': ['texture_worst', 'radius_se', 'concavity_worst', 'concave points_mean', 'area_se', 'symmetry_worst', 'concave points_se', 'perimeter_se', 'area_worst', 'smoothness_mean', 'fractal_dimension_mean', 'perimeter_worst', 'concavity_mean', 'radius_worst', 'symmetry_mean', 'symmetry_se', 'compactness_worst', 'concavity_se', 'texture_se', 'compactness_se', 'compactness_mean']
}

@app.route('/predict', methods=['POST'])
def predict():
    model_name = request.form['model']
    input_data = request.form.to_dict(flat=True)
    input_data.pop('model', None)

    # Konversi data ke bentuk numerik
    new_data = {k: [float(v)] for k, v in input_data.items()}
    new_data_df = pd.DataFrame(new_data)

    label_map = {1: 'M', 0: 'B'}

    try:
        if model_name == 'logreg_rfe':
            # Pastikan kolom yang dipilih cocok dengan data baru
            selected_columns = new_data_df.columns[logrfe.support_]
            selected_columns = [col for col in selected_columns if col in new_data_df.columns]
            new_data_selected = new_data_df[selected_columns]
            prediction = logrfe_model.predict(new_data_selected)

        elif model_name == 'logreg_rfecv':
            selected_columns = [col for col in logrfesmote['features'] if col in new_data_df.columns]
            new_data_selected = new_data_df[selected_columns]
            prediction = logrfesmote_model.predict(new_data_selected)

        elif model_name == 'svc_rfecv1':
            selected_columns = new_data_df.columns[svcrfe.support_]
            selected_columns = [col for col in selected_columns if col in new_data_df.columns]
            new_data_selected = new_data_df[selected_columns]
            prediction = svcrfe_model.predict(new_data_selected)

        elif model_name == 'svc_rfecv2':
            selected_columns = [col for col in svcrfesmote['features'] if col in new_data_df.columns]
            new_data_selected = new_data_df[selected_columns]
            prediction = svcrfesmote_model.predict(new_data_selected)

        else:
            return jsonify({'error': 'Model tidak dikenal'}), 400

        decoded_prediction = [label_map[p] for p in prediction]
        return jsonify(decoded_prediction[0])

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/result')
def result():
    return render_template('result.html',
                           akurasi_logreg_rfe=akurasi_logreg_rfe,
                           classification_report_logreg_rfe=classification_report_logreg_rfe,
                           accuracy_logreg_rfe_smote=accuracy_logreg_rfe_smote,
                           classification_report_logreg_rfe_smote=classification_report_logreg_rfe_smote,
                           akurasi_svc_rfe=akurasi_svc_rfe,
                           classification_report_svc_rfe=classification_report_svc_rfe,
                           accuracy_svc_rfe_smote=accuracy_svc_rfe_smote,
                           classification_report_svc_rfe_smote=classification_report_svc_rfe_smote,
                           classification_report_str=classification_report_str,
                           css_file='css/bootstrap.min.css')

# ==== Run App ====

if __name__ == '__main__':
    app.run(debug=True)
