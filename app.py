import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------
# Streamlit App: Heart Disease Risk Prediction (XGBoost)
# ----------------------------

st.set_page_config(
    page_title="Prediksi Risiko Penyakit Jantung (XGBoost)",
    page_icon="❤️",
    layout="wide",
)


def inject_custom_css():
    st.markdown(
        """
        <style>
        /* Background dan layout utama */
        .stApp {
            background: radial-gradient(circle at top left, #f9fcff 0, #e3f2fd 35%, #e8f5e9 100%);
            font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        /* Container utama */
        .main {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
        }

        /* Kartu judul */
        .hero-card {
            background: linear-gradient(120deg, #1565c0, #00897b);
            border-radius: 18px;
            padding: 1.4rem 1.8rem;
            color: white !important;
            box-shadow: 0 18px 45px rgba(15, 76, 129, 0.35);
            margin-bottom: 1.2rem;
        }

        .hero-title {
            font-size: 1.9rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }

        .hero-subtitle {
            font-size: 0.95rem;
            opacity: 0.95;
        }

        /* Card section */
        .section-card {
            background: rgba(255, 255, 255, 0.96);
            border-radius: 16px;
            padding: 1.2rem 1.3rem;
            box-shadow: 0 10px 28px rgba(15, 76, 129, 0.18);
            border: 1px solid rgba(21, 101, 192, 0.08);
            margin-bottom: 1.1rem;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d47a1, #004d40);
        }

        section[data-testid="stSidebar"] * {
            color: #e3f2fd !important;
        }

        /* Tombol utama */
        .stButton>button[kind="primary"] {
            background: linear-gradient(120deg, #1565c0, #00897b);
            color: #ffffff;
            border-radius: 999px;
            border: none;
            padding: 0.55rem 1.6rem;
            font-weight: 600;
            box-shadow: 0 10px 22px rgba(21, 101, 192, 0.35);
        }

        .stButton>button[kind="primary"]:hover {
            background: linear-gradient(120deg, #0d47a1, #00695c);
            box-shadow: 0 14px 30px rgba(13, 71, 161, 0.4);
        }

        /* Progress bar warna risiko */
        .stProgress > div > div {
            background: linear-gradient(90deg, #43a047, #fdd835, #e53935);
        }

        /* Metric card tweak */
        [data-testid="stMetricValue"] {
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_custom_css()

MODEL_PATH_DEFAULT = "xgb_heart_pipeline.joblib"

@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)

def get_required_columns():
    return [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]

def build_single_row_input(
    age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
) -> pd.DataFrame:
    data = {
        "age": [age],
        "sex": [sex],
        "cp": [cp],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [fbs],
        "restecg": [restecg],
        "thalach": [thalach],
        "exang": [exang],
        "oldpeak": [oldpeak],
        "slope": [slope],
        "ca": [ca],
        "thal": [thal],
    }
    return pd.DataFrame(data)

def safe_predict(pipeline, X: pd.DataFrame):
    pred = pipeline.predict(X)
    proba = pipeline.predict_proba(X)[:, 1]  # prob kelas 1
    return pred, proba

def format_percent(x: float) -> str:
    return f"{x*100:.2f}%"

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">❤️ Prediksi Risiko Penyakit Jantung (XGBoost)</div>
        <div class="hero-subtitle">
            Bantu tenaga kesehatan dan pengguna umum untuk mengestimasi risiko penyakit jantung
            berdasarkan data klinis sederhana. Dibangun untuk kebutuhan Tugas Akhir.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Aplikasi ini menggunakan model **XGBoost** untuk memprediksi label **target (0/1)** "
    "berdasarkan fitur klinis pada dataset tabular."
)

with st.expander("ℹ️ Cara pakai", expanded=False):
    st.markdown(
        """
- Pilih **Mode Input**: *Manual* atau *Upload CSV*.
- Klik **Prediksi** untuk melihat hasil.
- Output utama: **kelas prediksi (0/1)** dan **probabilitas risiko (kelas 1)**.
> Catatan: Ini adalah demo akademik, **bukan alat diagnosis medis**.
        """
    )

# ----------------------------
# Model loader (sidebar)
# ----------------------------
st.sidebar.header("Pengaturan Model")
model_path = st.sidebar.text_input("Path model (.joblib)", value=MODEL_PATH_DEFAULT)

pipeline = None
model_error = None
if os.path.exists(model_path):
    try:
        pipeline = load_model(model_path)
        st.sidebar.success("Model berhasil dimuat ✅")
    except Exception as e:
        model_error = str(e)
else:
    model_error = f"File model tidak ditemukan: {model_path}"

uploaded_model = st.sidebar.file_uploader("Atau upload model .joblib", type=["joblib"])
if uploaded_model is not None:
    # simpan ke file sementara
    tmp_path = "uploaded_model.joblib"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_model.read())
    try:
        pipeline = load_model(tmp_path)
        model_error = None
        st.sidebar.success("Model dari upload berhasil dimuat ✅")
    except Exception as e:
        model_error = str(e)

if model_error and pipeline is None:
    st.error(
        "Model belum siap. Pastikan file **xgb_heart_pipeline.joblib** sudah ada "
        "(jalankan cell 'Simpan Model untuk Deployment' di notebook), "
        "atau upload file model di sidebar.\n\n"
        f"Detail: {model_error}"
    )
    st.stop()

# ----------------------------
# Mode Input
# ----------------------------
mode = st.radio("Mode Input", ["Manual Input", "Upload CSV (Batch)"], horizontal=True)

# ============================
# Manual Input
# ============================
if mode == "Manual Input":
    st.subheader("🧾 Input Manual")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age (umur)", 1, 120, 54)
        trestbps = st.slider("trestbps (tekanan darah istirahat)", 50, 250, 130)
        chol = st.slider("chol (kolesterol)", 80, 600, 240)
        thalach = st.slider("thalach (max heart rate)", 50, 250, 150)
        oldpeak = st.slider("oldpeak (ST depression)", 0.0, 10.0, 1.0, 0.1)

    with col2:
        sex = st.selectbox("sex", options=[0, 1], format_func=lambda x: "0 (Female)" if x == 0 else "1 (Male)")
        cp = st.selectbox(
            "cp (chest pain type)",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "0 (Typical angina)",
                1: "1 (Atypical angina)",
                2: "2 (Non-anginal pain)",
                3: "3 (Asymptomatic)",
            }[x],
        )
        fbs = st.selectbox("fbs (fasting blood sugar > 120 mg/dl)", options=[0, 1])
        restecg = st.selectbox(
            "restecg (resting ECG)",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "0 (Normal)",
                1: "1 (ST-T abnormality)",
                2: "2 (Left ventricular hypertrophy)",
            }[x],
        )
        exang = st.selectbox("exang (exercise induced angina)", options=[0, 1])
        slope = st.selectbox(
            "slope (slope of peak exercise ST segment)",
            options=[0, 1, 2],
            format_func=lambda x: {0: "0 (Upsloping)", 1: "1 (Flat)", 2: "2 (Downsloping)"}[x],
        )
        ca = st.selectbox("ca (number of major vessels)", options=[0, 1, 2, 3, 4])
        thal = st.selectbox(
            "thal",
            options=[0, 1, 2, 3],
            format_func=lambda x: {0: "0 (Unknown)", 1: "1 (Normal)", 2: "2 (Fixed defect)", 3: "3 (Reversible defect)"}[x],
        )

    X_input = build_single_row_input(
        age=age,
        sex=sex,
        cp=cp,
        trestbps=trestbps,
        chol=chol,
        fbs=fbs,
        restecg=restecg,
        thalach=thalach,
        exang=exang,
        oldpeak=oldpeak,
        slope=slope,
        ca=ca,
        thal=thal,
    )

    st.markdown("#### Preview Input")
    st.dataframe(X_input, use_container_width=True)

    if st.button("🔮 Prediksi", type="primary"):
        pred, proba = safe_predict(pipeline, X_input)
        pred_class = int(pred[0])
        risk_prob = float(proba[0])

        st.markdown("### ✅ Hasil Prediksi")
        c1, c2, c3 = st.columns(3)
        c1.metric("Prediksi Kelas", str(pred_class))
        c2.metric("Prob. Risiko (kelas 1)", format_percent(risk_prob))
        c3.metric("Prob. Tidak Risiko (kelas 0)", format_percent(1 - risk_prob))

        st.progress(min(max(risk_prob, 0.0), 1.0))

        if pred_class == 1:
            st.warning("Model memprediksi **RISIKO penyakit jantung (target=1)**.")
        else:
            st.success("Model memprediksi **TIDAK berisiko (target=0)**.")

        st.caption("Interpretasi: probabilitas adalah keluaran model dan bukan diagnosis medis.")

# ============================
# Batch CSV
# ============================
else:
    st.subheader("📤 Upload CSV (Batch Prediction)")
    st.write("CSV harus berisi kolom fitur berikut (tanpa kolom `target`):")
    st.code(", ".join(get_required_columns()))

    uploaded = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
        except Exception:
            st.error("Gagal membaca CSV. Pastikan format file benar.")
            st.stop()

        required = set(get_required_columns())
        missing_cols = sorted(list(required - set(df_up.columns)))

        if missing_cols:
            st.error(f"Kolom berikut tidak ditemukan di CSV: {missing_cols}")
            st.stop()

        X_batch = df_up[get_required_columns()].copy()

        st.markdown("#### Preview Data")
        st.dataframe(X_batch.head(20), use_container_width=True)

        if st.button("🔮 Prediksi Batch", type="primary"):
            pred, proba = safe_predict(pipeline, X_batch)
            out = df_up.copy()
            out["pred_class"] = pred.astype(int)
            out["prob_risk_class1"] = proba.astype(float)

            st.success("Prediksi batch selesai ✅")
            st.dataframe(out.head(50), use_container_width=True)

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download hasil prediksi (CSV)",
                data=csv_bytes,
                file_name="prediksi_heart_xgboost.csv",
                mime="text/csv",
            )

# ----------------------------
# Feature Importance (Opsional tampil)
# ----------------------------
st.divider()
with st.expander("📌 Feature Importance (Model Interpretability)", expanded=False):
    st.write(
        "Bagian ini menampilkan fitur yang paling berpengaruh menurut model XGBoost. "
        "Jika pipeline menggunakan OneHotEncoder, nama fitur bisa menjadi seperti `cat__cp_3`, dll."
    )
    try:
        pre = pipeline.named_steps.get("preprocess")
        xgb = pipeline.named_steps.get("model")
        if pre is None or xgb is None:
            st.info("Pipeline tidak memiliki langkah preprocess/model yang dikenali.")
        else:
            feature_names = pre.get_feature_names_out()
            importances = xgb.feature_importances_
            fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
            top_n = st.slider("Top-N fitur", 5, 30, 15)
            st.dataframe(fi.head(top_n).reset_index().rename(columns={"index": "feature", 0: "importance"}))
            st.bar_chart(fi.head(top_n))
    except Exception as e:
        st.info(f"Tidak dapat menampilkan feature importance: {e}")

st.caption("© TA-14 — XGBoost Heart Disease | Deployment demo dengan Streamlit")
