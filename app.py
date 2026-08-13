import json
import os
import warnings
from datetime import datetime
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
warnings.filterwarnings("ignore")


# KONFIGURASI DASAR
st.set_page_config(
    page_title="Klasifikasi Risiko Burnout Pekerja Tech",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
HISTORY_FILE = os.path.join(BASE_DIR, "riwayat_klasifikasi.json")

MODEL_PATHS = {
    "XGBoost": os.path.join(MODEL_DIR, "xgboost_burnout_pipeline.joblib"),
    "Random Forest": os.path.join(MODEL_DIR, "random_forest_burnout_pipeline.joblib"),
}

# Urutan fitur HARUS sama persis dengan urutan 27 variabel saat training.
# Kalau urutan/nama kolom di dataset training kamu berbeda, ubah di sini.
FEATURE_ORDER = [
    "age", "gender", "country",
    "job_role", "seniority_level", "years_experience", "years_at_company",
    "company_size", "industry", "work_mode", "salary_usd",
    "work_hours_per_week", "meetings_per_day", "team_size",
    "sleep_hours_per_night", "exercise_days_per_week", "vacation_days_taken",
    "therapy_access", "uses_therapy", "ai_tools_daily",
    "manager_support_score", "work_life_balance_score", "job_satisfaction_score",
    "social_support_score", "deadline_pressure_score", "autonomy_score", "stress_score",
]

CLASS_ORDER = ["Low", "Moderate", "High", "Severe"]
CLASS_LABELS_ID = {
    "Low": "Risiko Rendah",
    "Moderate": "Risiko Sedang",
    "High": "Risiko Tinggi",
    "Severe": "Risiko Sangat Tinggi",
}
CLASS_COLORS = {
    "Low": "#2ecc71",
    "Moderate": "#f1c40f",
    "High": "#e67e22",
    "Severe": "#e74c3c",
}
CLASS_DESC = {
    "Low": "Kondisi kerja dan gaya hidup relatif seimbang. Tetap jaga rutinitas sehat.",
    "Moderate": "Mulai ada tanda tekanan kerja. Perlu perhatian pada keseimbangan kerja dan gaya hidup.",
    "High": "Tanda-tanda burnout cukup signifikan. Disarankan mengambil langkah pemulihan aktif.",
    "Severe": "Risiko burnout sangat tinggi. Sangat disarankan mencari dukungan profesional segera.",
}


# KONFIGURASI FORM INPUT (27 VARIABEL, DIKELOMPOKKAN 4 KATEGORI SESUAI SITEMAP)
# type: "number" (int), "float", "select", "slider"
# Untuk "select": options berupa list (label_tampilan, nilai_ke_model)
# CATATAN: sesuaikan opsi/skala di bawah ini kalau berbeda dengan dataset training kamu.

VARIABLE_CONFIG = {
    "Demografi": [
        {
            "name": "age",
            "label": "Usia",
            "type": "number",
            "min": 22,
            "max": 55,
            "default": 30,
            "step": 1
        },
        {
            "name": "gender",
            "label": "Jenis Kelamin",
            "type": "select",
            "options": [
                ("Laki-laki", "Male"),
                ("Perempuan", "Female"),
                ("Non-binary", "Non-binary"),
                ("Tidak ingin menyebutkan", "Prefer not to say"),
            ]
        },
        {
            "name": "country",
            "label": "Negara",
            "type": "select",
            "options": [(v, v) for v in [
                "Australia",
                "Brazil",
                "Canada",
                "France",
                "Germany",
                "India",
                "Netherlands",
                "Singapore",
                "UK",
                "USA",
            ]]
        },
    ],

    "Konteks Kerja": [
        {
            "name": "job_role",
            "label": "Peran Pekerjaan",
            "type": "select",
            "options": [(v, v) for v in [
                "Backend Developer",
                "Cloud Engineer",
                "Cybersecurity Engineer",
                "Data Analyst",
                "Data Scientist",
                "DevOps Engineer",
                "Frontend Developer",
                "Full Stack Developer",
                "ML Engineer",
                "Product Manager",
                "QA Engineer",
                "Software Engineer",
            ]]
        },
        {
            "name": "seniority_level",
            "label": "Tingkat Senioritas",
            "type": "select",
            "options": [(v, v) for v in [
                "Junior",
                "Mid",
                "Senior",
                "Lead",
                "Manager",
                "Principal",
            ]]
        },
        {
            "name": "years_experience",
            "label": "Total Pengalaman Kerja (tahun)",
            "type": "number",
            "min": 0,
            "max": 25,
            "default": 5,
            "step": 1
        },
        {
            "name": "years_at_company",
            "label": "Lama Bekerja di Perusahaan Saat Ini (tahun)",
            "type": "float",
            "min": 0.1,
            "max": 15.0,
            "default": 2.0,
            "step": 0.1
        },
        {
            "name": "company_size",
            "label": "Ukuran Perusahaan",
            "type": "select",
            "options": [(v, v) for v in [
                "Startup (1-50)",
                "Small (51-200)",
                "Mid (201-1000)",
                "Large (1001-5000)",
                "Enterprise (5000+)",
            ]]
        },
        {
            "name": "industry",
            "label": "Industri",
            "type": "select",
            "options": [(v, v) for v in [
                "AI / ML Startup",
                "Consulting",
                "Cybersecurity",
                "E-commerce",
                "Enterprise Software",
                "Fintech",
                "Gaming",
                "Healthcare Tech",
                "SaaS / Cloud",
                "Social Media / AdTech",
            ]]
        },
        {
            "name": "work_mode",
            "label": "Mode Kerja",
            "type": "select",
            "options": [
                ("Remote / WFH", "Remote"),
                ("Hybrid", "Hybrid"),
                ("On-site / WFO", "On-site"),
            ]
        },
        {
            "name": "salary_usd",
            "label": "Gaji Tahunan (USD)",
            "type": "number",
            "min": 40000,
            "max": 267961,
            "default": 60000,
            "step": 1000
        },
        {
            "name": "work_hours_per_week",
            "label": "Jam Kerja per Minggu",
            "type": "number",
            "min": 35,
            "max": 72,
            "default": 40,
            "step": 1
        },
        {
            "name": "meetings_per_day",
            "label": "Jumlah Meeting per Hari",
            "type": "float",
            "min": 0.0,
            "max": 12.0,
            "default": 3.0,
            "step": 0.1
        },
        {
            "name": "team_size",
            "label": "Ukuran Tim",
            "type": "number",
            "min": 2,
            "max": 59,
            "default": 8,
            "step": 1
        },
    ],

    "Gaya Hidup": [
        {
            "name": "sleep_hours_per_night",
            "label": "Jam Tidur per Malam",
            "type": "float",
            "min": 3.0,
            "max": 10.0,
            "default": 7.0,
            "step": 0.1
        },
        {
            "name": "exercise_days_per_week",
            "label": "Hari Olahraga per Minggu",
            "type": "number",
            "min": 0,
            "max": 7,
            "default": 3,
            "step": 1
        },
        {
            "name": "vacation_days_taken",
            "label": "Hari Cuti Diambil",
            "type": "number",
            "min": 0,
            "max": 30,
            "default": 10,
            "step": 1
        },
        {
            "name": "therapy_access",
            "label": "Memiliki Akses Terapi/Konseling?",
            "type": "select",
            "options": [
                ("Ya", 1),
                ("Tidak", 0),
            ]
        },
        {
            "name": "uses_therapy",
            "label": "Menggunakan Terapi/Konseling?",
            "type": "select",
            "options": [
                ("Ya", 1),
                ("Tidak", 0),
            ]
        },
        {
            "name": "ai_tools_daily",
            "label": "Menggunakan Tools AI Setiap Hari?",
            "type": "select",
            "options": [
                ("Ya", 1),
                ("Tidak", 0),
            ]
        },
    ],

    "Budaya Tempat Kerja": [
        {
            "name": "manager_support_score",
            "label": "Skor Dukungan Manajer (1-10)",
            "type": "float",
            "min": 1.0,
            "max": 10.0,
            "default": 5.0,
            "step": 0.1
        },
        {
            "name": "work_life_balance_score",
            "label": "Skor Keseimbangan Kerja-Hidup (1-10)",
            "type": "float",
            "min": 1.0,
            "max": 10.0,
            "default": 5.0,
            "step": 0.1
        },
        {
            "name": "job_satisfaction_score",
            "label": "Skor Kepuasan Kerja (1-10)",
            "type": "float",
            "min": 1.0,
            "max": 10.0,
            "default": 5.0,
            "step": 0.1
        },
        {
            "name": "social_support_score",
            "label": "Skor Dukungan Sosial (1-10)",
            "type": "float",
            "min": 1.0,
            "max": 10.0,
            "default": 5.0,
            "step": 0.1
        },
        {
            "name": "deadline_pressure_score",
            "label": "Skor Tekanan Deadline (1-10)",
            "type": "float",
            "min": 1.0,
            "max": 10.0,
            "default": 5.0,
            "step": 0.1
        },
        {
            "name": "autonomy_score",
            "label": "Skor Otonomi Kerja (1-10)",
            "type": "float",
            "min": 1.0,
            "max": 10.0,
            "default": 5.0,
            "step": 0.1
        },
        {
            "name": "stress_score",
            "label": "Skor Stres (1-10)",
            "type": "float",
            "min": 1.0,
            "max": 10.0,
            "default": 5.0,
            "step": 0.1
        },
    ],
}

VARIABLE_INFO_DESC = {
    "age": "Usia Pekerja Tech. Faktor demografis dasar yang sering berkorelasi dengan tahap karier.",
    "gender": "Jenis kelamin Pekerja Tech.",
    "country": "Negara tempat Pekerja Tech bekerja, memengaruhi budaya kerja & regulasi.",
    "job_role": "Peran/posisi pekerjaan Pekerja Tech saat ini.",
    "seniority_level": "Tingkat senioritas dalam jenjang karier.",
    "years_experience": "Total pengalaman kerja profesional (tahun).",
    "years_at_company": "Lama bekerja di perusahaan saat ini (tahun).",
    "company_size": "Skala ukuran perusahaan tempat bekerja.",
    "industry": "Sektor industri tempat bekerja.",
    "work_mode": "Model kerja: remote, hybrid, atau onsite.",
    "salary_usd": "Estimasi gaji tahunan dalam USD.",
    "work_hours_per_week": "Rata-rata jam kerja per minggu.",
    "meetings_per_day": "Rata-rata jumlah meeting per hari kerja.",
    "team_size": "Jumlah anggota tim langsung.",
    "sleep_hours_per_night": "Rata-rata jam tidur per malam indikator pemulihan fisik.",
    "exercise_days_per_week": "Frekuensi olahraga per minggu.",
    "vacation_days_taken": "Jumlah hari cuti yang benar-benar diambil per tahun.",
    "therapy_access": "Apakah punya akses ke layanan terapi/konseling (dari kantor/asuransi).",
    "uses_therapy": "Apakah benar-benar memanfaatkan layanan terapi/konseling.",
    "ai_tools_daily": "Apakah menggunakan tools AI (mis. asisten AI) setiap hari kerja.",
    "manager_support_score": "Persepsi tingkat dukungan dari atasan langsung.",
    "work_life_balance_score": "Persepsi keseimbangan antara kerja dan kehidupan pribadi.",
    "job_satisfaction_score": "Tingkat kepuasan terhadap pekerjaan saat ini.",
    "social_support_score": "Persepsi dukungan sosial dari rekan kerja/lingkungan.",
    "deadline_pressure_score": "Tingkat tekanan akibat tenggat waktu pekerjaan.",
    "autonomy_score": "Tingkat kebebasan/otonomi dalam mengambil keputusan kerja.",
    "stress_score": "Tingkat stres kerja yang dirasakan secara umum.",
}


# CSS KUSTOM

st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700; color: #1a1a2e;
        padding-bottom: 0.2rem;
    }
    .sub-header { color: #555; font-size: 1rem; margin-bottom: 1.5rem; }
    .card {
        background-color: #f8f9fb; border-radius: 12px; padding: 1.2rem 1.5rem;
        border: 1px solid #eaeaea; margin-bottom: 1rem;
    }
    .risk-badge {
        display: inline-block; padding: 0.6rem 1.4rem; border-radius: 30px;
        color: white; font-weight: 700; font-size: 1.3rem; text-align: center;
    }
    .step-header {
        font-weight: 700; font-size: 1.05rem; color: #1a1a2e;
        border-left: 4px solid #6c5ce7; padding-left: 0.6rem; margin: 1rem 0 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# LOAD MODEL

@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    errors = {}

    for name, path in MODEL_PATHS.items():
        try:
            if not os.path.exists(path):
                errors[name] = (
                    f"File tidak ditemukan: {path}"
                )
                continue

            # Informasi file
            file_size = os.path.getsize(path)

            # Coba membaca beberapa byte pertama
            with open(path, "rb") as f:
                first_bytes = f.read(100)


            # Load model
            model = joblib.load(path)
            models[name] = model

            print(f"STATUS     : BERHASIL LOAD {name}")
            print("=" * 80)

        except Exception as e:
            import traceback

            error_type = type(e).__name__
            error_message = str(e)
            traceback_text = traceback.format_exc()

            print("=" * 80)
            print(f"MODEL ERROR: {name}")
            print(f"TYPE       : {error_type}")
            print(f"MESSAGE    : {error_message}")
            print("TRACEBACK:")
            print(traceback_text)
            print("=" * 80)

            errors[name] = {
                "type": error_type,
                "message": error_message,
                "traceback": traceback_text,
            }

    return models, errors


TARGET_CLASS_MAP = {
    0: "Low",
    1: "Moderate",
    2: "High",
    3: "Severe",
}


def get_model_classes(model):
    """
    Mengambil kelas asli dari model.
    Biasanya model menghasilkan [0, 1, 2, 3].
    """

    classes = getattr(model, "classes_", None)

    if classes is None:
        try:
            final_step = model.steps[-1][1]
            classes = getattr(final_step, "classes_", None)
        except Exception:
            classes = None

    if classes is None:
        return [0, 1, 2, 3]

    return list(classes)


def predict_risk(model, input_df):
    """
    Menjalankan prediksi dan mengubah hasil numerik
    menjadi Low, Moderate, High, Severe.
    """
    raw_pred = model.predict(input_df)[0]

    try:
        pred_code = int(raw_pred)
        label = TARGET_CLASS_MAP.get(
            pred_code,
            str(raw_pred)
        )
    except (ValueError, TypeError):
        # Kalau model ternyata langsung menghasilkan string
        label = str(raw_pred)

    # PROBABILITAS
    proba_dict = {}

    if hasattr(model, "predict_proba"):

        try:
            probabilities = model.predict_proba(input_df)[0]
            model_classes = get_model_classes(model)

            for cls, probability in zip(
                model_classes,
                probabilities
            ):

                # Model class berupa angka 0,1,2,3
                try:
                    cls_code = int(cls)

                    cls_label = TARGET_CLASS_MAP.get(
                        cls_code,
                        str(cls)
                    )

                except (ValueError, TypeError):

                    # Jika model langsung mempunyai
                    # label Low/Moderate/High/Severe
                    cls_label = str(cls)

                proba_dict[cls_label] = float(probability)

        except Exception as e:

            st.warning(
                f"Probabilitas tidak dapat ditampilkan: {e}"
            )

    return label, proba_dict


# RIWAYAT PREDIKSI (PERSISTEN DI FILE JSON LOKAL)
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:  # noqa: BLE001
            return []
    return []


def save_history(history):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception:  # noqa: BLE001
        pass


def init_session_state():
    if "riwayat" not in st.session_state:
        st.session_state.riwayat = load_history()
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def add_to_history(model_name, input_values, label, proba_dict):
    entry = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "hasil": label,
        "hasil_id": CLASS_LABELS_ID.get(label, label),
        "probabilitas": proba_dict,
        "input": input_values,
    }
    st.session_state.riwayat.insert(0, entry)
    save_history(st.session_state.riwayat)


# KOMPONEN UI

def render_probability_chart(proba_dict):
    if not proba_dict:
        st.info("Model ini tidak menyediakan probabilitas (predict_proba tidak tersedia).")
        return
    labels = [CLASS_LABELS_ID.get(c, c) for c in CLASS_ORDER if c in proba_dict]
    values = [proba_dict[c] * 100 for c in CLASS_ORDER if c in proba_dict]
    colors = [CLASS_COLORS[c] for c in CLASS_ORDER if c in proba_dict]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        height=260, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Probabilitas (%)", range=[0, 100]),
        yaxis=dict(title=""),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_risk_badge(label):
    color = CLASS_COLORS.get(label, "#888")
    text = CLASS_LABELS_ID.get(label, label)
    st.markdown(
        f'<div class="risk-badge" style="background-color:{color};">{text}</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown(f"_{CLASS_DESC.get(label, '')}_")


def validate_input(values: dict):
    errors = []
    if values.get("years_at_company", 0) > values.get("years_experience", 0):
        errors.append("Lama bekerja di perusahaan saat ini tidak boleh melebihi total pengalaman kerja.")
    if values.get("sleep_hours_per_night", 0) > 24:
        errors.append("Jam tidur per malam tidak valid (maksimal 24 jam).")
    if values.get("exercise_days_per_week", 0) > 7:
        errors.append("Hari olahraga per minggu tidak valid (maksimal 7 hari).")
    if values.get("age", 0) < 18:
        errors.append("Usia minimal 18 tahun.")
    return errors


# HALAMAN: BERANDA
def page_beranda():
    st.markdown('<div class="main-header">🔥 Tech Risk Burnout Web</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Deteksi dini risiko burnout berbasis Machine Learning '
        '(XGBoost & Random Forest)</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["ℹ️ Informasi Aplikasi", "🎯 Tujuan Sistem", "📖 Petunjuk Penggunaan"])

    with tab1:
        st.markdown("""
        <div class="card">
        Aplikasi ini membantu memprediksi <b>tingkat risiko burnout</b> seseorang berdasarkan
        27 variabel yang mencakup <b>demografi</b>, <b>konteks kerja</b>, <b>gaya hidup</b>,
        dan <b>budaya tempat kerja</b>. Prediksi dihasilkan oleh dua model machine learning
        yang bisa dipilih secara interaktif: <b>XGBoost</b> dan <b>Random Forest</b>.
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("""
        <div class="card">
        <ul>
            <li>Membantu individu maupun organisasi mengenali risiko burnout secara dini.</li>
            <li>Memberikan gambaran probabilitas risiko pada 4 kategori: Rendah, Sedang, Tinggi, dan Sangat Tinggi.</li>
            <li>Menjadi alat bantu skrining awal bukan pengganti diagnosis profesional kesehatan mental.</li>
            <li>Menyediakan riwayat prediksi agar tren risiko dapat dipantau dari waktu ke waktu.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("""
        <div class="card">
        <ol>
            <li>Buka menu <b>Klasifikasi Risiko</b> di sidebar.</li>
            <li>Isi seluruh 27 variabel pada form yang terbagi dalam 4 kategori.</li>
            <li>Sistem akan memvalidasi kelengkapan & kewajaran input.</li>
            <li>Pilih model prediksi: <b>XGBoost</b> atau <b>Random Forest</b>.</li>
            <li>Klik tombol <b>Jalankan Klasifikasi</b> untuk melihat hasil kategori risiko beserta probabilitasnya.</li>
            <li>Hasil otomatis tersimpan di menu <b>Riwayat Prediksi</b>, dan dapat diekspor ke CSV.</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

    st.info("💡 Lihat menu **Informasi Kelompok Variabel** untuk penjelasan tiap variabel sebelum mengisi form.")


# HALAMAN: KLASIFIKASI RISIKO
def page_klasifikasi(models, load_errors):
    st.markdown('<div class="main-header">🧮 Klasifikasi Risiko Burnout</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Isi seluruh variabel di bawah untuk mendapatkan hasil klasifikasi.</div>',
                unsafe_allow_html=True)
    
    if load_errors:
        for name, error in load_errors.items():
            st.error(
                f"**{name}: Gagal memuat model**"
            )
            
            st.code(
                f"Jenis Error : {error['type']}\n"
                f"Pesan       : {error['message']}\n\n"
                f"Traceback:\n{error['traceback']}",
                language="text"
            )
            
        if not models:
            st.stop()
            
    st.markdown('<div class="step-header">Input Semua Variabel</div>', unsafe_allow_html=True)

    values = {}
    tabs = st.tabs([f"👤 {k}" if k == "Demografi" else
                    f"💼 {k}" if k == "Konteks Kerja" else
                    f"🧘 {k}" if k == "Gaya Hidup" else
                    f"🏢 {k}" for k in VARIABLE_CONFIG.keys()])

    for tab, (category, fields) in zip(tabs, VARIABLE_CONFIG.items()):
        with tab:
            cols = st.columns(2)
            for i, field in enumerate(fields):
                col = cols[i % 2]
                with col:
                    key = f"input_{field['name']}"
                    if field["type"] == "number":
                        values[field["name"]] = st.number_input(
                            field["label"], min_value=field["min"], max_value=field["max"],
                            value=field["default"], step=field["step"], key=key,
                        )
                    elif field["type"] == "float":
                        values[field["name"]] = st.number_input(
                            field["label"], min_value=field["min"], max_value=field["max"],
                            value=field["default"], step=field["step"], key=key, format="%.1f",
                        )
                    elif field["type"] == "slider":
                        values[field["name"]] = st.slider(
                            field["label"], min_value=field["min"], max_value=field["max"],
                            value=field["default"], key=key,
                        )
                    elif field["type"] == "select":
                        display_options = [o[0] for o in field["options"]]
                        choice = st.selectbox(field["label"], display_options, key=key)
                        mapped_value = dict(field["options"])[choice]
                        values[field["name"]] = mapped_value

    st.markdown('<div class="step-header">Pilih Model</div>', unsafe_allow_html=True)
    available_models = list(models.keys())
    if not available_models:
        st.warning("Belum ada model yang berhasil dimuat.")
        return
    model_choice = st.radio("Model klasifikasi:", available_models, horizontal=True)

    st.markdown('<div class="step-header">Jalankan Klasifikasi</div>', unsafe_allow_html=True)
    run = st.button("🚀 Jalankan Klasifikasi", type="primary", use_container_width=True)

    if run:
        errors = validate_input(values)
        if errors:
            st.error("**Input belum valid:**\n\n" + "\n".join(f"- {e}" for e in errors))
            return

        # Transformasi input -> DataFrame sesuai urutan fitur training
        try:
            input_df = pd.DataFrame([values])[FEATURE_ORDER]
        except KeyError as e:
            st.error(f"Kolom input tidak lengkap/tidak cocok dengan FEATURE_ORDER: {e}")
            return

        with st.spinner(f"Menjalankan model {model_choice}..."):
            try:
                model = models[model_choice]
                label, proba_dict = predict_risk(model, input_df)
            except Exception as e:  # noqa: BLE001
                st.error(
                    f"Gagal menjalankan prediksi: {e}\n\n"
                    f"Kemungkinan penyebab: nama/format kategori pada input tidak sama persis "
                    f"dengan yang dipakai saat training (mis. encoder tidak mengenali kategori baru). "
                    f"Cek bagian **Debug: data terkirim ke model** di bawah dan sesuaikan `VARIABLE_CONFIG` di app.py."
                )
                with st.expander("🔍 Debug: data yang terkirim ke model"):
                    st.dataframe(input_df)
                return

        st.session_state.last_result = (label, proba_dict, model_choice, values)
        add_to_history(model_choice, values, label, proba_dict)

    if st.session_state.last_result:
        label, proba_dict, model_choice, values = st.session_state.last_result
        st.divider()
        st.markdown('<div class="step-header">Hasil Kategori dan Probabilitas</div>', unsafe_allow_html=True)
        res_col1, res_col2 = st.columns([1, 1.4])
        with res_col1:
            st.caption(f"Model digunakan: **{model_choice}**")
            render_risk_badge(label)
        with res_col2:
            render_probability_chart(proba_dict)

        with st.expander("🔍 Lihat data input yang dikirim ke model"):
            st.dataframe(pd.DataFrame([values])[FEATURE_ORDER])



# HALAMAN: INFORMASI KELOMPOK VARIABEL
def page_informasi_variabel():
    st.markdown('<div class="main-header">📚 Informasi Kelompok Variabel</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Penjelasan 27 variabel yang digunakan model, dikelompokkan menjadi 4 kategori.</div>',
        unsafe_allow_html=True,
    )

    icons = {"Demografi": "👤", "Konteks Kerja": "💼", "Gaya Hidup": "🧘", "Budaya Tempat Kerja": "🏢"}
    tabs = st.tabs([f"{icons[k]} {k}" for k in VARIABLE_CONFIG.keys()])

    for tab, (category, fields) in zip(tabs, VARIABLE_CONFIG.items()):
        with tab:
            for field in fields:
                desc = VARIABLE_INFO_DESC.get(field["name"], "")
                type_label = {"number": "Numerik", "float": "Numerik (desimal)",
                              "slider": "Skala 1-10", "select": "Kategorikal"}[field["type"]]
                st.markdown(f"""
                <div class="card">
                    <b>{field['label']}</b> <span style="color:#888;">({field['name']})</span><br>
                    <span style="color:#666; font-size:0.85rem;">Tipe: {type_label}</span>
                    <p style="margin-top:0.4rem;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)



# HALAMAN: RIWAYAT PREDIKSI
def page_riwayat():
    st.markdown('<div class="main-header">🕘 Riwayat Prediksi</div>', unsafe_allow_html=True)

    riwayat = st.session_state.riwayat
    if not riwayat:
        st.info("Belum ada riwayat prediksi. Jalankan klasifikasi terlebih dahulu di menu **Klasifikasi Risiko**.")
        return

    st.markdown('<div class="step-header">Lihat Riwayat</div>', unsafe_allow_html=True)
    table_rows = [{
        "Waktu": r["waktu"],
        "Model": r["model"],
        "Hasil": r["hasil_id"],
        "Usia": r["input"].get("age"),
        "Peran": r["input"].get("job_role"),
        "Stres Score": r["input"].get("stress_score"),
    } for r in riwayat]
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    st.markdown('<div class="step-header">Eksport Hasil Riwayat</div>', unsafe_allow_html=True)
    export_df = pd.json_normalize(riwayat, sep="_")
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Unduh Riwayat (CSV)", data=csv_bytes,
        file_name=f"riwayat_prediksi_burnout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="step-header">Hapus Salah Satu Riwayat</div>', unsafe_allow_html=True)
        options = {f"{r['waktu']} — {r['model']} — {r['hasil_id']}": r["id"] for r in riwayat}
        selected = st.selectbox("Pilih entri riwayat:", list(options.keys()), key="del_one_select")
        if st.button("🗑️ Hapus Entri Ini", use_container_width=True):
            target_id = options[selected]
            st.session_state.riwayat = [r for r in st.session_state.riwayat if r["id"] != target_id]
            save_history(st.session_state.riwayat)
            st.session_state.last_result = None
            st.success("Entri riwayat berhasil dihapus.")
            st.rerun()

    with col2:
        st.markdown('<div class="step-header">Hapus Semua Riwayat</div>', unsafe_allow_html=True)
        confirm = st.checkbox("Saya yakin ingin menghapus seluruh riwayat.")
        if st.button("🗑️ Hapus Semua Riwayat", type="primary", use_container_width=True, disabled=not confirm):
            st.session_state.riwayat = []
            save_history([])
            st.session_state.last_result = None
            st.success("Seluruh riwayat berhasil dihapus.")
            st.rerun()


# MAIN / NAVIGASI


def main():
    init_session_state()

    with st.sidebar:
        st.markdown("## 🔥 Tech Risk Burnout Web")
        st.caption("Menu Aplikasi")
        page = st.radio(
            "Menu",
            ["🏠 Beranda", "🧮 Klasifikasi Risiko", "📚 Informasi Kelompok Variabel", "🕘 Riwayat Prediksi"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption(f"Total riwayat tersimpan: **{len(st.session_state.riwayat)}**")

    models, load_errors = load_models()

    if page == "🏠 Beranda":
        page_beranda()
    elif page == "🧮 Klasifikasi Risiko":
        page_klasifikasi(models, load_errors)
    elif page == "📚 Informasi Kelompok Variabel":
        page_informasi_variabel()
    elif page == "🕘 Riwayat Prediksi":
        page_riwayat()


if __name__ == "__main__":
    main()
