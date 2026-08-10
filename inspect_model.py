"""
Script bantu untuk memverifikasi isi pipeline model sebelum dipakai di app.py.
Jalankan di environment lokal yang SUDAH terpasang semua dependency
(lihat requirements.txt) — terutama xgboost & imbalanced-learn.

Cara pakai:
    python inspect_model.py
"""

import os
import joblib
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

MODEL_PATHS = {
    "XGBoost": os.path.join(MODEL_DIR, "xgboost_burnout_pipeline.joblib"),
    "Random Forest": os.path.join(MODEL_DIR, "random_forest_burnout_pipeline.joblib"),
}


def inspect(name, path):
    print("=" * 70)
    print(f"MODEL: {name}  ({path})")
    print("=" * 70)
    if not os.path.exists(path):
        print("  File tidak ditemukan.")
        return

    model = joblib.load(path)
    print(f"  Tipe objek        : {type(model)}")

    if hasattr(model, "steps"):
        print("  Langkah pipeline  :")
        for step_name, step_obj in model.steps:
            print(f"    - {step_name}: {type(step_obj)}")

    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "steps"):
        classes = getattr(model.steps[-1][1], "classes_", None)
    print(f"  classes_ (urutan kelas output) : {classes}")

    # Coba lihat nama fitur yang diharapkan (kalau tersedia dari ColumnTransformer)
    try:
        for step_name, step_obj in model.steps:
            if hasattr(step_obj, "feature_names_in_"):
                print(f"  feature_names_in_ pada step '{step_name}': {list(step_obj.feature_names_in_)}")
            if hasattr(step_obj, "transformers_"):
                for tname, _, cols in step_obj.transformers_:
                    print(f"    ColumnTransformer '{tname}' -> kolom: {cols}")
    except Exception as e:  # noqa: BLE001
        print(f"  (Tidak bisa membaca detail preprocessing: {e})")

    print()


if __name__ == "__main__":
    for name, path in MODEL_PATHS.items():
        inspect(name, path)

    print("Selesai. Bandingkan 'feature_names_in_' / kolom ColumnTransformer di atas")
    print("dengan FEATURE_ORDER dan opsi kategori (VARIABLE_CONFIG) di app.py.")
    print("Jika ada perbedaan nama kategori (mis. 'Male' vs 'male'), sesuaikan app.py.")
