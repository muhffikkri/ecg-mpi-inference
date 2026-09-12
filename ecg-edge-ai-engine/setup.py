from setuptools import setup
import os

# Read requirements from requirements.txt
requirements = []
if os.path.exists("requirements.txt"):
    with open("requirements.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Skip tflite-runtime on Windows (NT) to avoid install failure,
            # since Windows users use tensorflow for local testing.
            if "tflite-runtime" in line and os.name == "nt":
                continue
            requirements.append(line)

setup(
    name="ecg_edge_ai_engine",
    version="1.0.0",
    description="A modular black-box 3-lead 250 Hz ECG inference engine for Raspberry Pi",
    author="PKM ECGRhythmia",
    packages=["ecg_edge_ai_engine"],
    package_dir={"ecg_edge_ai_engine": "src"},
    install_requires=requirements,
    python_requires=">=3.11",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
)
