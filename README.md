# ECG MPI Inference

**MPI-Based Parallel Processing for Large-Scale ECG Arrhythmia Classification**

Implementasi **data-parallel ECG inference** menggunakan **Python + `mpi4py`**, dataset
**Chapman ECG** (45.152 rekaman), dan engine inferensi **ECG Edge AI Engine** (1D-CNN
TFLite, 3-lead). Tujuan proyek ini adalah menerapkan pemrograman paralel MPI untuk membagi
beban kerja klasifikasi ECG ke banyak process sehingga hasilnya diproses lebih cepat namun
**identik** dengan inferensi sekuensial.

## Ikhtisar

- `sequential.py` — baseline sekuensial (satu process, satu instans engine).
- `mpi_inference.py` — inferensi paralel MPI (`bcast` → `split_workload` manual → inferensi
  lokal per rank → `gather` → sort → validasi). **Tidak** menggunakan `Scatter`/`Scatterv`.
- `ecg_common.py` — helper bersama (adapter engine, akses dataset, pembagian workload,
  penulisan/validasi hasil).
- `ecg-edge-ai-engine/` — ECG Edge AI Engine (source tree; hanya `src/engine.py` yang diberi
  patch kecil pada parsing threshold).

## Hasil Singkat (subset 1.000 rekaman)

| `p` | Waktu (s) | Speedup | Efisiensi |
|----:|----------:|--------:|----------:|
|   1 |     19,96 |    1,00x |    100,0% |
|   2 |     10,47 |    1,91x |     95,3% |
|   4 |      5,19 |    3,85x |     96,2% |
|   8 |      3,03 |    6,59x |     82,4% |

Konsistensi prediksi vs sekuensial: **999/999 = 100%**.

## Menjalankan

```powershell
python -m venv venv; venv\Scripts\activate
pip install -r requirements.txt

python sequential.py --limit 1000
& "$env:ProgramFiles\Microsoft MPI\Bin\mpiexec.exe" -n 4 venv\Scripts\python mpi_inference.py --limit 1000
```

## Laporan Lengkap

Seluruh metodologi, desain, data eksperimen, dan analisis tersedia di
**[`MPI-Report.md`](MPI-Report.md)**.