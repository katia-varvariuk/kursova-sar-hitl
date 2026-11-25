# HITL-розмітка людей на SAR-знімках з активним навчанням

Курсова робота (ЧНУ, «Активне машинне навчання»). Система розмітки шару 1 (людина, поза, видимість)
для датасетів C2A і SARD: гармонізація, цикл АМН, маршрутизація рамок, верифікація в Label Studio,
трекінг у MLflow.

Стек: Python 3.11, Ultralytics YOLO11, Cleanlab, FiftyOne, MLflow, Label Studio, FastAPI, Docker Compose.

## Встановлення

```powershell
python -m venv .venv
.venv\Scripts\pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\pip install -r requirements.txt
copy docker\.env.example docker\.env
docker compose -f docker/docker-compose.yml up -d
```

## Запуск

```powershell
.venv\Scripts\python src\harmonize_c2a.py
.venv\Scripts\python src\harmonize_sard.py
.venv\Scripts\python src\train_baseline.py --epochs 30 --name yolo11n_e30
powershell -ExecutionPolicy Bypass -File scripts\run_overnight.ps1
.venv\Scripts\python src\analyze.py --q1-tag main --q2-tag q2_sard
.venv\Scripts\python src\build_crops.py
.venv\Scripts\python src\validate_manifest.py
```

Оркестратор:

```powershell
.venv\Scripts\python -m uvicorn orchestrator:app --app-dir src --port 8000
```

API: `/init`, `/rounds/select`, `/sync`, `/train`, `/metrics`, `/export`. Документація — http://localhost:8000/docs.

Прогнозний модуль і демо (модель `models/yolo11n_sard.pt` — копія
`data/work/al/q2_sard_random_seed0/runs/r6/weights/last.pt`):

```powershell
.venv\Scripts\python src\predict.py data\processed\sard\images	est\<файл>.jpg
.venv\Scripts\python src\demo_app.py
.venv\Scripts\python srcaselines.py
.venv\Scripts\python src\explain.py
```

Демо — http://127.0.0.1:7860.

## Структура

- `src/` — гармонізація, оракул, експерименти, маршрутизація, оркестратор, маніфест кропів
- `configs/` — схема класів, модель часу анотатора, схема маніфесту
- `labelstudio/` — шаблон верифікації рамок
- `docker/` — Label Studio, MLflow, оркестратор
- `scripts/` — прогін експериментів

## Основні результати

- Аудит розмітки (3 сіди): Cleanlab знаходить 37.5% помилок за 20% перевірок, випадковий порядок — 20.7%.
- АМН C2A → SARD (5 сідів), 350 запитів: recall 0.59 [0.58; 0.60] (випадковий відбір) і 0.53 [0.51; 0.55] (невизначеність), Wilcoxon p = 0.0625.
- Маршрутизація на SARD: 54–59% рамок приймаються автоматично, до людини потрапляє 81–88% помилок.
- Драбина базових рішень на тесті SARD (recall): правило без ML 0, YOLO11n COCO 0.25, після C2A 0.13, після АМН на SARD 0.65.
- SAHI (нарізка 320/256): на C2A recall 0.58 → 0.68–0.69 (дрібні люди), на SARD без приросту, precision падає.
