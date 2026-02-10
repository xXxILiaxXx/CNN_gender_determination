# CNN Gender Classification (Men/Women)

Проект для **бинарной классификации изображений** (`men` / `women`) с использованием **PyTorch**.  
Датасет: Kaggle `playlist/men-women-classification`.

> ⚠️ Важно: модель обучается на датасете с метками `men/women`. В реальном мире интерпретация “пола по фото” может быть сложнее (качество фото, ракурс, возраст, стиль и т.д.). Здесь задача учебная/демонстрационная.

---

## Возможности

- Загрузка и подготовка датасета (Kaggle)
- Обучение CNN-модели (SimpleCNN)
- Сохранение лучшего чекпоинта по `val accuracy`
- Оценка на тестовой выборке (accuracy, classification report, confusion matrix)
- Инференс по одному изображению
- Экспорт модели:
  - **TorchScript (.pt)** — “один файл, без кода”
  - **ONNX (.onnx)** — совместимость с другими runtime

---

## Архитектура модели (SimpleCNN)

Используется компактная CNN-архитектура: **4 блока Conv-BN-ReLU-MaxPool + MLP-голова**.

### Вход
- RGB изображение `3 × 128 × 128` (по умолчанию)
- Предобработка: `Resize(img_size)` → `ToTensor()`

### Feature extractor (сверточная часть)

4 блока вида: `Conv2D → BatchNorm → ReLU → MaxPool(2)`

1) `Conv(3 → 32)` → `MaxPool` : `128×128 → 64×64`  
2) `Conv(32 → 64)` → `MaxPool`: `64×64 → 32×32`  
3) `Conv(64 → 128)` → `MaxPool`: `32×32 → 16×16`  
4) `Conv(128 → 256)` → `MaxPool`: `16×16 → 8×8`

На выходе: `256 × 8 × 8`

### Classifier (MLP-голова)

- `Flatten`: `256 * 8 * 8 = 16384`
- `Linear(16384 → 256)` → ReLU
- Dropout (обычно `0.3` на обучении)
- `Linear(256 → 2)` → logits

### Выход
- logits размерности `[2]`
- вероятности: `softmax(logits)`

---

## Пайплайн данных

### Классы
- `men`
- `women`

### Трансформации

**Train transforms (пример):**
- Resize до `img_size`
- RandomHorizontalFlip(p=0.5)
- ToTensor()

**Eval transforms:**
- Resize до `img_size`
- ToTensor()

### Баланс классов
Если классы несбалансированы, используется **взвешенная CrossEntropyLoss**:
- подсчёт `class_counts`
- вычисление `class_weights`
- передача весов в `CrossEntropyLoss(weight=...)`

---

## Структура проекта

- `src/gender_classifier/` — основная логика (модель, данные, обучение, инференс)
  - `models/` — архитектуры (SimpleCNN)
  - `data/` — dataset/dataloaders/transforms
  - `training/` — train/eval/утилиты
  - `scripts/` — запускаемые скрипты (`train`, `eval`, `predict`, `export`)
- `artifacts/`
  - `models/` — чекпоинты обучения
  - `exports/`
    - `TorchScript/` — экспорт `.pt` + `meta.json`
    - `ONNX/` — экспорт `.onnx` + `meta.json`

---
### Обучение 
python -m src.gender_classifier.scripts.01_train_baseline

### Оценка на тесте 
python -m src.gender_classifier.scripts.02_eval_test

### Предсказание на одном изображении
python -m src.gender_classifier.scripts.03_predict testing/test1.jpg

## Установка

### Виртуальное окружение
```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt



## Установка

1) Создать виртуальное окружение (PyCharm создаст сам) или вручную:
```bash
python -m venv .venv
source .venv/bin/activate