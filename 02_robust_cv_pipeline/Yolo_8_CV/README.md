# Yolo_8_CV — Clasificación de Alzheimer con YOLOv8n-cls y validación cruzada 5-fold

Pipeline formal de entrenamiento y evaluación para la clasificación de IRM cerebrales
en cuatro estadios de Alzheimer (`NonDemented`, `VeryMildDemented`, `MildDemented`,
`ModerateDemented`).

> ⚠️ Los resultados son orientativos y **no constituyen diagnóstico médico**.

---

## 1. Por qué existe esta carpeta

`Yolo_8/modelo_YOLO_8.ipynb` es el experimento preliminar: entrena con
`AugmentedAlzheimerDataset` y evalúa con `OriginalDataset`.

Ese esquema tiene **fuga de datos**: `AugmentedAlzheimerDataset` contiene los 6,400
originales redimensionados a 180×180 **y** variantes aumentadas de esos mismos
originales. El modelo ve cada imagen de test durante el entrenamiento y la accuracy
sale artificialmente alta.

`Yolo_8_CV` corrige el protocolo:

- se parte `OriginalDataset` **una sola vez** en 5 folds;
- el aumento se genera **después del split** y **solo sobre los folds de train**;
- el fold de test se mantiene original, sin aumentar, y se evalúa una sola vez;
- las 6,400 imágenes se testean exactamente una vez (predicciones *out-of-fold*).

---

## 2. Protocolo

| Elemento | Valor |
|---|---|
| Dataset fuente | `OriginalDataset` — 6,400 imágenes |
| Clases | NonDemented 3,200 · VeryMildDemented 2,240 · MildDemented 896 · ModerateDemented 64 |
| Validación | `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` |
| Agrupación | cortes del mismo volumen (`NN (v)`) no se separan entre folds |
| Split interno | 10 % de los 4 folds de train → validación (original, sin aumentar) |
| Aumento | solo train, offline, determinista (`SEED=42` + hash de ruta) |
| Factores | Non 2× · VeryMild 3× · Mild 5× · Moderate 15× |
| Modelo | `yolov8n-cls.pt`, fine-tuning completo |
| Hiperparámetros | `imgsz=224`, `batch=16`, `epochs=20`, `optimizer=AdamW`, `lr0=1e-3`, `cos_lr=True` |

### Distribución por fold

| Fold | Train (original) | Val (original) | Test (original) |
|---|---:|---:|---:|
| 1–5 | 4,608 | 512 | 1,280 |

Test por fold: Non 640 · VeryMild 448 · Mild 179 · Moderate 13.

### Aumento por fold (desde 4,608 originales de train)

| Clase | Originales | Aumentadas | Total en `train/` |
|---|---:|---:|---:|
| NonDemented | 2,304 | 4,608 (2×) | 6,912 |
| VeryMildDemented | 1,612 | 4,836 (3×) | 6,448 |
| MildDemented | 646 | 3,230 (5×) | 3,876 |
| ModerateDemented | 46 | 690 (15×) | 736 |
| **Total** | **4,608** | **13,364** | **17,972** |

Las cifras varían ±1 imagen entre folds por el redondeo del split estratificado.

### Preprocesamiento (idéntico en train/val/test)

1. Lectura en escala de grises.
2. Recorte del cráneo (`> 10`) con margen del 5 %.
3. Redimensionado preservando aspecto y *padding* negro a 224×224.
4. Normalización robusta por percentiles 1–99.

### Transformaciones de aumento (solo train)

`Affine` (±10°, ±5 %, escala 0.95–1.05, shear ±5), brillo/contraste ±10 %,
gamma 90–110, ruido gaussiano, desenfoque gaussiano y `ElasticTransform` suave.
Sin tintes de color, sin *flips* verticales, sin *mixup/cutmix*.

**Aumento en línea desactivado.** Ultralytics aplica por defecto, además, su propio
aumento sobre el split de train (`RandomResizedCrop`, volteo horizontal, `RandAugment`,
`RandomErasing` y jitter HSV). Eso se sumaría al aumento offline y haría irrepetible
el protocolo, así que el notebook fuerza los transforms deterministas
(*resize* + *center crop* + *normalize*) en los tres splits y además declara
`scale=0`, `fliplr=0`, `flipud=0`, `erasing=0`, `auto_augment=None` y `hsv_*=0`
en la llamada de entrenamiento. **El único aumento es el offline documentado.**

---

## 3. Estructura

```
Yolo_8_CV/
├── modelo_YOLO_8_CV.ipynb          # pipeline completo (19 celdas de código)
├── README.md
├── .gitignore
├── weights/
│   ├── fold_1..5/
│   │   ├── best.pt                 # modelos de la CV (solo métricas)
│   │   └── last.pt
│   └── final/
│       ├── best.pt                 # modelo de despliegue
│       ├── last.pt                 # checkpoint intermedio (no versionado)
│       └── best.onnx               # artefacto para FastAPI
├── reports/
│   ├── cv_summary.json             # media ± std de los 5 folds
│   ├── fold_metrics.csv
│   ├── oof_predictions.csv         # 6,400 predicciones out-of-fold
│   ├── classification_report.txt
│   ├── confusion_matrix_oof.png    # absoluta + normalizada
│   ├── roc_curves_oof.png
│   ├── pr_curves_oof.png
│   ├── training_curves/fold_1..5.png
│   └── gradcam/ejemplo_inferencia.png
└── splits/
    ├── folds.json                  # split reproducible
    └── manifest_fold_1..5.csv      # mapeo aumentada → original
```

Las imágenes aumentadas **no se versionan**: se generan en
`/content/work_yolo8_cv/generated/` durante la ejecución y se borran al terminar
cada fold.

---

## 4. Cómo ejecutar

1. Subir `data_Alzheimer.zip` a Google Drive en
   `/MyDrive/Tallermovil/Dataset/data_Alzheimer.zip`.
2. Abrir `modelo_YOLO_8_CV.ipynb` en Google Colab con GPU T4.
3. Ejecutar todas las celdas en orden.

El notebook es **reanudable**: al empezar, cada fold se restaura desde Drive
(`weights/fold_k/`, `reports/fold_k_probs.npz`, `manifest_fold_k.csv`) y, si ya
tiene `best.pt` y sus probabilidades, se omite. Así una desconexión de Colab no
obliga a repetir los folds ya entrenados.

Versiones de referencia (probadas): `ultralytics 8.4.191`, `albumentations 2.0.8`.
El notebook instala `ultralytics>=8.4,<9` y `albumentations>=2.0,<3`.

### Tiempo estimado (Colab T4)

| Etapa | Tiempo |
|---|---:|
| Generación por fold (~13.4k aumentadas) | 2–3 min |
| Entrenamiento por fold (20 épocas) | 12–18 min |
| **5 folds** | **75–110 min** |
| Modelo final (~25k imágenes) | 15–22 min |
| Exportación ONNX | < 1 min |

---

## 5. Artefactos para la app

| Archivo | Uso |
|---|---|
| `weights/final/best.pt` | Inferencia con Ultralytics (PyTorch) |
| `weights/final/best.onnx` | Inferencia en FastAPI con ONNX Runtime |

El preprocesamiento de la API debe replicar exactamente
`preprocess_image()` (gris → recorte → 224×224 con padding → percentiles 1–99)
y la salida debe exponerse como:

```json
{
  "predicted_class": "MildDemented",
  "confidence": 0.9134,
  "scores": {
    "MildDemented": 0.9134,
    "ModerateDemented": 0.0210,
    "NonDemented": 0.0502,
    "VeryMildDemented": 0.0154
  }
}
```

---

## 6. Reproducibilidad

- `SEED = 42` en Python, NumPy y PyTorch.
- `deterministic=True` en Ultralytics.
- Cada variante aumentada usa una semilla derivada de
  `sha256(SEED | ruta_original | índice_variante)`, por lo que la regeneración es
  idéntica byte a byte.
- `splits/folds.json` y `splits/manifest_fold_k.csv` permiten auditar el split y
  demostrar que ninguna imagen de test tiene una versión aumentada en train.

---

## 7. Limitaciones

- `ModerateDemented` solo aporta 13 imágenes al test de cada fold (64 en total OOF).
  Las métricas de esa clase tienen intervalos de confianza amplios.
- El modelo final se entrena con el 100 % de los originales aumentados; su
  rendimiento estimado es el de la CV, no existe un test adicional reservado.
- El `val` del modelo final es un subconjunto del train, usado solo para seleccionar
  el checkpoint; no se reporta como métrica.
