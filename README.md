# Clasificación Temporal de Texto

Notebooks para clasificar textos históricos por década usando transformers.

## Notebooks

| Notebook | Modelo | Descripción |
|----------|--------|-------------|
| `fastBaseline.ipynb` | DistilBERT | Baseline rápido |
| `bert_classifier.ipynb` | BERT | Migración a BERT |
| `competition_notebook.ipynb` | DeBERTa-v3 | Notebook competitivo con K-Fold, ensemble y features temporales |

## Instalación

### 1. Crear entorno virtual

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Instalar dependencias base

```bash
pip install -r requirements.txt
```

### 3. Instalar PyTorch con soporte CUDA (solo si hay GPU NVIDIA)

Verificar que CUDA esté disponible:

```bash
nvidia-smi
```

Luego instalar PyTorch con CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Verificar GPU

Ejecutar en Python:

```python
import torch
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

Si `torch.cuda.is_available()` es `False`, PyTorch usará CPU.

## Datos

Los archivos `train.csv` y `eval.csv` deben estar en `../data/`.
