# Streamlit-дашборд в Google Colab через LocalTunnel (пошаговый гайд) + как связать с шаблоном ML

Ниже — полностью рабочая схема: **Colab → Streamlit (на фоне) → LocalTunnel → публичная ссылка**, плюс пример, как подключить твой **шаблон предобработки/обучения/Optuna** так, чтобы дашбордом было удобно пользоваться.

---

## 0) Что важно понимать про Streamlit (самое главное)

- Streamlit — это **веб-приложение на Python**, где интерфейс (кнопки, формы, графики) пишется прямо кодом.
- **Скрипт `app.py` выполняется сверху вниз при каждом изменении** (клик, ввод, переключатель).
- Поэтому всё тяжёлое (загрузка данных, обучение, optuna) надо делать:
  - либо **по кнопке**,
  - либо **кэшировать** (`st.cache_data`, `st.cache_resource`),
  - либо хранить результат в `st.session_state`.

---

## 1) Установка в Colab (Streamlit + LocalTunnel)

В Colab запускай в отдельных ячейках:

```python
!pip install -q streamlit optuna scikit-learn pandas numpy matplotlib seaborn joblib
!npm -q install localtunnel
```

> `npm` в Colab обычно уже есть. Если вдруг нет — установи Node.js или напиши мне, дам команды.

---

## 2) Создание приложения `app.py`

В Colab сделай ячейку:

```python
%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="ML Dashboard", layout="wide")

st.title("ML Dashboard (Colab + LocalTunnel)")
st.write("Проверь, что всё запускается. Затем подключим обучение/предсказания.")

uploaded = st.file_uploader("Загрузить CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.subheader("Первые строки")
    st.dataframe(df.head(20))

    st.subheader("Размерность")
    st.write(df.shape)

    st.subheader("Describe")
    st.dataframe(df.describe(include="all").transpose())
else:
    st.info("Загрузи CSV, чтобы увидеть таблицу.")
```

---

## 3) Запуск Streamlit в фоне

Streamlit должен крутиться **в фоне**, а Colab-ячейка не должна “висеть” вечно.

```python
!streamlit run app.py --server.port 8501 --server.address 0.0.0.0 \
  --server.enableCORS false --server.enableXsrfProtection false \
  > /content/streamlit_logs.txt 2>&1 &
```

Проверить логи, если что-то не работает:

```python
!tail -n 50 /content/streamlit_logs.txt
```

---

## 4) Запуск LocalTunnel и получение публичной ссылки

### 4.1) Узнать публичный IP (как на подсказке)

```python
!wget -qO- ipv4.icanhazip.com
```

Скопируй IP (например `12.34.56.78`).

### 4.2) Поднять туннель на порт 8501

```python
!npx localtunnel --port 8501
```

В выводе появится ссылка вида `https://xxxx.loca.lt` — открой её в браузере.

### 4.3) Если страница просит IP

Иногда LocalTunnel показывает страницу “Enter your IP” — вставь **тот IP**, который получил на шаге 4.1.

---

## 5) Типовые проблемы и быстрые решения

### “Ссылка открылась, но приложение не грузится”
1) Проверь, что Streamlit жив:
```python
!lsof -i :8501 | head
```

2) Проверь логи:
```python
!tail -n 100 /content/streamlit_logs.txt
```

### “Команда localtunnel висит”
Это нормально: туннель держит соединение, пока ты не остановишь ячейку.

### “Ничего не меняется после кликов”
Streamlit **перезапускает скрипт** на каждый интерактив. Если у тебя тяжёлая часть без кэша — будет ощущение “залипания”.

---

# 6) Как связать Streamlit с шаблоном обучения/обработки моделей

Идея правильной архитектуры:

- **`ml_pipeline.py`** — всё про данные/препроцессинг/обучение/optuna/метрики (твои шаблоны).
- **`app.py`** — только UI: загрузка файла, выбор цели/признаков, кнопки “обучить”, “оценить”, “предсказать”.

В Colab это удобно делать через `%%writefile` двумя файлами.

---

## 6.1) Мини-модуль `ml_pipeline.py` (универсальный, под любой датасет)

```python
%%writefile ml_pipeline.py
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

import optuna


def make_preprocessor(X: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    num_pipe = Pipeline(steps=num_steps)

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


def build_model(model_name: str, params: dict, seed: int = 42):
    if model_name == "ridge":
        return Ridge(random_state=seed, **params)
    if model_name == "elasticnet":
        return ElasticNet(random_state=seed, max_iter=20000, **params)
    if model_name == "random_forest":
        return RandomForestRegressor(random_state=seed, n_jobs=-1, **params)
    if model_name == "hist_gb":
        return HistGradientBoostingRegressor(random_state=seed, **params)
    raise ValueError(f"Unknown model_name: {model_name}")


def suggest_params(trial: optuna.Trial, model_name: str) -> dict:
    if model_name == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True)}

    if model_name == "elasticnet":
        return {
            "alpha": trial.suggest_float("alpha", 1e-5, 10.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        }

    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "max_depth": trial.suggest_int("max_depth", 3, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 40),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }

    if model_name == "hist_gb":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 255),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 60),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 1e-2, log=True),
        }

    raise ValueError(f"Unknown model_name: {model_name}")


def cv_rmse(model, X, y, n_splits=5, seed=42) -> float:
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
    return float(-scores.mean())


def train_with_optuna(
    df: pd.DataFrame,
    target: str,
    model_name: str,
    scale_numeric: bool,
    test_size: float = 0.25,
    seed: int = 42,
    n_trials: int = 30,
    n_splits: int = 5,
):
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    pre = make_preprocessor(X_train, scale_numeric=scale_numeric)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, model_name)
        base_model = build_model(model_name, params, seed=seed)
        pipe = Pipeline(steps=[("pre", pre), ("model", base_model)])
        return cv_rmse(pipe, X_train, y_train, n_splits=n_splits, seed=seed)

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_model = build_model(model_name, best_params, seed=seed)
    best_pipe = Pipeline(steps=[("pre", pre), ("model", best_model)])
    best_pipe.fit(X_train, y_train)

    pred_tr = best_pipe.predict(X_train)
    pred_te = best_pipe.predict(X_test)

    metrics_train = compute_metrics(y_train, pred_tr)
    metrics_test = compute_metrics(y_test, pred_te)

    return {
        "study": study,
        "model": best_pipe,
        "best_params": best_params,
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
        "splits": (X_train, X_test, y_train, y_test),
    }
```

---

## 6.2) Streamlit-дашборд, который “кликается” и обучает модель по кнопке

```python
%%writefile app.py
import streamlit as st
import pandas as pd
import joblib

from ml_pipeline import train_with_optuna

st.set_page_config(page_title="ML Dashboard", layout="wide")
st.title("ML Dashboard (Colab + LocalTunnel)")

@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

uploaded = st.file_uploader("Загрузить CSV", type=["csv"])

if uploaded is None:
    st.info("Загрузи CSV, затем выбери target и модель.")
    st.stop()

df = load_csv(uploaded)
st.write("Размерность:", df.shape)
st.dataframe(df.head(15))

target = st.selectbox("Целевая колонка (target)", options=df.columns.tolist())

model_name = st.selectbox(
    "Модель",
    options=["ridge", "elasticnet", "random_forest", "hist_gb"]
)

scale_numeric = st.checkbox("Скалировать числовые признаки (актуально для линейных)", value=True)

col1, col2, col3 = st.columns(3)
with col1:
    n_trials = st.slider("Optuna trials", 10, 200, 30, step=10)
with col2:
    n_splits = st.slider("CV splits", 3, 10, 5)
with col3:
    test_size = st.slider("Test size", 0.1, 0.4, 0.25, step=0.05)

if "trained" not in st.session_state:
    st.session_state.trained = None

if st.button("Обучить (Optuna + CV)"):
    with st.spinner("Идёт подбор гиперпараметров и обучение..."):
        out = train_with_optuna(
            df=df,
            target=target,
            model_name=model_name,
            scale_numeric=scale_numeric,
            test_size=test_size,
            n_trials=n_trials,
            n_splits=n_splits,
        )
    st.session_state.trained = out
    st.success("Готово!")

trained = st.session_state.trained
if trained is None:
    st.stop()

st.subheader("Лучшие параметры")
st.json(trained["best_params"])

st.subheader("Метрики")
mtr = trained["metrics_train"]
mte = trained["metrics_test"]

c1, c2 = st.columns(2)
with c1:
    st.write("Train")
    st.write(mtr)
with c2:
    st.write("Test")
    st.write(mte)

st.subheader("Сохранение модели")
if st.button("Сохранить модель в model.joblib"):
    joblib.dump(trained["model"], "model.joblib")
    st.success("Сохранено: model.joblib (в файловой системе Colab)")
```

---

## 7) Как “работать” со Streamlit (минимальный workflow)

1) Открыл ссылку `https://xxxx.loca.lt`  
2) Загрузил CSV  
3) Выбрал `target`  
4) Выбрал модель  
5) Нажал **“Обучить”**  
6) Получил метрики + параметры  
7) (опционально) сохранил модель

> Если ты кликаешь любой элемент — Streamlit перезапускает `app.py`.  
> Поэтому результаты храним в `st.session_state`.

---

## 8) Как подключить твой “большой шаблон по моделям” к этому дашборду

Твой файл с шаблонами — это “база”. Чтобы использовать его в дашборде:

- Берёшь из него:
  - `make_preprocessor`
  - `compute_metrics`, `rmse`
  - `suggest_params` для моделей
  - `tune_model_optuna` / CV-оценку
  - построение grid-таблиц и графиков
- Складываешь в `ml_pipeline.py`, а UI остаётся в `app.py`

То есть:
- **шаблон = библиотека функций**
- **streamlit = оболочка кнопок и вывода**

---

## 9) Важные советы (чтобы не мучиться)

- **Не запускай Optuna на каждом изменении**. Только по кнопке.
- Для тяжёлых вычислений:
  - `st.cache_data` — для данных/таблиц
  - `st.cache_resource` — для моделей/обучения (если нужно)
- Ограничивай `n_trials` (например 30–80), иначе Colab может “устать”.
- Храни результаты в `st.session_state`.

---

Если захочешь расширить дашборд, логичный следующий шаг:
- выбор нескольких версий данных (no_scaling / standard / minmax / robust),
- обучение **всего пула моделей** одной кнопкой,
- вывод **сводной таблицы (grid)** и **grid-графиков** прямо в Streamlit.
