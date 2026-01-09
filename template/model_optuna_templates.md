# Шаблоны обучения моделей + подбор гиперпараметров через Optuna (универсальный конспект)

Этот файл — заготовка “под любой датасет”: вы меняете только **загрузку данных**, список **признаков/цели** и (при необходимости) **препроцессинг**.  
Дальше можно быстро обучить пул моделей, подобрать гиперпараметры через **Optuna + кросс-валидацию**, сравнить результаты в “grid”-таблицах и провести базовую интерпретацию.

---

## Содержание

1. [Быстрый старт](#быстрый-старт)
2. [Общие правила и анти-ошибки](#общие-правила-и-анти-ошибки)
3. [Метрики (регрессия и классификация)](#метрики-регрессия-и-классификация)
4. [Универсальный препроцессинг (ColumnTransformer + Pipeline)](#универсальный-препроцессинг-columntransformer--pipeline)
5. [Пул моделей и кратко “как работает”](#пул-моделей-и-кратко-как-работает)
6. [Шаблон: Optuna + CV + refit лучшей модели](#шаблон-optuna--cv--refit-лучшей-модели)
7. [Шаблоны гиперпараметров (search spaces) для множества моделей](#шаблоны-гиперпараметров-search-spaces-для-множества-моделей)
8. [Сводная таблица результатов (grid)](#сводная-таблица-результатов-grid)
9. [Графики “Predicted vs Actual” и “Residuals” в grid-формате](#графики-predicted-vs-actual-и-residuals-в-grid-формате)
10. [Интерпретация модели: коэффициенты, feature_importances, permutation importance](#интерпретация-модели-коэффициенты-feature_importances-permutation-importance)
11. [Опционально: внешние библиотеки (XGBoost/LightGBM/CatBoost/SHAP)](#опционально-внешние-библиотеки-xgboostlightgbmcatboostshap)

---

## Быстрый старт

### 1) Установка (если нужно)

```bash
pip install -U scikit-learn optuna pandas numpy matplotlib
```

### 2) Минимальный каркас “под любой датасет”

```python
import numpy as np
import pandas as pd

SEED = 42

# 1) Загрузка данных (замени на свой источник)
df = pd.read_csv("your_data.csv")

# 2) Укажи целевой столбец и признаки
TARGET = "target"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# 3) Разбиение
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=SEED
)

# 4) Дальше см. разделы: препроцессинг, Optuna, сравнение моделей
```

---

## Общие правила и анти-ошибки

- **Не делайте fit препроцессинга на test.**  
  Любой `fit` — только на train (в `Pipeline` это гарантируется автоматически).
- **Не удаляйте строки (выбросы) на test** — иначе тест перестанет быть честной проверкой.
- **Индексы X и y должны совпадать**. Если удаляете строки из `X_train`, удаляйте те же индексы из `y_train`.
- **Кросс-валидация** должна строиться только по train, тест используется только в финале.
- **Не допускайте leakage**: признаки не должны включать целевой признак или прямые “производные” от него.

---

## Метрики (регрессия и классификация)

### Регрессия
- **MAE**: средняя абсолютная ошибка  
- **MSE / RMSE**: средняя квадратичная / корень из неё (часто целевая метрика)
- **R²**: доля объяснённой дисперсии (чем ближе к 1, тем лучше)

Чаще всего для подбора гиперпараметров берут **RMSE** (штрафует большие ошибки).

### Классификация (если нужно)
- Accuracy, F1, ROC-AUC, PR-AUC, LogLoss  
- Для дисбаланса: F1 / PR-AUC часто информативнее Accuracy.

---

## Универсальный препроцессинг (ColumnTransformer + Pipeline)

Подходит, когда есть числовые + категориальные признаки, пропуски, нужно кодирование и/или скейлинг.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def make_preprocessor(X: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    num_pipe = Pipeline(steps=num_steps)

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop"
    )
    return pre
```

> Совет: для деревьев и бустинга скейлинг обычно не обязателен. Для линейных, kNN, SVR — чаще нужен.

---

## Пул моделей и кратко “как работает”

Ниже — коротко “по смыслу”, без плюсов/минусов.

### Линейные
- **Ridge**: линейная комбинация признаков + L2-регуляризация коэффициентов.
- **Lasso**: линейная модель + L1-регуляризация (может обнулять часть коэффициентов).
- **ElasticNet**: сочетание L1 и L2.
- **HuberRegressor**: линейная модель с “мягкой” устойчивостью к выбросам в целевой.
- **SGDRegressor**: линейная модель, обучаемая стохастическим градиентом (для больших данных).

### На расстояниях / ядрах
- **KNeighborsRegressor**: прогноз по среднему (или взвешенному) среди k ближайших объектов.
- **SVR**: строит функцию через ядро (RBF/linear/poly), стараясь уложиться в “epsilon-трубку”.

### Деревья и ансамбли
- **DecisionTreeRegressor**: последовательные разбиения признаков на интервалы.
- **RandomForestRegressor**: множество деревьев на бутстрап-выборках → усреднение.
- **ExtraTreesRegressor**: как лес, но с более случайными разбиениями.
- **GradientBoostingRegressor**: деревья добавляются последовательно, исправляя ошибки прошлых.
- **HistGradientBoostingRegressor**: бустинг с бинингом признаков (быстрее на больших данных).
- **AdaBoostRegressor**: последовательное усиление слабых моделей (часто деревья-стампы).

### Нейросети
- **MLPRegressor**: многослойный персептрон (полносвязная сеть) для аппроксимации сложных зависимостей.

---

## Шаблон: Optuna + CV + refit лучшей модели

### 1) Универсальная оценка через CV

```python
import optuna
import numpy as np
from sklearn.model_selection import KFold, cross_val_score

def rmse_cv_score(model, X, y, n_splits=5, seed=42):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # В sklearn RMSE обычно задают как neg_root_mean_squared_error
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
    return float(-scores.mean())
```

### 2) Общий каркас Optuna

```python
def tune_model_optuna(
    build_model_fn,              # функция (trial)->model
    X_train, y_train,
    n_trials=50,
    n_splits=5,
    seed=42,
    direction="minimize"
):
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)

    def objective(trial):
        model = build_model_fn(trial)
        return rmse_cv_score(model, X_train, y_train, n_splits=n_splits, seed=seed)

    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_model = build_model_fn(optuna.trial.FixedTrial(study.best_params))
    best_model.fit(X_train, y_train)

    return study, best_model
```

> Важно: `build_model_fn` должен создавать модель **с фиксированным random_state**, если он у неё есть.

---

## Шаблоны гиперпараметров (search spaces) для множества моделей

Ниже — пример “словаря моделей” + функций, которые:
1) дают модель,
2) задают search space для Optuna.

### Импорт моделей

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
```

### 1) Search spaces (готовые шаблоны)

```python
def build_ridge(trial, seed=42):
    alpha = trial.suggest_float("alpha", 1e-4, 100.0, log=True)
    return Ridge(alpha=alpha, random_state=seed)

def build_lasso(trial, seed=42):
    alpha = trial.suggest_float("alpha", 1e-5, 10.0, log=True)
    return Lasso(alpha=alpha, random_state=seed, max_iter=20000)

def build_elasticnet(trial, seed=42):
    alpha = trial.suggest_float("alpha", 1e-5, 10.0, log=True)
    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
    return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=seed, max_iter=20000)

def build_huber(trial, seed=42):
    epsilon = trial.suggest_float("epsilon", 1.1, 2.0)
    alpha = trial.suggest_float("alpha", 1e-6, 1e-1, log=True)
    return HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=2000)

def build_sgd(trial, seed=42):
    alpha = trial.suggest_float("alpha", 1e-7, 1e-2, log=True)
    penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])
    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0) if penalty == "elasticnet" else 0.15
    return SGDRegressor(
        alpha=alpha, penalty=penalty, l1_ratio=l1_ratio,
        random_state=seed, max_iter=5000, tol=1e-4
    )

def build_knn(trial, seed=42):
    n_neighbors = trial.suggest_int("n_neighbors", 2, 80)
    weights = trial.suggest_categorical("weights", ["uniform", "distance"])
    p = trial.suggest_int("p", 1, 2)  # 1=Manhattan, 2=Euclidean
    return KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, p=p)

def build_svr(trial, seed=42):
    C = trial.suggest_float("C", 1e-2, 1e3, log=True)
    epsilon = trial.suggest_float("epsilon", 1e-4, 1.0, log=True)
    gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
    return SVR(C=C, epsilon=epsilon, gamma=gamma, kernel="rbf")

def build_tree(trial, seed=42):
    max_depth = trial.suggest_int("max_depth", 2, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 50)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 30)
    return DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=seed
    )

def build_rf(trial, seed=42):
    n_estimators = trial.suggest_int("n_estimators", 200, 1200)
    max_depth = trial.suggest_int("max_depth", 3, 40)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 40)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=seed,
        n_jobs=-1
    )

def build_extra_trees(trial, seed=42):
    n_estimators = trial.suggest_int("n_estimators", 200, 1200)
    max_depth = trial.suggest_int("max_depth", 3, 40)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 40)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=seed,
        n_jobs=-1
    )

def build_gb(trial, seed=42):
    n_estimators = trial.suggest_int("n_estimators", 100, 1200)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    max_depth = trial.suggest_int("max_depth", 2, 8)
    subsample = trial.suggest_float("subsample", 0.6, 1.0)
    return GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        random_state=seed
    )

def build_hgb(trial, seed=42):
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    max_depth = trial.suggest_int("max_depth", 2, 12)
    max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 15, 255)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 10, 60)
    l2_regularization = trial.suggest_float("l2_regularization", 1e-8, 1e-2, log=True)
    return HistGradientBoostingRegressor(
        learning_rate=learning_rate,
        max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        random_state=seed
    )

def build_adaboost(trial, seed=42):
    n_estimators = trial.suggest_int("n_estimators", 50, 800)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 1.0, log=True)
    loss = trial.suggest_categorical("loss", ["linear", "square", "exponential"])
    return AdaBoostRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        loss=loss,
        random_state=seed
    )

def build_mlp(trial, seed=42):
    hidden_layer_sizes = trial.suggest_categorical(
        "hidden_layer_sizes",
        [(64,), (128,), (64, 32), (128, 64), (256, 128)]
    )
    alpha = trial.suggest_float("alpha", 1e-6, 1e-2, log=True)
    learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 5e-2, log=True)
    return MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=2000,
        random_state=seed
    )
```

### 2) Единый реестр моделей

```python
MODEL_BUILDERS = {
    "Ridge": build_ridge,
    "Lasso": build_lasso,
    "ElasticNet": build_elasticnet,
    "Huber": build_huber,
    "SGDRegressor": build_sgd,
    "KNN": build_knn,
    "SVR_RBF": build_svr,
    "DecisionTree": build_tree,
    "RandomForest": build_rf,
    "ExtraTrees": build_extra_trees,
    "GradientBoosting": build_gb,
    "HistGradientBoosting": build_hgb,
    "AdaBoost": build_adaboost,
    "MLP": build_mlp,
}
```

---

## Сводная таблица результатов (grid)

Задача: сравнить модели по train/test метрикам на разных версиях данных (например: no_scaling / standard / minmax / robust).

```python
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def compute_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }

def compute_metrics_aligned(y, y_pred, X):
    common = X.index.intersection(y.index)
    if len(common) > 0:
        return compute_metrics(y.loc[common], y_pred[:len(common)])
    n = min(len(y), len(y_pred), len(X))
    return compute_metrics(y.iloc[:n], y_pred[:n])

def make_summary_grid(DATA_VERSIONS, MODEL_POOL, best_models, y_train, y_test):
    rows = []
    for version_name, (Xtr, Xte) in DATA_VERSIONS.items():
        for model_name in MODEL_POOL:
            model = best_models[(version_name, model_name)]
            pred_tr = model.predict(Xtr)
            pred_te = model.predict(Xte)
            m_tr = compute_metrics_aligned(y_train, pred_tr, Xtr)
            m_te = compute_metrics_aligned(y_test,  pred_te, Xte)

            rows.append({
                "data_version": version_name,
                "model": model_name,
                "train_MAE": m_tr["MAE"], "train_RMSE": m_tr["RMSE"], "train_R2": m_tr["R2"],
                "test_MAE":  m_te["MAE"], "test_RMSE":  m_te["RMSE"], "test_R2":  m_te["R2"],
            })

    df = pd.DataFrame(rows)

    long = df.melt(
        id_vars=["data_version", "model"],
        value_vars=["train_MAE", "train_RMSE", "train_R2", "test_MAE", "test_RMSE", "test_R2"],
        var_name="metric",
        value_name="value",
    )

    summary_grid = (
        long.pivot_table(index=["data_version", "metric"], columns="model", values="value")
        .sort_index()
        .round(6)
    )
    return df, summary_grid
```

---

## Графики “Predicted vs Actual” и “Residuals” в grid-формате

```python
import math
import numpy as np
import matplotlib.pyplot as plt

def _align_y_to_X(y, X):
    common = X.index.intersection(y.index)
    if len(common) > 0:
        return y.loc[common]
    return y.iloc[:len(X)]

def plot_pred_vs_actual_grid(DATA_VERSIONS, MODEL_POOL, best_models, y_test, max_cols=3):
    n_plots = len(DATA_VERSIONS) * len(MODEL_POOL)
    cols = min(max_cols, len(MODEL_POOL))
    rows = math.ceil(n_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    axes = np.array(axes).reshape(-1)

    k = 0
    for version_name, (Xtr, Xte) in DATA_VERSIONS.items():
        yte = _align_y_to_X(y_test, Xte)
        for model_name in MODEL_POOL:
            ax = axes[k]
            model = best_models[(version_name, model_name)]
            pred = model.predict(Xte)[:len(yte)]

            ax.scatter(yte, pred, s=10, alpha=0.6)
            mn = min(float(yte.min()), float(pred.min()))
            mx = max(float(yte.max()), float(pred.max()))
            ax.plot([mn, mx], [mn, mx])

            ax.set_title(f"{model_name} | {version_name}")
            ax.set_xlabel("True")
            ax.set_ylabel("Predicted")
            k += 1

    for j in range(k, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_residuals_grid(DATA_VERSIONS, MODEL_POOL, best_models, y_test, max_cols=3):
    n_plots = len(DATA_VERSIONS) * len(MODEL_POOL)
    cols = min(max_cols, len(MODEL_POOL))
    rows = math.ceil(n_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    axes = np.array(axes).reshape(-1)

    k = 0
    for version_name, (Xtr, Xte) in DATA_VERSIONS.items():
        yte = _align_y_to_X(y_test, Xte)
        for model_name in MODEL_POOL:
            ax = axes[k]
            model = best_models[(version_name, model_name)]
            pred = model.predict(Xte)[:len(yte)]
            residuals = yte.values - pred

            ax.scatter(pred, residuals, s=10, alpha=0.6)
            ax.axhline(0)

            ax.set_title(f"{model_name} | {version_name}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Residuals (y_true - y_pred)")
            k += 1

    for j in range(k, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
```

---

## Интерпретация модели: коэффициенты, feature_importances, permutation importance

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

def plot_importance(series: pd.Series, title: str, top_n: int = 15):
    s = series.sort_values(key=lambda x: x.abs(), ascending=False).head(top_n)
    plt.figure(figsize=(10, max(4, 0.35 * len(s))))
    plt.barh(s.index[::-1], s.values[::-1])
    plt.title(title)
    plt.xlabel("Importance")
    plt.show()

def explain_model(model, feature_names, X_ref, y_ref, top_n=15, seed=42):
    # 1) Линейные коэффициенты
    if hasattr(model, "coef_"):
        coefs = pd.Series(model.coef_, index=feature_names)
        plot_importance(coefs, f"Coefficients (|coef|)", top_n=top_n)
        return

    # 2) Деревья / леса
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_names)
        plot_importance(imp, "Feature Importances", top_n=top_n)
        return

    # 3) Общее: permutation importance
    r = permutation_importance(
        model, X_ref, y_ref,
        n_repeats=10, random_state=seed, n_jobs=-1
    )
    imp = pd.Series(r.importances_mean, index=feature_names)
    plot_importance(imp, "Permutation Importance", top_n=top_n)
```

---

## Опционально: внешние библиотеки (XGBoost/LightGBM/CatBoost/SHAP)

Если разрешено использовать внешние библиотеки, то часто добавляют:
- `xgboost.XGBRegressor`
- `lightgbm.LGBMRegressor`
- `catboost.CatBoostRegressor`
- `shap` для интерпретации

Но в чистом шаблоне выше достаточно sklearn + optuna.

---

## Полный пример “пул моделей + optuna + сравнение”

```python
# 0) Определяем список моделей
MODEL_POOL = ["Ridge", "ElasticNet", "RandomForest", "HistGradientBoosting"]

# 1) Выбираем препроцессор
# Для линейных/kNN/SVR лучше scale_numeric=True
# Для деревьев можно scale_numeric=False
pre = make_preprocessor(X_train, scale_numeric=True)

# 2) Готовим версии данных (пример):
# - одна версия: preprocessor + model в Pipeline
# - можно делать несколько версий с разным препроцессингом/скейлингом

from sklearn.pipeline import Pipeline

DATA_VERSIONS = {
    "base": (X_train, X_test),
}

# 3) Подбор моделей
best_models = {}
best_params = {}
best_studies = {}

for version_name, (Xtr, Xte) in DATA_VERSIONS.items():
    for model_name in MODEL_POOL:
        builder = MODEL_BUILDERS[model_name]

        def build_model_fn(trial, _builder=builder):
            # Оборачиваем модель в Pipeline, чтобы препроцессинг обучался внутри CV честно
            model = _builder(trial, seed=SEED)
            return Pipeline(steps=[("pre", pre), ("model", model)])

        study, fitted_model = tune_model_optuna(
            build_model_fn,
            Xtr, y_train,
            n_trials=30,
            n_splits=5,
            seed=SEED
        )

        best_studies[(version_name, model_name)] = study
        best_params[(version_name, model_name)] = study.best_params
        best_models[(version_name, model_name)] = fitted_model

# 4) Сводная таблица
results_long, summary_grid = make_summary_grid(DATA_VERSIONS, MODEL_POOL, best_models, y_train, y_test)
display(summary_grid)
```

---

### Конец файла
