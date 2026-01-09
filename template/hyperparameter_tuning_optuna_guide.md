# Подбор гиперпараметров (Hyperparameter Tuning) — подробный гайд + шаблоны Optuna

Этот файл — конспект и “копипаст”-шаблоны для подбора гиперпараметров в задачах **регрессии и классификации**.  
Фокус: **Optuna + кросс‑валидация + Pipeline**, чтобы не было leakage и всё воспроизводилось.

---

## Содержание

1. [Что такое гиперпараметры](#что-такое-гиперпараметры)  
2. [Базовая стратегия подбора](#базовая-стратегия-подбора)  
3. [Виды поиска: Grid / Random / Bayesian / Evolutionary](#виды-поиска-grid--random--bayesian--evolutionary)  
4. [Почему нужен Pipeline и как избежать утечек](#почему-нужен-pipeline-и-как-избежать-утечек)  
5. [Выбор целевой метрики](#выбор-целевой-метрики)  
6. [Правильная CV-схема: KFold / StratifiedKFold / TimeSeriesSplit](#правильная-cv-схема-kfold--stratifiedkfold--timeseriessplit)  
7. [Optuna: основные понятия и настройки](#optuna-основные-понятия-и-настройки)  
8. [Шаблон: Optuna + CV + refit лучшей модели](#шаблон-optuna--cv--refit-лучшей-модели)  
9. [Search space: как задавать диапазоны](#search-space-как-задавать-диапазоны)  
10. [Pruning (ранняя остановка) и экономия времени](#pruning-ранняя-остановка-и-экономия-времени)  
11. [Варианты “версий данных” и сравнение](#варианты-версий-данных-и-сравнение)  
12. [Результаты: сводная таблица, сохранение, воспроизводимость](#результаты-сводная-таблица-сохранение-воспроизводимость)  
13. [Частые ошибки](#частые-ошибки)

---

## Что такое гиперпараметры

- **Параметры модели** — обучаются из данных (например, коэффициенты линейной регрессии).
- **Гиперпараметры** — задаются *до обучения* и управляют поведением алгоритма  
  (например `alpha` у Ridge, `max_depth` у деревьев, `learning_rate` у бустинга).

Подбор гиперпараметров нужен, чтобы найти режим работы модели, который лучше всего подходит под данные.

---

## Базовая стратегия подбора

Правильный порядок:

1) Подготовить данные (cleaning)  
2) Разделить train/test  
3) На **train** сделать подбор гиперпараметров через **CV**  
4) Обучить финальную модель на **train** с лучшими параметрами  
5) Оценить на **test** (один раз, честно)  
6) Зафиксировать результат + параметры + seed

---

## Виды поиска: Grid / Random / Bayesian / Evolutionary

- **GridSearch**: перебирает все комбинации сетки параметров (дорого при большой сетке).
- **RandomSearch**: выбирает случайные комбинации (часто эффективнее на большом пространстве).
- **Bayesian optimization**: использует историю испытаний, чтобы выбирать следующие параметры умнее.
  - Optuna (TPE sampler) — популярная реализация.
- **Evolutionary/Genetic**: эволюционный подбор (реже в стандартных проектах).

---

## Почему нужен Pipeline и как избежать утечек

### Главная ошибка
Делать `fit_transform()` на всём датасете до CV или до train/test split.

### Правильно
Положить **препроцессинг** и **модель** в `Pipeline`. Тогда:
- в каждом CV-фолде `fit` препроцессора делается только на train‑части фолда;
- утечки не будет.

---

## Выбор целевой метрики

### Регрессия
- RMSE — часто хорошая целевая метрика для подбора (штрафует большие ошибки)
- MAE — если важнее средняя ошибка без сильного штрафа за редкие большие промахи
- R² — удобен для интерпретации, но не всегда лучший для оптимизации

### Классификация
- Accuracy — только если классы сбалансированы
- F1 — если важен баланс Precision/Recall
- ROC-AUC — если важно ранжирование
- PR-AUC — если сильный дисбаланс

**Правило:** подбираем гиперпараметры по одной **целевой метрике**, а в отчёте показываем несколько.

---

## Правильная CV-схема: KFold / StratifiedKFold / TimeSeriesSplit

- **KFold** (регрессия): обычная кросс‑валидация.
- **StratifiedKFold** (классификация): сохраняет доли классов в фолдах.
- **TimeSeriesSplit**: для временных рядов (нельзя перемешивать будущее с прошлым).

---

## Optuna: основные понятия и настройки

- **Trial** — одно испытание (один набор параметров).
- **Study** — весь процесс оптимизации.
- **Sampler** — стратегия выбора параметров (часто `TPESampler`).
- **Pruner** — ранняя остановка плохих trial’ов (экономит время).
- **Direction** — `minimize` или `maximize`.

Рекомендуемая база:
- `TPESampler(seed=SEED)`
- `MedianPruner(n_warmup_steps=5)`

---

## Шаблон: Optuna + CV + refit лучшей модели

### 1) Метрики

```python
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def metrics_regression(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }
```

### 2) CV-оценка (регрессия по RMSE)

```python
from sklearn.model_selection import KFold, cross_val_score

def cv_rmse(model, X, y, n_splits=5, seed=42):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
    return float(-scores.mean())
```

### 3) Optuna‑обвязка

```python
import optuna

def tune_optuna(build_model_fn, X_train, y_train, n_trials=50, n_splits=5, seed=42):
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)

    def objective(trial):
        model = build_model_fn(trial)
        return cv_rmse(model, X_train, y_train, n_splits=n_splits, seed=seed)

    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_model = build_model_fn(optuna.trial.FixedTrial(study.best_params))
    best_model.fit(X_train, y_train)

    return study, best_model
```

---

## Search space: как задавать диапазоны

### Основные приёмы
- **логарифмический диапазон** для параметров масштаба (`alpha`, `C`, `learning_rate`)  
  `suggest_float(..., log=True)`
- **категориальные варианты** для дискретных режимов (`max_features`, `criterion`)  
  `suggest_categorical(...)`
- **целые параметры** для количества (`n_estimators`, `max_depth`)  
  `suggest_int(...)`

### Пример “хорошего” search space
- Ridge: `alpha` от `1e-4` до `1e2` (log)
- ElasticNet: `alpha` `1e-5..10` (log), `l1_ratio` `0..1`
- RandomForest: `n_estimators 200..1200`, `max_depth 3..40`, `min_samples_leaf 1..20`
- HistGB: `learning_rate 0.01..0.3` (log), `max_leaf_nodes 15..255`, `l2_regularization 1e-8..1e-2`

---

## Pruning (ранняя остановка) и экономия времени

Pruning нужен, когда trial “явно плохой” и нет смысла тратить время до конца.

### 1) Когда pruning реально помогает
- большие `n_trials`
- дорогие модели
- много фолдов CV

### 2) Как использовать
- включить pruner (`MedianPruner` обычно хватает)
- в objective репортить промежуточные значения (актуально, если ты сам пишешь цикл обучения/валидации)

Для стандартного `cross_val_score` pruning работает ограниченно (там нет промежуточных шагов), но study всё равно будет нормально искать.

---

## Варианты “версий данных” и сравнение

Частая практика: несколько вариантов данных (например скейлинг):
- `no_scaling`
- `standard`
- `minmax`
- `robust`

Оптимально:
- подобрать гиперпараметры **для каждой версии данных** и каждой модели
- свести всё в таблицу и выбрать лучшую связку по test‑метрике

---

## Результаты: сводная таблица, сохранение, воспроизводимость

### 1) Что сохранять
- `best_params`
- `best_value` (лучшее значение целевой метрики)
- seed / схема CV / n_splits
- итоговую обученную модель (joblib)
- итоговые метрики train/test

### 2) Сохранение модели
```python
import joblib
joblib.dump(best_model, "best_model.joblib")
```

### 3) Сохранение study (если хочешь воспроизводить)
Optuna умеет хранить study в SQLite (удобно для больших экспериментов).

---

## Частые ошибки

### 1) Leakage
- скейлинг/импутация сделаны до CV → завышенные результаты  
**Решение:** `Pipeline`

### 2) Несовпадение длин X и y
- удалили строки в X_train, забыли в y_train  
**Проверка:**
```python
X_train.index.equals(y_train.index)
```

### 3) Слишком широкие диапазоны
- Optuna тратит trials на бессмысленные области  
**Решение:** сузить search space по здравому смыслу.

### 4) Неправильная метрика
- оптимизируешь R², а нужно минимизировать RMSE  
**Решение:** выбрать 1 целевую метрику под задачу.

---

## Пример: готовый “builder” для Ridge внутри Pipeline

```python
import numpy as np
import optuna
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

def build_ridge_pipe(trial, preprocessor, seed=42):
    alpha = trial.suggest_float("alpha", 1e-4, 100.0, log=True)
    model = Ridge(alpha=alpha, random_state=seed)
    return Pipeline([("pre", preprocessor), ("model", model)])
```

---

### Конец файла
