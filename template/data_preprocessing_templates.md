# Универсальный шаблон предобработки данных (ML) — подробный конспект + кодовые заготовки

Этот файл — “памятка + шаблоны кода” для предобработки **любого табличного датасета** под задачи ML.  
Подходит для ноутбука, проекта, Streamlit-дашборда.

---

## Содержание

1. [Цель предобработки](#цель-предобработки)  
2. [Мини-чеклист (что сделать в первую очередь)](#мини-чеклист-что-сделать-в-первую-очередь)  
3. [Загрузка данных и первичная диагностика](#загрузка-данных-и-первичная-диагностика)  
4. [Типы данных, “грязные” значения и приведение типов](#типы-данных-грязные-значения-и-приведение-типов)  
5. [Пропуски: поиск, стратегия заполнения, утечки](#пропуски-поиск-стратегия-заполнения-утечки)  
6. [Дубликаты и константные признаки](#дубликаты-и-константные-признаки)  
7. [Категориальные признаки: кодирование](#категориальные-признаки-кодирование)  
8. [Числовые признаки: скейлинг и преобразования](#числовые-признаки-скейлинг-и-преобразования)  
9. [Выбросы/аномалии: как проверять и как удалять аккуратно](#выбросыаномалии-как-проверять-и-как-удалять-аккуратно)  
10. [Feature engineering: генерация признаков и защита от leakage](#feature-engineering-генерация-признаков-и-защита-от-leakage)  
11. [Разбиение train/test и кросс-валидация](#разбиение-traintest-и-кросс-валидация)  
12. [Pipeline/ColumnTransformer: правильный production-подход](#pipelinecolumntransformer-правильный-production-подход)  
13. [Универсальная функция prepare_data()](#универсальная-функция-prepare_data)  
14. [Частые ошибки и как их быстро ловить](#частые-ошибки-и-как-их-быстро-ловить)

---

## Цель предобработки

Предобработка нужна, чтобы:
- привести данные к виду, который **модель может “переварить”**;
- уменьшить шум/ошибки (битые значения, пропуски, дубликаты);
- корректно обработать категориальные/числовые признаки;
- подготовить воспроизводимый пайплайн, который одинаково работает на train/test/production.

---

## Мини-чеклист (что сделать в первую очередь)

1) `df.shape`, `df.head()`  
2) `df.info()` и типы  
3) поиск пропусков / “грязных” значений (`'?'`, `'unknown'`, `'-'`, пустые строки)  
4) удаление идентификаторов (ID) и почти пустых колонок  
5) разделение на `X` и `y` + `train_test_split`  
6) заполнение пропусков **только через pipeline**  
7) кодирование категорий (обычно one-hot)  
8) скейлинг (если нужно: линейные/к-NN/SVR/нейросети)  
9) выбросы — осторожно: чаще на train и по понятной логике  
10) финальная проверка: `X_train.shape`, совпадение индексов `X/y`

---

## Загрузка данных и первичная диагностика

```python
import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")  # или read_excel, read_parquet

print("Shape:", df.shape)
display(df.head())
display(df.sample(5, random_state=42))
display(df.info())
```

Полезно:
```python
display(df.describe(include="all").T)
```

---

## Типы данных, “грязные” значения и приведение типов

### 1) Замена “битых” значений на NaN

Часто в файлах встречаются: `"?"`, `"NA"`, `"nan"`, `"null"`, `"-"`, `""`.

```python
bad_tokens = ["?", "NA", "N/A", "null", "None", "-", "—", ""]
df = df.replace(bad_tokens, np.nan)
```

### 2) Приведение числовых колонок к float

```python
to_numeric_cols = ["col1", "col2"]

for c in to_numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")  # всё “кривое” станет NaN
```

> Совет: не делай `astype(float)` сразу — оно упадёт на мусоре.

---

## Пропуски: поиск, стратегия заполнения, утечки

### 1) Быстрый отчёт по пропускам

```python
na = df.isna().mean().sort_values(ascending=False)
display((na * 100).round(2).to_frame("missing_%"))
```

### 2) Удаление почти пустых колонок

```python
thr = 0.95  # 95% пропусков
drop_cols = na[na > thr].index.tolist()
df = df.drop(columns=drop_cols)
```

### 3) Стратегии заполнения (общая логика)
- числовые: `median` (устойчиво), иногда `mean`
- категориальные: `most_frequent`
- продвинутый вариант: итеративная импутация (но только аккуратно и на train)

**Важно:** заполнять пропуски нужно так, чтобы не “подглядывать” в test → используем `Pipeline`.

---

## Дубликаты и константные признаки

### Дубликаты строк

```python
before = len(df)
df = df.drop_duplicates()
print("Removed duplicates:", before - len(df))
```

### Константные признаки (одинаковое значение почти везде)

```python
nunique = df.nunique(dropna=False)
const_cols = nunique[nunique <= 1].index.tolist()
df = df.drop(columns=const_cols)
```

---

## Категориальные признаки: кодирование

Самый универсальный способ — **OneHotEncoder** в `ColumnTransformer`.

Альтернатива для деревьев: иногда можно использовать `OrdinalEncoder`, но осторожно (появляется “ложный порядок”).

---

## Числовые признаки: скейлинг и преобразования

Скейлинг обычно нужен для:
- линейных моделей (Ridge/Lasso/ElasticNet)
- kNN
- SVM/SVR
- нейросетей

Не обязателен (но не мешает) для:
- деревьев, случайного леса, бустинга

### Популярные варианты:
- `StandardScaler` (z-score)
- `MinMaxScaler` (в [0..1])
- `RobustScaler` (устойчив к выбросам)

---

## Выбросы/аномалии: как проверять и как удалять аккуратно

Главная идея: **тест не трогаем**, а на train удаляем только “явные” точки.

### 1) Простая проверка через IQR

```python
def iqr_outliers_mask(s: pd.Series, k: float = 1.5):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (s < lo) | (s > hi)
```

### 2) Пересечение нескольких детекторов (осторожное удаление)

Идея: если точка подозрительна **сразу несколькими методами**, вероятность “мусора” выше.

---

## Feature engineering: генерация признаков и защита от leakage

### База
- агрегаты: сумма/разность/отношение
- взаимодействия: `a*b`, `a/b`, `log(a)`, `sqrt(a)`
- биннинг/категории из чисел (иногда)

### Защита от leakage
**Нельзя** строить признаки, где используется целевая переменная напрямую или косвенно.

---

## Разбиение train/test и кросс-валидация

```python
from sklearn.model_selection import train_test_split

TARGET = "target"
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
```

---

## Pipeline/ColumnTransformer: правильный production-подход

### Универсальный препроцессор

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def make_preprocessor(X, scale_numeric=True):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(steps=num_steps)

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])
```

### Встраивание в модель

```python
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

pre = make_preprocessor(X_train, scale_numeric=True)
model = Ridge(alpha=1.0)

pipe = Pipeline([
    ("pre", pre),
    ("model", model),
])

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
```

---

## Универсальная функция prepare_data()

Ниже — шаблон, который можно копировать в проект.

```python
def prepare_data(
    df: pd.DataFrame,
    target: str,
    drop_cols: list[str] | None = None,
    missing_threshold: float = 0.95
):
    df = df.copy()

    # 1) почистим “мусор” в строках
    bad_tokens = ["?", "NA", "N/A", "null", "None", "-", "—", ""]
    df = df.replace(bad_tokens, np.nan)

    # 2) удаление почти пустых колонок
    na = df.isna().mean()
    df = df.drop(columns=na[na > missing_threshold].index.tolist(), errors="ignore")

    # 3) удаление пользовательских колонок (ID и т.п.)
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # 4) отделяем X и y
    X = df.drop(columns=[target])
    y = df[target]

    return X, y
```

---

## Частые ошибки и как их быстро ловить

### 1) “Found input variables with inconsistent numbers of samples”
- где-то X и y разной длины
- часто: удалили строки в X_train, но забыли удалить в y_train

Проверка:
```python
print(len(X_train), len(y_train))
print(X_train.index.equals(y_train.index))
```

### 2) “x and y must be the same size” (scatter)
- `X[col]` и `y` имеют разные индексы/длины  
Решение: выравнивать по индексу:
```python
common = X.index.intersection(y.index)
x = X.loc[common, col]
yy = y.loc[common]
```

### 3) “unknown category” при one-hot на test
- решается: `OneHotEncoder(handle_unknown="ignore")`

---

### Конец файла
