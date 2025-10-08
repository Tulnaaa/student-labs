import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# 1. Загрузка данных

print("Загрузка данных...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. Проверка структуры данных

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# идентификатор записи и целевая переменная
if train_df.columns[0] != 'id' or train_df.columns[-1] != 'smoking':
    raise ValueError("Ожидается, что первый столбец — 'id', последний — 'smoking'.")

feature_columns = train_df.columns[1:-1]
target_column = 'smoking'
id_column = 'id'

# Проверка train и test
if not test_df.columns[1:].equals(feature_columns):
    raise ValueError("Несовпадение признаков между train и test!")

# 3. Подготовка данных

X_train = train_df[feature_columns]
y_train = train_df[target_column]
X_test = test_df[feature_columns]
test_ids = test_df[id_column]

# Проверка на пропуски
if X_train.isnull().any().any() or X_test.isnull().any().any():
    print("⚠️ Обнаружены пропущенные значения! Заполняем средним.")
    X_train.fillna(X_train.mean(), inplace=True)
    X_test.fillna(X_test.mean(), inplace=True)

# 4. Настройка модели

model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=42,
    validation_fraction=0.1,
    n_iter_no_change=10,
    tol=1e-4
)

# 5. Оценка качества модели

print("Оценка модели...")

# Stratified hold-out validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

model_val = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=42,
    validation_fraction=0.1,
    n_iter_no_change=10,
    tol=1e-4
)
model_val.fit(X_tr, y_tr)
val_auc = roc_auc_score(y_val, model_val.predict_proba(X_val)[:, 1])
print(f"Валидационный ROC-AUC (hold-out): {val_auc:.4f}")

# Кросс-валидация
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
print(f"Кросс-валидация ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std() * 2:.4f}")

# 6. Обучение финальной модели на всех данных

print("Обучение финальной модели...")
model.fit(X_train, y_train)

# 7. Предсказание и сохранение

test_pred = model.predict_proba(X_test)[:, 1]

submission_df = pd.DataFrame({
    'id': test_ids.astype(int),
    'smoking': test_pred
})

submission_df.to_csv('submission.csv', index=False)
print("✅ Файл submission.csv успешно создан!")