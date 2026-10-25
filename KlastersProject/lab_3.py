import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier

# БАЗОВЫЙ ДАТАСЕТ

data = {
    'продукт': ['Яблоко', 'Салат', 'Бекон', 'Банан', 'Орехи', 'Рыба', 'Сыр', 'Виноград', 'Морковь', 'Апельсин'],
    'сладость': [7, 2, 1, 9, 1, 1, 1, 8, 2, 6],
    'хруст':    [7, 5, 2, 1, 5, 1, 1, 1, 8, 1],
    'класс':    ['Фрукт', 'Овощ', 'Протеин', 'Фрукт', 'Протеин', 'Протеин', 'Протеин', 'Фрукт', 'Овощ', 'Фрукт']
}
df = pd.DataFrame(data)
df['label'] = df['класс'].map({'Протеин': 0, 'Овощ': 1, 'Фрукт': 2})
inv = {0: 'Протеин', 1: 'Овощ', 2: 'Фрукт'}

print("Базовый датасет (2 признака):")
print(df[['продукт', 'сладость', 'хруст', 'класс']].to_string(index=False))

plt.figure(figsize=(10, 10))
colors = {0: 'blue', 1: 'orange', 2: 'green'}
markers = {0: 'o', 2: 'o', 1: 'o'}
for lbl in [0, 1, 2]:
    sub = df[df['label'] == lbl] #выбираются только те строки, у которых label равен текущему lbl
    plt.scatter(sub['сладость'], sub['хруст'], c=colors[lbl], marker=markers[lbl],
                s=150, label=inv[lbl])
for _, r in df.iterrows():
    plt.annotate(r['продукт'], (r['сладость'], r['хруст'])) # подписи точек
plt.xlabel('Сладость'); plt.ylabel('Хруст') # подписи осей
plt.title('Базовый датасет')
plt.legend(); plt.grid(alpha=0.3); plt.xlim(-1, 11); plt.ylim(-1, 11)
plt.show()

# РАСШИРЕННЫЙ ДАТАСЕТ

data2 = {
    'продукт': ['Яблоко', 'Банан', 'Виноград', 'Апельсин', 'Груша', 'Салат', 'Морковь', 'Огурец', 'Капуста', 'Бекон', 'Орехи', 'Рыба', 'Сыр', 'Курица', 'Молоко', 'Творог', 'Йогурт', 'Сметана', 'Сливки'],
    'сладость': [7, 9, 8, 6, 8, 2, 2, 1, 2, 1, 1, 1, 1, 1, 4, 2, 5, 1, 3],
    'хруст':    [7, 1, 1, 1, 6, 5, 8, 10, 7, 2, 5, 1, 1, 2, 0, 0, 0, 0, 0],
    'жирность': [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 1, 3, 2, 6, 8, 5, 10, 7],
    'класс':    ['Фрукт', 'Фрукт', 'Фрукт', 'Фрукт', 'Фрукт', 'Овощ', 'Овощ', 'Овощ', 'Овощ', 'Протеин', 'Протеин', 'Протеин', 'Протеин', 'Протеин', 'Молочный', 'Молочный', 'Молочный', 'Молочный', 'Молочный']
}
df2 = pd.DataFrame(data2)
map2 = {'Протеин': 0, 'Овощ': 1, 'Фрукт': 2, 'Молочный': 3}
inv2 = {v: k for k, v in map2.items()}
df2['label'] = df2['класс'].map(map2)

print("\nРасширенный датасет (3 признака):")
print(df2[['продукт', 'сладость', 'хруст', 'жирность', 'класс']].to_string(index=False))

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
clr = {0: 'blue', 1: 'orange', 2: 'green', 3: 'purple'}
for lbl in [0, 1, 2, 3]:
    sub = df2[df2['label'] == lbl]
    ax.scatter(sub['сладость'], sub['хруст'], sub['жирность'],
               c=clr[lbl], s=120, label=inv2[lbl])
for _, r in df2.iterrows():
    ax.text(r['сладость'], r['хруст'], r['жирность'], r['продукт'], fontsize=7)
ax.set_xlabel('Сладость'); ax.set_ylabel('Хруст'); ax.set_zlabel('Жирность')
ax.set_title('Расширенный датасет')
ax.legend()
plt.show()

# KNN

X2 = df2[['сладость', 'хруст', 'жирность']].values
y2 = df2['label'].values
indices = np.arange(len(df2))
X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split( # функция делит данные
    X2, y2, indices, test_size=0.3, random_state=20, stratify=y2
)

print(f"\nОбучающая выборка: {len(X_tr)} продуктов")
print(f"Тестовая выборка: {len(X_te)} продуктов")

print("\nКросс-валидация:")
scores_ext = []
for k in range(1, 9):
    try:
        s = cross_val_score(KNeighborsClassifier(k), X_tr, y_tr,
                             cv=KFold(3, shuffle=True, random_state=42)).mean()
    except:
        s = np.nan
    scores_ext.append(s)

best_k = np.nanargmax(scores_ext) + 1
print(f"Лучший K: {best_k}")

model = KNeighborsClassifier(best_k).fit(X_tr, y_tr)
pred_te = model.predict(X_te)

print("\nРезультаты на тестовых продуктах:")
results = pd.DataFrame({
    'продукт': df2.iloc[idx_te]['продукт'].values,
    'на самом деле': df2.iloc[idx_te]['класс'].values,
    'предсказано': [inv2[p] for p in pred_te]
})
print(results.to_string(index=False))
