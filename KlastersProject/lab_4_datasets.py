import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Perceptron
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score

np.random.seed(20)

# загрузка данных
df = pd.read_csv('IRIS.csv')
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

# кодирование для кластеризации
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# подготовка данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=20, stratify=y
)

# обучение на сырых данных
models = {
    'Perceptron': Perceptron(max_iter=1000, random_state=20, tol=1e-3),
    'OutputCodeClassifier': OutputCodeClassifier(SVC(kernel='rbf'), code_size=2, random_state=20)
}

results_raw = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results_raw[name] = accuracy_score(y_test, y_pred)

# стандартизация
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# обучение на маштабируемых данных
results_scaled = {}
for name, model in models.items():
    if name == 'Perceptron':
        model_scaled = Perceptron(max_iter=1000, random_state=20, tol=1e-3)
    else:
        model_scaled = OutputCodeClassifier(SVC(kernel='rbf'), code_size=2, random_state=20)

    model_scaled.fit(X_train_scaled, y_train)
    y_pred = model_scaled.predict(X_test_scaled)
    results_scaled[name] = accuracy_score(y_test, y_pred)

# кросс валидация (cv)
scaler_full = StandardScaler()
X_scaled = scaler_full.fit_transform(X)

cv_scores_5fold = {} # точности 5-fold CV
cv_scores_10fold = {} # точности 10-fold CV
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=20)

for name, model in models.items():
    if name == 'Perceptron':
        model_cv = Perceptron(max_iter=1000, random_state=20, tol=1e-3)
    else:
        model_cv = OutputCodeClassifier(SVC(kernel='rbf'), code_size=2, random_state=20)

    scores5 = cross_val_score(model_cv, X_scaled, y, cv=5)
    scores10 = cross_val_score(model_cv, X_scaled, y, cv=skf)
    cv_scores_5fold[name] = scores5.mean()
    cv_scores_10fold[name] = scores10.mean()

# сводная таблица
results_df = pd.DataFrame({
    'Модель': list(results_raw.keys()),
    'Сырые': [results_raw['Perceptron'], results_raw['OutputCodeClassifier']],
    'Масштаб.': [results_scaled['Perceptron'], results_scaled['OutputCodeClassifier']],
    '5-fold CV': [cv_scores_5fold['Perceptron'], cv_scores_5fold['OutputCodeClassifier']],
    '10-fold CV': [cv_scores_10fold['Perceptron'], cv_scores_10fold['OutputCodeClassifier']]
})
print(results_df.to_string(index=False))

# ГРАФИКИ
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

x = np.arange(len(results_df))
width = 0.2

ax[0].bar(x - width * 1.5, results_df['Сырые'], width, label='Сырые', color='steelblue')
ax[0].bar(x - width / 2, results_df['Масштаб.'], width, label='Масштаб.', color='orange')
ax[0].bar(x + width / 2, results_df['5-fold CV'], width, label='5-fold', color='green')
ax[0].bar(x + width * 1.5, results_df['10-fold CV'], width, label='10-fold', color='red')
ax[0].set_xticks(x)
ax[0].set_xticklabels(results_df['Модель'])
ax[0].set_ylabel('Accuracy')
ax[0].set_title('Сравнение классификаторов')
ax[0].legend()
ax[0].set_ylim([0, 1.05])

cv_comparison = pd.DataFrame({
    'Модель': ['Perceptron', 'Perceptron', 'OutputCode', 'OutputCode'],
    'Метод': ['5-fold', '10-fold', '5-fold', '10-fold'],
    'Точность': [cv_scores_5fold['Perceptron'], cv_scores_10fold['Perceptron'],
                 cv_scores_5fold['OutputCodeClassifier'], cv_scores_10fold['OutputCodeClassifier']]
})
sns.barplot(data=cv_comparison, x='Модель', y='Точность', hue='Метод', ax=ax[1], palette='Set2')
ax[1].set_title('Сравнение кросс-валидации')
ax[1].set_ylim([0, 1.05])
plt.tight_layout()
plt.show()

# кластеризация
X_scaled_full = StandardScaler().fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=20, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled_full)

ari = adjusted_rand_score(y_encoded, cluster_labels)
ami = adjusted_mutual_info_score(y_encoded, cluster_labels)
hom = homogeneity_score(y_encoded, cluster_labels)

print(f"\nARI: {ari:.4f}, AMI: {ami:.4f}, Homogeneity: {hom:.4f}")

# матрица соответствия
confusion_cluster = pd.crosstab(y, cluster_labels, rownames=['Реальный'], colnames=['Кластер'])
print(confusion_cluster)

# визуализация кластеризации
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled_full)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, cmap='viridis', edgecolor='k', s=60)
axes[0].set_title('Реальные классы')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')

scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', edgecolor='k', s=60)
axes[1].set_title('KMeans кластеры')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
plt.colorbar(scatter, ax=axes[1], ticks=[0, 1, 2])
plt.tight_layout()
plt.show()

# выводы
print("\nВыводы")
print(f"1. Perceptron: сырые={results_raw['Perceptron']:.3f}, масштаб.={results_scaled['Perceptron']:.3f}")
print(
    f"2. OutputCodeClassifier: сырые={results_raw['OutputCodeClassifier']:.3f}, масштаб.={results_scaled['OutputCodeClassifier']:.3f}")
print(
    f"3. Лучший результат: OutputCodeClassifier ({max(results_raw['OutputCodeClassifier'], results_scaled['OutputCodeClassifier']):.3f})")
print(f"4. Кластеризация ARI={ari:.3f}: setosa отделён, versicolor/virginica смешаны")