import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, LassoLarsIC
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

np.random.seed(20)

print("Методы: ElasticNetCV и LassoLarsIC")

# загрузка и подготовка данных
df = pd.read_csv('IRIS.csv')

X = df[['sepal_width', 'petal_length', 'petal_width']]
X = pd.concat([X, pd.get_dummies(df['species'], prefix='species')], axis=1)
y = df['sepal_length']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# обучение
elasticnet_cv = ElasticNetCV(
    l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
    alphas=np.logspace(-4, 1, 50),
    cv=5,
    random_state=20,
    max_iter=10000,
    n_jobs=-1
)

lassolars_ic = LassoLarsIC(criterion='bic', max_iter=1000)

elasticnet_cv.fit(X_train_scaled, y_train)
y_pred_en = elasticnet_cv.predict(X_test_scaled)

lassolars_ic.fit(X_train_scaled, y_train)
y_pred_ll = lassolars_ic.predict(X_test_scaled)

# оценка качества
def evaluate(y_true, y_pred):
    return {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R^2': r2_score(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100
    }

res_en = evaluate(y_test, y_pred_en)
res_ll = evaluate(y_test, y_pred_ll)

print("\nРезультаты:")
print(f"{'Метрика':<10} {'ElasticNetCV':<15} {'LassoLarsIC':<15}")
for m in ['MSE', 'RMSE', 'MAE', 'R^2', 'MAPE']:
    print(f"{m:<10} {res_en[m]:<15.4f} {res_ll[m]:<15.4f}")

print(f"\nElasticNetCV: alpha={elasticnet_cv.alpha_:.6f}, l1_ratio={elasticnet_cv.l1_ratio_:.4f}")

criterion_val = lassolars_ic.criterion_
if isinstance(criterion_val, np.ndarray):
    criterion_val = criterion_val[-1]
print(f"LassoLarsIC: alpha={lassolars_ic.alpha_:.6f}, BIC={criterion_val:.2f}")

# кросс валидация
kfold = KFold(n_splits=5, shuffle=True, random_state=20)

scores_en = cross_val_score(elasticnet_cv, X_train_scaled, y_train, cv=kfold, scoring='r2')
scores_ll = cross_val_score(lassolars_ic, X_train_scaled, y_train, cv=kfold, scoring='r2')

print(f"\n5-fold CV R^2:")
print(f"ElasticNetCV: {scores_en.mean():.4f} (±{scores_en.std():.4f})")
print(f"LassoLarsIC: {scores_ll.mean():.4f} (±{scores_ll.std():.4f})")

# визаулизация
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# сравнение моделей
axes[0, 0].bar(['ElasticNetCV', 'LassoLarsIC'], [res_en['R^2'], res_ll['R^2']],
               color=['steelblue', 'darkorange'], edgecolor='black')
axes[0, 0].set_ylabel('R^2')
axes[0, 0].set_title('Сравнение R^2')
axes[0, 0].set_ylim([0, 1])
for i, v in enumerate([res_en['R^2'], res_ll['R^2']]):
    axes[0, 0].text(i, v + 0.01, f'{v:.4f}', ha='center')

# сравнение ошибок
x = np.arange(2)
width = 0.35
axes[0, 1].bar(x - width/2, [res_en['RMSE'], res_ll['RMSE']], width, label='RMSE', color='coral')
axes[0, 1].bar(x + width/2, [res_en['MAE'], res_ll['MAE']], width, label='MAE', color='teal')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(['ElasticNetCV', 'LassoLarsIC'])
axes[0, 1].set_ylabel('Ошибка')
axes[0, 1].set_title('Сравнение ошибок')
axes[0, 1].legend()

# коэффициенты
coef_df = pd.DataFrame({
    'Признак': X.columns,
    'ElasticNetCV': elasticnet_cv.coef_,
    'LassoLarsIC': lassolars_ic.coef_
})
x_pos = np.arange(len(coef_df))
axes[1, 0].bar(x_pos - width/2, coef_df['ElasticNetCV'], width, label='ElasticNetCV', color='steelblue')
axes[1, 0].bar(x_pos + width/2, coef_df['LassoLarsIC'], width, label='LassoLarsIC', color='darkorange')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(coef_df['Признак'], rotation=45, ha='right')
axes[1, 0].set_ylabel('Коэффициент')
axes[1, 0].set_title('Коэффициенты моделей')
axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[1, 0].legend()

# остатки
residuals_en = y_test - y_pred_en
residuals_ll = y_test - y_pred_ll
axes[1, 1].scatter(y_pred_en, residuals_en, alpha=0.5, label='ElasticNetCV', c='blue')
axes[1, 1].scatter(y_pred_ll, residuals_ll, alpha=0.5, label='LassoLarsIC', c='green', marker='s')
axes[1, 1].axhline(y=0, color='red', linestyle='--')
axes[1, 1].set_xlabel('Предсказанные значения')
axes[1, 1].set_ylabel('Остатки')
axes[1, 1].set_title('График остатков')
axes[1, 1].legend()

plt.tight_layout()
plt.show()