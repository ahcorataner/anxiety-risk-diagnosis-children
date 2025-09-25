import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 1. Load dataset
path = r"C:\Users\Windows-SSD\Desktop\Artigo CILAMCE\dados\base_limpa_normalizada_tratada_ajustada.xlsx"
df = pd.read_excel(path)

# 2. Separate features and target
X = df.drop(columns=['Anxiety_Multilevel'])
y = df['Anxiety_Multilevel']

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

# ===============================================================
# PARTE A – Ajuste de hiperparâmetros (somente para responder revisor)
# ===============================================================

# Definir grids
rf_param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20],
    'criterion': ['gini', 'entropy']
}

svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf', probability=True, random_state=42))
])
svm_param_grid = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': [0.001, 0.01, 0.1, 1]
}

mlp_param_grid = {
    'hidden_layer_sizes': [(64,), (100,), (64, 32)],
    'activation': ['relu', 'tanh'],
    'learning_rate_init': [0.001, 0.01]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Random Forest
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42),
                       rf_param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
grid_rf.fit(X_train, y_train)

# SVM
grid_svm = GridSearchCV(svm_pipeline,
                        svm_param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
grid_svm.fit(X_train, y_train)

# MLP
grid_mlp = GridSearchCV(MLPClassifier(max_iter=500, random_state=42),
                        mlp_param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
grid_mlp.fit(X_train, y_train)

print("Melhores parâmetros RF:", grid_rf.best_params_)
print("Melhor acurácia média RF:", grid_rf.best_score_)

print("Melhores parâmetros SVM:", grid_svm.best_params_)
print("Melhor acurácia média SVM:", grid_svm.best_score_)

print("Melhores parâmetros MLP:", grid_mlp.best_params_)
print("Melhor acurácia média MLP:", grid_mlp.best_score_)

# ===============================================================
# PARTE B – Treino com parâmetros originais (mantém os resultados do artigo)
# ===============================================================

# 4. Train models com parâmetros fixos (iguais ao artigo)
rf = RandomForestClassifier(random_state=42)
svm = SVC(kernel='rbf', probability=True, random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)

rf.fit(X_train, y_train)
svm.fit(X_train, y_train)
mlp.fit(X_train, y_train)

# 5. Predictions
y_pred_rf = rf.predict(X_test)
y_pred_svm = svm.predict(X_test)
y_pred_mlp = mlp.predict(X_test)

# 6. Predictions dictionary
models = {
    'Random Forest': y_pred_rf,
    'SVM': y_pred_svm,
    'MLP': y_pred_mlp
}

# 7. Metrics by model
results = []
for name, y_pred in models.items():
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    accuracy = np.trace(cm) / cm.sum()
    sensitivity = cm.diagonal() / cm.sum(axis=1)
    specificity = []
    for i in range(3):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(np.delete(cm, i, axis=0)[:, i])
        specificity.append(tn / (tn + fp))
    results.append({
        'Model': name,
        'Accuracy': round(accuracy, 3),
        'Sens_Low': round(sensitivity[0], 3),
        'Sens_Moderate': round(sensitivity[1], 3),
        'Sens_High': round(sensitivity[2], 3),
        'Spec_Low': round(specificity[0], 3),
        'Spec_Moderate': round(specificity[1], 3),
        'Spec_High': round(specificity[2], 3)
    })

# 8. Create DataFrame and export
df_metrics = pd.DataFrame(results)
df_metrics.to_excel("classification_metrics.xlsx", index=False)

# 9. Display and save table as image
def display_table_as_image(df, file_name='metrics_table.png'):
    fig, ax = plt.subplots(figsize=(11, 2.5))
    ax.axis('off')
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    plt.title("Model Performance Metrics", fontsize=14, pad=10)
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    plt.show()

display_table_as_image(df_metrics)

# 10. Bar plot comparing models
labels = ['Accuracy', 'Average Sensitivity', 'Average Specificity']
bar_values = []
for row in df_metrics.itertuples():
    avg_sens = np.mean([row.Sens_Low, row.Sens_Moderate, row.Sens_High])
    avg_spec = np.mean([row.Spec_Low, row.Spec_Moderate, row.Spec_High])
    bar_values.append([row.Accuracy, avg_sens, avg_spec])

x = np.arange(len(labels))
bar_width = 0.25
plt.figure(figsize=(10, 6))
for i, model in enumerate(models.keys()):
    plt.bar(x + i * bar_width, np.array(bar_values[i]) * 100, width=bar_width, label=model)

plt.xticks(x + bar_width, labels, fontsize=11)
plt.ylabel('Performance (%)')
plt.title('Model Performance Comparison', fontsize=14)
plt.legend()
plt.ylim(60, 105)
for i in range(3):
    for j in range(len(models)):
        value = bar_values[j][i] * 100
        plt.text(x[i] + j * bar_width, value + 1.5, f"{value:.0f}%", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig("model_comparison_barplot.png", dpi=300)
plt.show()

# 11. Confusion matrices
readable_labels = ['Low Risk', 'Moderate Risk', 'High Risk']
plt.figure(figsize=(18, 5))
for i, (name, y_pred) in enumerate(models.items()):
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    plt.subplot(1, 3, i + 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=readable_labels, yticklabels=readable_labels)
    plt.title(f'{name}', fontsize=13)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')

plt.suptitle('Confusion Matrices — Anxiety Classification', fontsize=16)
plt.tight_layout()
plt.savefig("confusion_matrices_models.png", dpi=300)
plt.show()
