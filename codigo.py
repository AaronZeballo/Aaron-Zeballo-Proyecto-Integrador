#importar librerias
import pandas as pd
import numpy as np
import sklearn as sc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, classification_report, accuracy_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import StandardScaler

#cargar dataset
df = pd.read_csv("spam_base.csv")


# Separar el dataset por clases
df_spam = df[df['class'] == 1]
df_no_spam = df[df['class'] == 0]


cantidad_spam = int(len(df_no_spam) / 19)

# Tomar una muestra aleatoria de la clase SPAM con la cantidad calculada
df_spam_reducido = df_spam.sample(n=cantidad_spam, random_state=42)

# Unir nuevamente los dataframes y mezclar las filas para que no queden ordenadas
df_reducido = pd.concat([df_no_spam, df_spam_reducido]).sample(frac=1, random_state=42).reset_index(drop=True)

# Mostrar las cantidades y el porcentaje para verificar que funcionó
print("Cantidad de filas por clase:")
print(df_reducido['class'].value_counts())
print("\nPorcentaje de cada clase:")
print(df_reducido['class'].value_counts(normalize=True) * 100)


#Prueba para ver cuales son las predicciones con el desbalanceo artificial
x = df_reducido.drop('class', axis=1)
y = df_reducido['class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# --- Estandarizar los datos ---
# Esto es CRUCIAL para que el SVM (basado en distancias) pueda funcionar bien
scaler = StandardScaler()
# El fit (aprender medias y desviaciones) se hace SOLO en el train para evitar data leakage
x_train = scaler.fit_transform(x_train)
# El test solo se transforma usando las reglas aprendidas del train
x_test = scaler.transform(x_test)

# Diccionario con los modelos a probar
modelos = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Máquinas de Soporte Vectorial (SVM)": SVC(random_state=42),
    "Árbol de Decisión": DecisionTreeClassifier(random_state=42)
}

# Listas para guardar las métricas de cada experimento
resultados = {
    "Original": {"F1-Score": [], "Accuracy": [], "Precision": [], "Recall": []},
    "SMOTE": {"F1-Score": [], "Accuracy": [], "Precision": [], "Recall": []},
    "ADASYN": {"F1-Score": [], "Accuracy": [], "Precision": [], "Recall": []}
}
nombres_modelos = list(modelos.keys())

def evaluar_modelo(modelo, x_train, y_train, x_test, y_test, nombre_escenario):
    modelo.fit(x_train, y_train)
    y_pred = modelo.predict(x_test)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    
    resultados[nombre_escenario]["F1-Score"].append(f1)
    resultados[nombre_escenario]["Accuracy"].append(acc)
    resultados[nombre_escenario]["Precision"].append(prec)
    resultados[nombre_escenario]["Recall"].append(rec)
    
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f}")

# Evaluar Original
print("\n=== Evaluando Original ===")
for nombre, modelo in modelos.items():
    print(f"\n--- {nombre} ({'Original'}) ---")
    evaluar_modelo(modelo, x_train, y_train, x_test, y_test, "Original")

# --- Balanceo con SMOTE ---
# sampling_strategy=0.4 significa que la clase minoritaria llegará a ser el 40% del tamaño de la mayoritaria (lo que en el total representa ~28.5%, casi un 70/30)
smote = SMOTE(random_state=42, sampling_strategy=0.4)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)
print("\nCantidad de filas por clase (después de SMOTE en entrenamiento):")
print(y_train_smote.value_counts())

print("\n=== Evaluando modelos entrenados con SMOTE ===")
for nombre, modelo in modelos.items():
    print(f"\n--- {nombre} ({'SMOTE'}) ---")
    evaluar_modelo(modelo, x_train_smote, y_train_smote, x_test, y_test, "SMOTE")

#Balanceo con ADASYN
adasyn = ADASYN(random_state=42, sampling_strategy=0.4)
x_train_adasyn, y_train_adasyn = adasyn.fit_resample(x_train, y_train)
print("\nCantidad de filas por clase (después de ADASYN en entrenamiento):")
print(y_train_adasyn.value_counts())

print("\n=== Evaluando modelos entrenados con ADASYN ===")
for nombre, modelo in modelos.items():
    print(f"\n--- {nombre} ({'ADASYN'}) ---")
    evaluar_modelo(modelo, x_train_adasyn, y_train_adasyn, x_test, y_test, "ADASYN")

# Generar gráfico comparativo con subplots (1x2) de barras
x = np.arange(len(nombres_modelos))
width = 0.25 # Más fino porque ahora son 3 barras

fig, axs = plt.subplots(1, 2, figsize=(14, 6))
metricas = ["F1-Score", "Recall"]
colores = {'Original': '#ff9999', 'SMOTE': '#26FF00', 'ADASYN': '#702963'}

for i, metrica in enumerate(metricas):
    ax = axs[i]
    
    rects1 = ax.bar(x - width, resultados["Original"][metrica], width, label='Original', color=colores['Original'])
    rects2 = ax.bar(x, resultados["SMOTE"][metrica], width, label='SMOTE', color=colores['SMOTE'])
    rects3 = ax.bar(x + width, resultados["ADASYN"][metrica], width, label='ADASYN', color=colores['ADASYN'])
    
    ax.set_ylabel(metrica, fontsize=12)
    ax.set_title(f'Comparación de {metrica}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(nombres_modelos, fontsize=11)
    ax.set_ylim([0, 1.15]) # Margen superior para los textos numéricos
    
    # Añadir valores numéricos sobre las barras
    ax.bar_label(rects1, fmt='%.2f', padding=3, fontsize=10)
    ax.bar_label(rects2, fmt='%.2f', padding=3, fontsize=10)
    ax.bar_label(rects3, fmt='%.2f', padding=3, fontsize=10)
    
    if i == 0: # Solo poner leyenda en el primer gráfico
        ax.legend(loc='upper right', fontsize=12)

fig.suptitle('Impacto del Balanceo y Estandarización (Original vs SMOTE vs ADASYN)', fontsize=18, y=1.05, fontweight='bold')
fig.tight_layout()
plt.savefig("grafico_comparativo.png", bbox_inches='tight')
print("\n¡Gráfico de barras (1x2) generado y guardado como 'grafico_comparativo.png'!")
plt.show()