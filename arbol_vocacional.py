import pandas as pd
import mysql.connector
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import dataframe_image as dfi
import matplotlib.image as mpimg
# ===================================
# 1. CONEXIÓN Y CARGA DE DATOS
# ===================================
conn = mysql.connector.connect(
    host="localhost",
    port=3306,
    user="root",
    password="root",
    database="test_vocacional"
)

# Leer todos los datos de la tabla "respuestas" antes de cerrar conexión
query = "SELECT * FROM respuestas"
df = pd.read_sql(query, conn)
conn.close()

# ===================================
# 2. PREPROCESAMIENTO DE DATOS
# ===================================

# Mapear grados a valores numéricos
grado_map = {
    "Tercero de Secundaria": 3,
    "Cuarto de Secundaria": 4,
    "Quinto de Secundaria": 5
}
df["GradoNum"] = df["Grado"].map(grado_map)

# Separar variables predictoras y la variable objetivo
X = df.loc[:, ["Edad", "GradoNum"] + [f"P{i}" for i in range(1, 31)]]
y = df["Carrera"]

# ===================================
# 3. ENTRENAMIENTO DEL MODELO
# ===================================

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Crear y entrenar el árbol de decisión
modelo = DecisionTreeClassifier(criterion="entropy", max_depth=5, min_samples_split=3)
modelo.fit(X_train, y_train)

# Predecir sobre el conjunto de prueba
y_pred = modelo.predict(X_test)

# Calcular precisión
precision = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {precision:.2f}")

# Obtener importancia de variables
importancias = modelo.feature_importances_
importancia_df = pd.DataFrame({
    'Variable': X.columns,
    'Importancia': importancias
}).sort_values(by='Importancia', ascending=False)

print("\nTop variables más influyentes en la decisión:")
print(importancia_df.head(10))

# ===================================
# 4. VISUALIZACIÓN DEL ÁRBOL
# ===================================
plt.figure(figsize=(20, 10))
plot_tree(modelo, feature_names=X.columns, class_names=modelo.classes_, filled=True)
plt.title("Árbol de Decisión - Test Vocacional", fontsize=16)
plt.savefig("arbol_decision.png", dpi=300, bbox_inches='tight')
plt.show()

# ===================================
# 5. BALANCEO DE LAS 5 CARRERAS MENOS TOMADAS
# ===================================

# Obtener las 5 carreras con menos estudiantes
bottom5 = df["Carrera"].value_counts(ascending=True).head(5).reset_index()
bottom5.columns = ["Carrera", "Cantidad"]
print("\nTop 5 carreras menos escogidas:")
print(bottom5)

# Filtrar los datos de esas 5 carreras
carreras_poco_tomadas = bottom5["Carrera"].tolist()
df_poco = df[df["Carrera"].isin(carreras_poco_tomadas)]

# Sobremuestreo para balancear
max_samples = df_poco["Carrera"].value_counts().max()
df_balanceado = df_poco.groupby("Carrera").apply(lambda x: x.sample(max_samples, replace=True)).reset_index(drop=True)

# Mostrar el nuevo conteo
print("\nCarreras poco tomadas (balanceadas):")
print(df_balanceado["Carrera"].value_counts())

# Visualizar gráfico de carreras balanceadas con legend=False para evitar FutureWarning
plt.figure(figsize=(10, 6))
sns.countplot(
    y="Carrera",
    data=df_balanceado,
    order=df_balanceado["Carrera"].value_counts().index,
    palette="Set2",
    legend=False
)
plt.title("Carreras menos tomadas (balanceadas)")
plt.xlabel("Cantidad de Estudiantes")
plt.tight_layout()
plt.savefig("carreras_menos_balanceadas.png", dpi=300)
plt.show()


# Obtener top 5 carreras más frecuentes
top5_carreras = df["Carrera"].value_counts().head(5).index.tolist()

# Filtrar solo esas carreras
df_top5 = df[df["Carrera"].isin(top5_carreras)]

# Generar resumen
resumen_top5 = df_top5.groupby("Carrera").agg({
    "Edad": ["count", "mean", "min", "max"],
    "Grado": lambda x: x.value_counts().idxmax()
})

# Crear gráfico de barras
plt.figure(figsize=(10, 6))
sns.countplot(
    y="Carrera",
    data=df_top5,
    order=df_top5["Carrera"].value_counts().index,
    palette="Set2",
    legend=False
)

resumen_top5.columns = ["Total_Estudiantes", "Edad_Promedio", "Edad_Min", "Edad_Max", "Grado_Mas_Frecuente"]
resumen_top5 = resumen_top5.reset_index()
plt.title("Carreras Mas tomadas (balanceadas)")
plt.xlabel("Cantidad de Estudiantes")
plt.tight_layout()
plt.savefig("carreras_mas_balanceadas.png", dpi=300)
plt.show()



# ===================================
# 6. RESUMEN ESTADÍSTICO POR CARRERA
# ===================================

# Generar resumen estadístico por carrera
resumen = df.groupby("Carrera").agg({
    "Edad": ["count", "mean", "min", "max"],
    "Grado": lambda x: x.value_counts().idxmax()
})

# Renombrar columnas
resumen.columns = ["Total_Estudiantes", "Edad_Promedio", "Edad_Min", "Edad_Max", "Grado_Mas_Frecuente"]
resumen = resumen.reset_index()

# Guardar como CSV
resumen.to_csv("Resumen_Estadistico_Carreras.csv", index=False, encoding="utf-8")



# Guardar como Excel también (requiere openpyxl instalado)
resumen.to_excel("Resumen_Estadistico_Carreras.xlsx", index=False)

# Guardar el resumen estadístico como imagen estilo Excel
dfi.export(resumen, "Resumen_Estadistico_Carreras.png")
print("📸 Imagen generada: Resumen_Estadistico_Carreras.png")



# Mostrar imagen del resumen estadístico como una tabla visual
img = mpimg.imread("Resumen_Estadistico_Carreras.png")
plt.figure(figsize=(12, 8))  # ajusta tamaño según lo que necesites
plt.imshow(img)
plt.axis("off")
plt.title("Resumen Estadístico por Carrera", fontsize=16)
plt.tight_layout()
plt.show()


# ===================================
# 7. CURVA ROC MULTICLASE - ÁRBOL DE DECISIÓN
# ===================================
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import numpy as np

# --- Preparar datos para ROC ---
y_encoded = pd.factorize(y)[0]
y_test_encoded = pd.factorize(y_test)[0]
y_test_bin = label_binarize(y_test_encoded, classes=np.unique(y_encoded))

try:
    y_score = modelo.predict_proba(X_test)
    fpr, tpr, roc_auc = {}, {}, {}
    clases_validas = []

    for i in range(y_score.shape[1]):
        if np.sum(y_test_bin[:, i]) > 0:
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            clases_validas.append(i)

    if clases_validas:
        all_fpr = np.unique(np.concatenate([fpr[i] for i in clases_validas]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in clases_validas:
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= len(clases_validas)
        macro_auc = auc(all_fpr, mean_tpr)

        plt.figure(figsize=(10, 8))
        for i in clases_validas:
            plt.plot(fpr[i], tpr[i], lw=1.5, label=f"Clase {modelo.classes_[i]} (AUC = {roc_auc[i]:.2f})")

        plt.plot(all_fpr, mean_tpr, "--", color="navy", label=f"Promedio Macro (AUC = {macro_auc:.2f})", lw=2)
        plt.plot([0, 1], [0, 1], "k--", lw=1.5)
        plt.xlabel("Tasa de Falsos Positivos")
        plt.ylabel("Tasa de Verdaderos Positivos")
        plt.title("Curva ROC Multiclase - Árbol de Decisión")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig("Curva_ROC_arbol.png", dpi=300)
        plt.show()
    else:
        print("⚠️ No se pudieron calcular curvas ROC: ninguna clase válida en test.")

except Exception as e:
    print("❌ Error al calcular la Curva ROC:", str(e))