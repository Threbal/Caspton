
import pandas as pd
import mysql.connector
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import dataframe_image as dfi
import os
import matplotlib.pyplot as plt

# ===================================
# 1. CARGA DE DATOS DESDE MYSQL
# ===================================
conn = mysql.connector.connect(
    host="localhost",
    port=3306,
    user="root",
    password="root",
    database="test_vocacional"
)
df = pd.read_sql("SELECT * FROM respuestas", conn)
conn.close()

# ===================================
# 2. PREPROCESAMIENTO
# ===================================
grado_map = {"Tercero de Secundaria": 3, "Cuarto de Secundaria": 4, "Quinto de Secundaria": 5}
df["GradoNum"] = df["Grado"].map(grado_map)

X = df.loc[:, ["Edad", "GradoNum"] + [f"P{i}" for i in range(1, 31)]]
y = df["Carrera"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ===================================
# 3. MODELO DE REGRESIÓN LOGÍSTICA
# ===================================
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, random_state=42)

modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

# Evaluación
y_pred = modelo.predict(X_test)
precision = accuracy_score(y_test, y_pred)
print(f"✅ Precisión del modelo de Regresión Logística: {precision:.2f}")

# ===================================
# 4. PREDICCIÓN Y COMPARACIÓN
# ===================================
y_pred_full = modelo.predict(X_scaled)
carreras_predichas = label_encoder.inverse_transform(y_pred_full)

df_resultado = df[["Edad", "Grado", "Carrera"]].copy()
df_resultado["Carrera_Predicha_LogReg"] = carreras_predichas
df_resultado["Coincide"] = df_resultado["Carrera"] == df_resultado["Carrera_Predicha_LogReg"]

# Guardar Excel e imagen
df_resultado.to_excel("Comparacion_Predicciones_Logistica.xlsx", index=False)
dfi.export(df_resultado.head(30), "Comparacion_Predicciones_Logistica.png")

# Abrir Excel automáticamente
os.startfile("Comparacion_Predicciones_Logistica.xlsx")

# ===================================
# 5. GRÁFICO DE COINCIDENCIAS
# ===================================
conteo = df_resultado["Coincide"].value_counts()
conteo.index = ["Correctas (True)", "Incorrectas (False)"]

plt.figure(figsize=(8, 5))
conteo.plot(kind="bar", color=["blue", "orange"])
plt.title("Comparación de Predicciones - Regresión Logística")
plt.ylabel("Cantidad de Estudiantes")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("Precision_Coincidencia_Logistica.png", dpi=300)
plt.show()



# ===================================
# 6. MATRIZ DE CONFUSIÓN - HEATMAP
# ===================================
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Calcular la matriz usando todo el dataset
mat = confusion_matrix(y_encoded, y_pred_full)
labels = label_encoder.classes_

plt.figure(figsize=(12, 10))
sns.heatmap(mat, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Matriz de Confusión - Regresión Logística", fontsize=14)
plt.xlabel("Predicción")
plt.ylabel("Carrera Real")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("Matriz_Confusion_Logistica.png", dpi=300)
plt.show()


# ===================================
# 7. CURVA ROC MULTICLASE - REGRESIÓN LOGÍSTICA
# ===================================
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# Repreparamos los datos binarizados
y_test_bin = label_binarize(y_test, classes=np.unique(y_encoded))

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
            nombre_clase = label_encoder.inverse_transform([i])[0]
            plt.plot(fpr[i], tpr[i], lw=1.5, label=f"{nombre_clase} (AUC = {roc_auc[i]:.2f})")

        plt.plot(all_fpr, mean_tpr, "--", color="darkgreen", label=f"Promedio Macro (AUC = {macro_auc:.2f})", lw=2)
        plt.plot([0, 1], [0, 1], "k--", lw=1.5)
        plt.xlabel("Tasa de Falsos Positivos")
        plt.ylabel("Tasa de Verdaderos Positivos")
        plt.title("Curva ROC Multiclase - Regresión Logística")
        plt.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.savefig("Curva_ROC_LogReg.png", dpi=300)
        plt.show()
    else:
        print("⚠️ No se pudo calcular curva ROC: no hay clases válidas con muestras en el test.")

except Exception as e:
    print("❌ Error al generar la curva ROC:", str(e))
