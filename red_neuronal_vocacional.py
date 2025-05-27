
import pandas as pd
import mysql.connector
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import dataframe_image as dfi
import os
import numpy as np
import pandas as pd
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
y_categorical = to_categorical(y_encoded)

# ===================================
# 3. MODELO Y ENTRENAMIENTO
# ===================================
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, random_state=42)

model = Sequential()
model.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_categorical.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Guardar el historial del entrenamiento
historia = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=0)

# ===================================
# 4. GRÁFICO DE PRECISIÓN POR ÉPOCA
# ===================================
plt.figure(figsize=(10, 6))
plt.plot(historia.history['accuracy'], label='Entrenamiento', linewidth=2)
plt.plot(historia.history['val_accuracy'], label='Validación', linewidth=2)
plt.title("Precisión por época - Red Neuronal")
plt.xlabel("Época")
plt.ylabel("Precisión")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("precision_por_epoca.png", dpi=300)
plt.show()


# --- Crear tabla de comparación (si no la tienes aún) ---
predicciones = model.predict(X_scaled)
indices_predichos = np.argmax(predicciones, axis=1)
carreras_predichas = label_encoder.inverse_transform(indices_predichos)

df_resultado = df[["Edad", "Grado", "Carrera"]].copy()
df_resultado["Carrera_Predicha_RN"] = carreras_predichas
df_resultado["Coincide"] = df_resultado["Carrera"] == df_resultado["Carrera_Predicha_RN"]

# --- Contar aciertos y errores ---
conteo = df_resultado["Coincide"].value_counts()
conteo.index = ["Correctas (True)", "Incorrectas (False)"]

# --- Crear gráfico ---
plt.figure(figsize=(8, 5))
conteo.plot(kind="bar", color=["green", "red"])
plt.title("Comparación de Predicciones - Red Neuronal")
plt.ylabel("Cantidad de Estudiantes")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("Precision_Coincidencia_Barras.png", dpi=300)
plt.show()

# ===================================
# 5. CURVA ROC MULTICLASE - RED NEURONAL
# ===================================
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# Volvemos a binarizar los labels reales
y_test_encoded = np.argmax(y_test, axis=1)
y_test_bin = label_binarize(y_test_encoded, classes=np.arange(y_categorical.shape[1]))

try:
    y_score = model.predict(X_test)
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

        plt.plot(all_fpr, mean_tpr, "--", color="darkorange", label=f"Promedio Macro (AUC = {macro_auc:.2f})", lw=2)
        plt.plot([0, 1], [0, 1], "k--", lw=1.5)
        plt.xlabel("Tasa de Falsos Positivos")
        plt.ylabel("Tasa de Verdaderos Positivos")
        plt.title("Curva ROC Multiclase - Red Neuronal")
        plt.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.savefig("Curva_ROC_RN.png", dpi=300)
        plt.show()
    else:
        print("⚠️ No se pudo generar la curva ROC: no hay clases válidas en test.")

except Exception as e:
    print("❌ Error al generar la curva ROC:", str(e))
