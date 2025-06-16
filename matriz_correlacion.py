
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mysql.connector

# Conexión a la base de datos
conn = mysql.connector.connect(
    host="localhost",
    port=3306,
    user="root",
    password="root",
    database="test_vocacional"
)

# Cargar datos relevantes
columnas = "Edad, Grado, " + ", ".join([f"P{i}" for i in range(1, 31)])
df = pd.read_sql(f"SELECT {columnas} FROM respuestas", conn)
conn.close()

# Mapear grados a número
grado_map = {
    "Tercero de Secundaria": 3,
    "Cuarto de Secundaria": 4,
    "Quinto de Secundaria": 5
}
df["GradoNum"] = df["Grado"].map(grado_map)
df = df.drop(columns=["Grado"])
df = df.dropna()

# Calcular matriz de correlación
corr = df.corr()

# Generar heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, linewidths=0.5)
plt.title("Fig. 9. Matriz de Correlación - Variables del Test Vocacional", fontsize=14)
plt.tight_layout()
plt.savefig("Matriz_Correlacion_Test_Vocacional.png", dpi=300)
plt.show()
