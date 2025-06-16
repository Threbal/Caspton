
import pandas as pd
import mysql.connector
import statsmodels.api as sm
from scipy.stats import skew, kurtosis
from statsmodels.stats.stattools import omni_normtest, jarque_bera

# === Conexión a la base de datos MySQL ===
conn = mysql.connector.connect(
    host="localhost",
    port=3306,
    user="root",
    password="root",
    database="test_vocacional"
)

# Cargar datos
columnas = "Edad, Grado, " + ", ".join([f"P{i}" for i in range(1, 31)]) + ", Carrera"
df = pd.read_sql(f"SELECT {columnas} FROM respuestas", conn)
conn.close()

# Preprocesamiento
grado_map = {
    "Tercero de Secundaria": 3,
    "Cuarto de Secundaria": 4,
    "Quinto de Secundaria": 5
}
df["GradoNum"] = df["Grado"].map(grado_map)
df["CarreraCod"] = df["Carrera"].astype("category").cat.codes
df = df.drop(columns=["Grado", "Carrera"])
df = df.dropna()

# Variables independientes (X) y dependiente (y)
X = df.drop(columns=["CarreraCod"])
X = sm.add_constant(X)
y = df["CarreraCod"]

# Entrenar modelo OLS
modelo = sm.OLS(y, X).fit()

# Obtener residuos
residuos = modelo.resid

# Calcular estadísticos
omni_stat, omni_pval = omni_normtest(residuos)
jb_stat, jb_pval, _, _ = jarque_bera(residuos)
dw = sm.stats.stattools.durbin_watson(residuos)
skw = skew(residuos)
krt = kurtosis(residuos)
cond_number = modelo.condition_number

# Imprimir resultados
print("\n=== Tabla 3: Análisis estadístico del modelo ===")
print(f"Omnibus: {omni_stat:.3f}")
print(f"Prob(Omnibus): {omni_pval:.3f}")
print(f"Skew (Asimetría): {skw:.3f}")
print(f"Kurtosis: {krt:.3f}")
print(f"Durbin-Watson: {dw:.3f}")
print(f"Jarque-Bera (JB): {jb_stat:.3f}")
print(f"Prob(JB): {jb_pval:.2e}")
print(f"Cond. No.: {cond_number:.2e}")
