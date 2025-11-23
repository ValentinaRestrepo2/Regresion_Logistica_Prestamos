import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Carga de Datos - Dataset
nombre_archivo = "Loan_Eligibility_1200.csv"
try:
    df = pd.read_csv(nombre_archivo)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{nombre_archivo}'")
    exit()
    
nuevos_nombres = {
    'Gender': 'Genero',
    'Married': 'Casado',
    'Dependents': 'Dependientes',
    'Education': 'Educacion',
    'Self_Employed': 'Autoempleado',
    'Applicant_Income': 'Ingreso_Solicitante',
    'Coapplicant_Income': 'Ingreso_CoSolicitante',
    'Loan_Amount': 'Monto_Prestamo',
    'Loan_Amount_Term': 'Plazo_Prestamo',
    'Credit_History': 'Historial_Credito',
    'Property_Area': 'Area_Propiedad',
    'Loan_Status': 'Estado_Prestamo'
}
# Limpieza y Transformación de Datos
if 'Customer_ID' in df.columns:
    df = df.drop('Customer_ID', axis=1)

df = df.rename(columns=nuevos_nombres)
print(df.columns.tolist())

if 'Dependientes' in df.columns:
    df['Dependientes'] = df['Dependientes'].replace('3+', '3').astype(float)

mappings = {
    'Estado_Prestamo': {'Y': 1, 'N': 0},
    'Genero': {'Male': 1, 'Female': 0},
    'Casado': {'Yes': 1, 'No': 0},
    'Educacion': {'Graduate': 1, 'Not Graduate': 0},
    'Autoempleado': {'Yes': 1, 'No': 0}
}

for col, mapping in mappings.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

# One-Hot Encoding para Area_Propiedad, esto creará columnas como 'Area_Propiedad_Semiurban'
df = pd.get_dummies(df, columns=['Area_Propiedad'], drop_first=True)

df = df.dropna().astype(float)

# Definición de variables X (Predictoras) e y (Objetivo)
X = df.drop('Estado_Prestamo', axis=1)
y = df['Estado_Prestamo']

nombres_columnas = X.columns.tolist()
print(nombres_columnas)

# División del dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Entrenamiento del modelo
modelo = LogisticRegression(max_iter=10000)
modelo.fit(X_train, y_train)

# Realizar predicciones
y_pred = modelo.predict(X_test)

# Intercepto y Coeficientes
intercepto = modelo.intercept_[0]
ce = modelo.coef_[0]

print("\n--- Ecuación del Modelo ---")
ecuacion_str = f"h(x) = {intercepto:.4f}"
for i, nombre in enumerate(nombres_columnas):
    signo = "+" if ce[i] >= 0 else "-" 
    valor_abs = abs(ce[i])
    ecuacion_str += f" {signo} ({valor_abs:.4f} * {nombre})"

print(ecuacion_str)
print("\nSalida final g(h(x)) --- 1 / (1 + e(-z))")

# Matriz de confusión
y_pred = modelo.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
etiquetas = ['Rechazado', 'Aprobado']

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=etiquetas, yticklabels=etiquetas)
plt.title('Matriz de Confusión')
plt.ylabel('Real')
plt.xlabel('Predicho')
plt.show()

print("\n--- Reporte de Clasificación ---")
print(classification_report(y_test, y_pred, target_names=etiquetas))
 
#Medidas

#Accurary (Exactitud)
print (metrics.accuracy_score(y_test,y_pred))
#Presicion (Presición)
print (metrics.precision_score(y_test,y_pred))
#Recall (Exhaustividad)
print (metrics.recall_score(y_test,y_pred))
#F1 Score
print (metrics.f1_score(y_test,y_pred))
#Tasa de Error
print(1 - metrics.accuracy_score(y_test,y_pred))
#Error Cuadrático Medio (MSE)
print(metrics.mean_squared_error(y_test, y_pred))