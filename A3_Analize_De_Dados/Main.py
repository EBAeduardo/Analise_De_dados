import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

medicalInsurance = pd.read_csv(r'C:\Users\bianc\Desktop\A3_Analize_De_Dados\BancoDeDados\medical_insurance.csv', keep_default_na= False)
print("primeiras 5 linhas:")
print(medicalInsurance.head())

print(" ")

print(f"Total de linhas: {len(medicalInsurance)}")

print(f"Total de colunas: {len(medicalInsurance.columns)}")

print("\n Nome das colunas:")
print(medicalInsurance.columns.tolist())

print("\n")
print("dataset medical premium")

#tratando dados
medicalInsuranceTratado = pd.get_dummies(medicalInsurance, columns= ['sex', 'region', 'urban_rural', 'education', 'marital_status', 
'employment_status', 'smoker', 'alcohol_freq', 'plan_type', 'network_tier'])

print(medicalInsuranceTratado)

#definindo parametros
y = medicalInsuranceTratado['annual_medical_cost']
x = medicalInsuranceTratado.drop(['annual_medical_cost', 'person_id'], axis=1)

#teste
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(f"Total de dados de treino (x_train): {X_train.shape[0]} linhas")
print(f"Total de dados de teste (x_teste): {X_test.shape[0]} linhas")

#regreção linear
medicalInsuranceTratadoLinear = LinearRegression()

medicalInsuranceTratadoLinear.fit(X_train,y_train)

medicalInsuranceTratadoLinearPreverCustos = medicalInsuranceTratadoLinear.predict(X_test)

r2_regrecaoLinear = r2_score(y_test, medicalInsuranceTratadoLinearPreverCustos)

rmse_regrecaoLinear= np.sqrt(mean_squared_error(y_test, medicalInsuranceTratadoLinearPreverCustos))

print("\n------Resutado Regreção linear------\n")
print(f"Quão bem o modelo explica o custo (quanto mais próximo de 1 melhor): {r2_regrecaoLinear:.4f}")
#esse erro é o valor que ele erra dos gastos tanto pra mais quanto pra menos na regreção linear
print(f"Erro médio (quanto menor melhor): $ {rmse_regrecaoLinear:.2f}")

mediaCustoTeste = y_test.mean()
print(f"(Contexto: O custo médio real nos dados de teste é: $ {mediaCustoTeste:.2f})")



#random forest
medicalInsuranceTratadoRandomForest = RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=-1)

medicalInsuranceTratadoRandomForest.fit(X_train, y_train)
medicalInsuranceTratadoRandomForestPreverCusto = medicalInsuranceTratadoRandomForest.predict(X_test)

r2_randomForest = r2_score(y_test, medicalInsuranceTratadoRandomForestPreverCusto)
rmse_randomForest = np.sqrt(mean_squared_error(y_test, medicalInsuranceTratadoRandomForestPreverCusto))

print("\n------Resutado Random Forest------\n")

print(f"Quão bem o modelo explica o custo (R²): {r2_randomForest:.4f}")
print(f"Erro médio (RMSE): $ {r2_randomForest:.2f}")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


medicalInsuranceTratadoRedesNeurais = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, early_stopping=True)

medicalInsuranceTratadoRedesNeurais.fit(X_train_scaled, y_train)

medicalInsuranceTratadoRedesNeuraisPreverCusto = medicalInsuranceTratadoRedesNeurais.predict(X_test_scaled)

r2_redesNeurais = r2_score(y_test, medicalInsuranceTratadoRedesNeuraisPreverCusto)
rmse_redesNeurais = np.sqrt(mean_squared_error(y_test, medicalInsuranceTratadoRedesNeuraisPreverCusto))


print("\n------Resultado Rede Neural (MLP)------\n")
print(f"Quão bem o modelo explica o custo (R²): {r2_redesNeurais:.4f}")
print(f"Erro médio (RMSE): $ {rmse_redesNeurais:.2f}")


print("\n----comparação entre eles----\n")

print(f"Regressão Linear (Modelo 1): $ {rmse_regrecaoLinear:.2f}")
print(f"Random Forest    (Modelo 2): $ {rmse_randomForest:.2f}")
print(f"Rede Neural      (Modelo 3): $ {rmse_redesNeurais:.2f}")