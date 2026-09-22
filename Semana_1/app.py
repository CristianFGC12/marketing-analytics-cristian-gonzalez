# Archivo base para el despliegue del Agente en Streamlit
#Librerias
import streamlit as st 
from sklearn.linear_model import LinearRegression 
import numpy as np 

#En streamlit vamos a agregar un titulo a la pagina
st.title("Configuracion inicial") 
# Agregamos un textbox para la pagina
st.write("Primera prueba de uso de streamlit y ambiente de MA2026") 

#Silder para ingresar parametro de inversion
gasto=st.slider("Seleccine nivel de gasto en publicicdad", 10,200,50) 

#Variables
variable_x = np.array([[10], [20], [30], [40],[50]]) 
variable_y = np.array([15,25,35,45,55]) 
modelo_lr = LinearRegression() 

#Entrenamiento
modelo_lr.fit(variable_x,variable_y) 
#Boton que dice Producir y al dar click activa el if
if st.button("Predecir"):
     resultado = modelo_lr.predict([[gasto]]) 
     #Streamlit muestra mensaje de exito
     st.success(f"Las ventas proyectadas para una inversion de ${gasto} son: ${resultado[0]}") 
