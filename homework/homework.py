# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
# flake8: noqa: E501

import pandas as pd
import os
import gzip
import pickle
import json
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix

def cargar_datos():

    df_train=pd.read_csv("files/input/train_data.csv.zip")
    df_test=pd.read_csv("files/input/test_data.csv.zip")
    return df_train, df_test

def limpieza(df):
    df=df.copy()
    df=df.rename(columns={"default payment next month":"default"})
    df=df.drop(columns=["ID"], errors="ignore")
    df=df.dropna()
    df=df[(df["EDUCATION"]>0) & (df["MARRIAGE"]>0)]
    df.loc[df["EDUCATION"]>=4, "EDUCATION"]=4
    
    return df

def separar_datos(base):
    base=base.copy()
    x=base
    y=base.pop("default")
    return x,y

def hacer_pipeline(estimador):
    columnas_categoricas=["SEX","EDUCATION","MARRIAGE"]

    preproceso=ColumnTransformer(
        transformers=[
            ("ohe",
             OneHotEncoder(handle_unknown="ignore"),
             columnas_categoricas),
        ],
        remainder=StandardScaler()
    )

    seleccion=SelectKBest(score_func=f_classif)
    pipeline=Pipeline(
        steps=[
            ("preproceso",preproceso),
            ("seleccion",seleccion),
            ("pca",PCA(n_components=None)),
            ("estimador",estimador)
        ]
    )

    return pipeline

def make_grid_search(estimador, param_grid, cv=10):

    grid_search = GridSearchCV(
        estimator=estimador,
        param_grid=param_grid,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=1
    )

    return grid_search

def train_mlp(x_train, y_train, x_test, y_test):
    mlp=MLPClassifier(random_state=42,max_iter=15000)

    pipeline = hacer_pipeline(estimador=mlp)

    param_grid = {
        "pca__n_components":[None],
        "seleccion__k": [20],      
        "estimador__hidden_layer_sizes":[(50,30,40,60)],
        "estimador__alpha": [0.28],
        "estimador__learning_rate_init":[0.001]
    }


    estimador = make_grid_search(
        estimador=pipeline,
        param_grid=param_grid,
        cv=10
    )
    estimador.fit(x_train, y_train)

    return estimador

def calculate_metrics(model, X, y, dataset_type):

    y_pred = model.predict(X)
    
    return {
        'type': 'metrics',
        'dataset': dataset_type,
        'precision': precision_score(y, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1_score': f1_score(y, y_pred)
    }

def calculate_confusion_matrix(model, X, y, dataset_type):
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    
    return {
        'type': 'cm_matrix',
        'dataset': dataset_type,
        'true_0': {
            "predicted_0": int(cm[0,0]), 
            "predicted_1": int(cm[0,1])
        },
        'true_1': {
            "predicted_0": int(cm[1,0]), 
            "predicted_1": int(cm[1,1])
        }
    }

def guardar_modelo(modelo):
    carpeta="files/models"
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)

    nombre = os.path.join(carpeta, "model.pkl.gz")
    with gzip.open(nombre, 'wb') as f:
        pickle.dump(modelo, f)

def guardar_metricas(model, x_train, y_train, x_test, y_test):

    m_train = calculate_metrics(model, x_train, y_train, 'train')
    m_test  = calculate_metrics(model, x_test, y_test, 'test')
    
    cm_train = calculate_confusion_matrix(model, x_train, y_train, 'train')
    cm_test  = calculate_confusion_matrix(model, x_test, y_test, 'test')
    
    resultados = [m_train, m_test, cm_train, cm_test]
    
    carpeta = "files/output"
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)
        
    ruta_archivo = os.path.join(carpeta, "metrics.json")
    
    with open(ruta_archivo, "w") as f:
        for item in resultados:
            f.write(json.dumps(item) + "\n")

if __name__=="__main__":
    base_train,base_test=cargar_datos()
    base_train=limpieza(base_train)
    base_test=limpieza(base_test)
    x_train,y_train=separar_datos(base_train)
    x_test,y_test=separar_datos(base_test)
    estimator=train_mlp(x_train,y_train,x_test,y_test)

    guardar_modelo(estimator)
    guardar_metricas(estimator,x_train,y_train,x_test,y_test)