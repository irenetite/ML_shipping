
# utils/preprocesamiento.py
# funciones auxiliares para el preprocesamiento del dataset


import pandas as pd


def cargar_y_limpiar(ruta_csv):
    '''
    carga el dataset original y aplica el preprocesamiento base:
    - elimina el ID (identificador sin valor predictivo)
    - codifica Product_importance con Ordinal Encoding
      (low=1, medium=2, high=3) preservando la jerarquía
    parámetros: ruta_csv (ruta al archivo shipping_data.csv)
    devuelve: df preprocesado listo para EDA o modelado
    '''
    df = pd.read_csv(ruta_csv)

    # elimino el ID: es un identificador único sin valor predictivo
    df = df.drop(columns=['ID'])

    # ordinal encoding para Product_importance
    # tiene un orden lógico (low < medium < high) así que mapeamos a 1, 2, 3
    mapa_imp = {'low': 1, 'medium': 2, 'high': 3}
    df['Product_importance'] = df['Product_importance'].map(mapa_imp)

    return df