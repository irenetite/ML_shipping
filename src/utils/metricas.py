
# utils/metricas.py
# funciones auxiliares para la evaluación de modelos

from sklearn.metrics import (accuracy_score,
                            precision_score,
                            recall_score,
                            f1_score,
                            roc_auc_score)

def evaluar_modelo(nombre, modelo, X_test, y_test):
    '''
    evalúa un modelo ya entrenado y devuelve un diccionario con sus métricas    
    Parámetros:
        nombre: nom del modelo (string, para el título)
        modelo: mod de sklearn ya entrenado (con .fit() hecho)
        X_test: varbles predictoras del conjunto de test
        y_test: target real del conjunto de test
    '''
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    metricas = {
        'Modelo'    : nombre,
        'Accuracy'  : accuracy_score(y_test, y_pred),
        'Precision' : precision_score(y_test, y_pred),
        'Recall'    : recall_score(y_test, y_pred),
        'F1-Score'  : f1_score(y_test, y_pred),
        'ROC-AUC'   : roc_auc_score(y_test, y_prob)
    }

    print(f'{nombre}')
    print(f'  Accuracy  : {metricas["Accuracy"]:.1%}')
    print(f'  Precision : {metricas["Precision"]:.1%}')
    print(f'  Recall    : {metricas["Recall"]:.1%} ← CLAVE')
    print(f'  F1-Score  : {metricas["F1-Score"]:.1%}')
    print(f'  ROC-AUC   : {metricas["ROC-AUC"]:.3f}')

    return metricas
