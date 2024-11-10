##  Formulario Examen Final

### Chapter Outline

- TEMA 1: Configuración y Visualización del Entorno
    - 1.1 Configuración de las Rutas de Importación
    - 1.2 Verificación de las Versiones de los Paquetes
    - 1.3 Visualización de Imágenes
    - 1.4 Importación de Paquetes
- TEMA 2: Análisis Exploratorio de Datos (Compresión de Datos y Reducción Dimensional)
    - 2.1 Carga y Exploración Inicial del Dataset
    - 2.2 Anonimización y Cálculo de la Correlación entre Características
    - 2.3 División de Variables Independientes y Dependiente
    - 2.4 Mapa de Calor de Correlaciones
    - 2.5 Histogramas de Distribución de las Características
- TEMA 3: Métodos de Compresión de Datos y Reducción Dimensional
    - 3.1 Reducción Dimensional No Supervisada mediante Análisis de Componentes Principales (PCA)
        - 3.1.1 Paso 1: Estandarización del Conjunto de Datos D-Dimensional
        - 3.1.2 Paso 2: Construcción de la Matriz de Covarianza
        - 3.1.3 Paso 3: Descomposición de la Matriz de Covarianza en Vectores y Valores Propios
        - 3.1.4 Paso 4: Ordenación de los Valores Propios en Orden Decreciente
        - 3.1.5 Paso 5: Selección de k Vectores Propios correspondientes a los k Valores Propios Más Grandes
        - 3.1.6 Paso 6: Contrucción de la Matriz de Proyección W
        - 3.1.7 Paso 7: Transformación del Dataset mediante la Matriz de Proyección W
        - 3.1.8 Visualización del Nuevo Espacio de Características
        - 3.1.9 Clasificación y Visualización de las Regiones de Decisión
        - 3.1.10 Explicación de la Varianza Total
        - 3.1.11 Carga de los Componentes Principales
    - 3.2 Compresión de Datos Supervisada mediante Análisis Discriminante Lineal (LDA)
        - 3.2.1 Paso 1: Estandarización del Conjunto de Datos D-Dimensional
        - 3.2.2 Paso 2: Cálculo del Vector Medio D-Dimensional para cada Clase
        - 3.2.3 Paso 3: Construcción de las Matrices de Dispersión dentro de clases (S_W) y entre Clases (S_B)
        - 3.2.4 Paso 4: Cálculo de Vectores y Valores Propios de (S_W)^-1 * S_B
        - 3.2.5 Paso 5: Ordenación de los Valores Propios en Orden Decreciente
        - 3.2.6 Paso 6: Selección de los k Vectores Propios Más Grandes para Contruir la Matriz de Tranformación W
        - 3.2.7 Paso 7: Proyección de Ejemplos en el Nuevo Subespacio usando la Matriz de Transformación W
        - 3.2.8 Clasificación y Visualización de las Regiones de Decisión en el Subespacio LDA
    - 3.3 Técnicas de Reducción Dimensional No Lineal](#33-técnicas-de-reducción-dimensional-no-lineal)
        - 3.3.1 Carga y Visualización de Imágenes de Dígitos
        - 3.3.2 Obtención de Dimensiones del Dataset y Separación de Características y Etiquetas
        - 3.3.3 Aplicación de t-SNE para Reducción Dimensional No Lineal
        - 3.3.4 Definición y Aplicación de la Función de Visualización
- ANEXO: Convertir Jupyter Notebook a Fichero Python
    - A.1 Script en el Directorio Actual
    - A.2 Script en el Directorio Padre

**Please refer to the [README.md](../ch01/README.md) file in [`../ch01`](../ch01) for more information about running the code examples.**
