# Análisis Predictivo de la Deserción por Fichas de Formación - CIMM

Este repositorio aloja el ecosistema analítico, el pipeline de preprocesamiento, los modelos de aprendizaje supervisado y la solución de inteligencia de negocios desarrollados para identificar de forma temprana las fichas de formación con tendencia a registrar altos índices de deserción en el **Centro Industrial de Mantenimiento y Manufactura (CIMM)**, Regional Boyacá del SENA.

## Estructura del Ecosistema de Reproducibilidad

La investigación se ejecutó siguiendo rigurosamente las fases de la metodología **CRISP-DM**, organizando los componentes en los siguientes directorios para garantizar la auditoría y replicabilidad total del proyecto:

*   `data/`: Registros institucionales depurados y consolidados a nivel de fichas de formación (350 registros definitivos de la vigencia 2024).
*   `notebooks/`: Código fuente estructurado en Python conteniendo la carga limpia con marcadores de vacío, limpieza de comillas y el Pipeline de Scikit-Learn para el control de fuga de información (*data leakage*).
*   `results/`: Archivos independientes en formato `.csv` que contienen las métricas de rendimiento obtenidas en cada uno de los pliegues de la validación cruzada anidada, sustentando la validez estadística del estudio.
*   `dashboard/`: Archivo ejecutable de Power BI (`.pbix`) que contiene el cuadro de mando interactivo con alcance operativo de monitoreo y diagnóstico de la oferta formativa del CIMM.

##  Requisitos e Instalación

Para replicar el entorno experimental y ejecutar los scripts locales, clone este repositorio e instale las dependencias tecnológicas ejecutando los siguientes comandos en su consola de desarrollo:

```bash
# Clonar el repositorio público
git clone https://github.com

# Acceder al directorio raíz del proyecto
cd Proyecto-Tesis-CIMM

# Instalar el stack tecnológico requerido
pip install -r requirements.txt
```

### Especificación del Stack Tecnológico (Python 3.12+)
*   **Pandas & NumPy:** Manipulación de matrices y normalización de texto.
*   **Scikit-Learn:** Arquitectura de pipelines, transformadores de columnas y algoritmos de clasificación.
*   **Matplotlib & Seaborn:** Generación de gráficos analíticos vectoriales en alta resolución y libres de comillas.

## Resumen de Rendimiento y Hallazgos

Ante el desbalanceo severo de la clase crítica (8.78% de fichas en alta deserción), se implementó un esquema de **Validación Cruzada Anidada** (`RepeatedStratifiedKFold` con 5 pliegues y 10 repeticiones externas) optimizado bajo la métrica primaria de **Precisión Media (AP)**. El algoritmo secuencial **Gradient Boosting** se consolidó como el modelo definitivo ganador del estudio:

| Algoritmo Evaluado | Precisión Media (AP Medio) | Desviación Estándar (σ) |
| :--- | :---: | :---: |
| Clasificador Base (*Dummy*) | 0.1000 | 0.0198 |
| Regresión Logística | 0.3498 | 0.1476 |
| Árbol de Decisión | 0.3573 | 0.1227 |
| *Random Forest* | 0.4568 | 0.1401 |
| **Gradient Boosting** | **0.4763** | **0.1423** |

El análisis de relevancia de variables (*Gini Importance*) determinó que la **Modalidad de Formación Virtual** (peso: 0.342) y el nivel de **Curso Especial** (peso: 0.268) explican de forma conjunta el **61.0%** de la capacidad predictiva del ensamble. Estos resultados validan empíricamente la adaptación del *Modelo de Integración Estudiantil de Vincent Tinto* a la escala de entornos colectivos, demostrando que la falta de interacción física y los periodos cortos de ejecución debilitan la cohesión social y el compromiso institucional del grupo, acelerando la deserción masiva.
