# Análisis predictivo de la deserción por fichas de formación - CIMM

Este repositorio contiene el código, los datos depurados, los resultados y el tablero de Power BI del trabajo de grado para identificar las fichas de formación del Centro Industrial de Mantenimiento y Manufactura (CIMM), Regional Boyacá del SENA, con mayor probabilidad de presentar alta deserción.

El modelo es transversal y de carácter asociativo-predictivo: estima asociaciones entre las características de la ficha y la deserción, no efectos causales. La unidad de análisis es la ficha de formación, no el aprendiz.

## Datos y ventana de observación

- Fuente: base de deserción de la formación profesional integral del SENA, publicada en datos.gov.co (conjunto `u4ze-bi7k`).
- Última actualización de los datos: 14 de marzo de 2024. La ventana de observación va del 1 de enero al 14 de marzo de 2024.
- Muestra de modelado: 350 fichas, de las cuales 31 (8,86 %) superan el umbral institucional de deserción de 8,41 %.
- Limitación: la ficha técnica del conjunto no describe el periodo que acumula la columna `DESERTORES_AÑO_ACTUAL`. El trabajo asume que acumula los desertores del año en curso hasta la fecha de corte.

## Estructura del repositorio

| Carpeta | Contenido |
|---|---|
| `data/` | `resultados_predicciones_cimm.csv`: datos depurados por ficha, entrada del modelado. No contiene predicciones. |
| `notebooks/` | Cuaderno de preparación de los datos (Google Colab) y `2_modelado_trazable.py`, que genera las cifras de los capítulos 4 y 5 del documento, y 3_figuras.py, que genera las Figuras 3 y 4 en figuras/. |
| `results/trazable/` | Salidas del script: métricas por pliegue, resumen de modelos, hiperparámetros, importancia por permutación, análisis de robustez y de sensibilidad, y `resultados_para_powerbi.csv`. |
| `dashboard/` | Tablero de Power BI (`.pbix`), alimentado por `results/trazable/resultados_para_powerbi.csv`. |
| `figuras/` | Figuras 3 y 4 del documento, generadas por notebooks/3_figuras.py. Pueden variar levemente con la versión de scikit-learn. |

## Cómo reproducir los resultados

Requisitos: Python 3.12 o superior, con `pandas`, `numpy`, `scipy` y `scikit-learn`.

```bash
git clone https://github.com/yohanal86/Proyecto-Aplicado-CIMM.git
cd Proyecto-Aplicado-CIMM
pip install pandas numpy scipy scikit-learn
```

El script se ejecuta desde la raíz del repositorio y escribe en `results/trazable/`:

```bash
python notebooks/2_modelado_trazable.py main     # análisis principal (50 pliegues externos)
python notebooks/2_modelado_trazable.py grupos   # robustez: validación agrupada por programa
python notebooks/2_modelado_trazable.py sens     # sensibilidad: umbral de 12 % y fichas completas
python notebooks/2_modelado_trazable.py --quick main   # prueba rápida (1 repetición)
```

## Diseño de la evaluación

- Validación cruzada anidada: lazo externo `RepeatedStratifiedKFold` (5 pliegues, 10 repeticiones, 50 pliegues externos) y lazo interno de 5 pliegues con `GridSearchCV`, optimizando la Precisión Media (AP).
- Umbral de decisión: elegido en el lazo interno con el criterio de máximo F2.
- Desbalanceo: `class_weight='balanced'` en Regresión Logística, Árbol de Decisión y Random Forest; `sample_weight` balanceado en Gradient Boosting.
- Comparación entre modelos: intervalos de confianza y prueba t corregida de Nadeau y Bengio.
- Robustez: validación agrupada por programa (123 programas) y sensibilidad con umbral de 12 %.
- Interpretación: importancia por permutación y dependencia parcial.

## Resultados principales

Métricas de los 50 pliegues externos. AP: Precisión Media, con intervalo de confianza del 95 % (Nadeau y Bengio). Sensibilidad, precisión y F2: fuera de pliegue, con el umbral del lazo interno.

| Modelo | AP (IC 95 %) | ROC-AUC | Brier | Sensibilidad | Precisión | F2 |
|---|---|---|---|---|---|---|
| Clasificador de referencia (Dummy) | 0,100 (0,079-0,121) | 0,500 | 0,183 | 11,3 % | 8,8 % | 0,107 |
| Regresión Logística | 0,326 (0,188-0,465) | 0,790 | 0,191 | 74,2 % | 18,1 % | 0,456 |
| Árbol de Decisión | 0,390 (0,245-0,534) | 0,793 | 0,137 | 65,2 % | 34,3 % | 0,550 |
| Random Forest | 0,491 (0,340-0,641) | 0,852 | 0,090 | 63,2 % | 42,2 % | 0,571 |
| Gradient Boosting | 0,484 (0,323-0,644) | 0,846 | 0,101 | 60,6 % | 42,8 % | 0,555 |

- Random Forest y Gradient Boosting superan al clasificador de referencia. La diferencia entre ellos no es estadísticamente significativa (p = 0,88). Se adopta Random Forest como modelo de referencia por estabilidad y parsimonia.
- La precisión de cerca del 42 % indica que más de la mitad de las fichas señaladas no son críticas. El modelo sirve para priorizar la revisión, no para decidir sobre una ficha.
- La duración planeada y los días de exposición tienen la mayor importancia por permutación. Los resultados son asociaciones predictivas y no establecen causalidad.
- La ventana de observación es corta: 251 de las 350 fichas terminaban después de la fecha de corte. Parte de la diferencia entre niveles puede deberse a la ventana y no al nivel.
- Nivel y modalidad están entrelazados (61 de las 75 fichas virtuales son Cursos Especiales; V de Cramér = 0,44).

## Limitaciones

- No hay datos de cierre de 2024, por lo que el modelo no se validó en el tiempo.
- Solo 31 fichas críticas y fichas no independientes (123 programas): los intervalos de confianza son amplios.
- Pendiente como trabajo futuro: validación externa con las 42.080 fichas del archivo nacional e integración de las bases de cupos y de certificación.
