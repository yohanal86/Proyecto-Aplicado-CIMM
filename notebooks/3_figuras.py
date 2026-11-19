# 3. Figuras 3 y 4 del documento (importancia por permutación y dependencia parcial)
# Se ejecuta desde la raíz del repositorio, después de 2_modelado_trazable.py:
#   python notebooks/3_figuras.py
# Entradas: results/trazable/importancia_permutacion.csv, resultados_para_powerbi.csv y modelo_final.json
# Salida: carpeta figuras/

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

R = 'results/trazable/'
SALIDA = 'figuras/'
os.makedirs(SALIDA, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'font.family': 'DejaVu Sans'})

# ---- Figura 3. Importancia por permutación (media ± DE de los 50 pliegues externos)
p = pd.read_csv(R + 'importancia_permutacion.csv')
ETIQUETAS = {'DURACION_PLANEADA_DIAS': 'Duración planeada (días)',
             'DIAS_EXPOSICION_CORTE': 'Días de exposición',
             'TOTAL_APRENDICES_MATRICULADOS': 'Aprendices matriculados',
             'MODALIDAD_FORMACION': 'Modalidad de formación',
             'NIVEL_FORMACION': 'Nivel de formación'}
ORDEN = list(ETIQUETAS)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
for ax, m, color, titulo in zip(axs, ['Random_Forest', 'Gradient_Boosting'], ['#1f5f8b', '#c9722b'],
                                ['Random Forest', 'Gradient Boosting']):
    q = p[p.modelo == m].set_index('variable').loc[ORDEN]
    y = np.arange(len(ORDEN))[::-1]
    ax.barh(y, q['mean'], xerr=q['std'], color=color, alpha=.9,
            error_kw=dict(ecolor='#555', lw=1, capsize=3))
    ax.axvline(0, color='#333', lw=.8)
    ax.set_title(titulo, fontsize=11)
    ax.set_yticks(y)
    ax.set_yticklabels([ETIQUETAS[v] for v in ORDEN])
    ax.set_xlabel('Caída del AP al permutar la variable')
    for yy, v in zip(y, q['mean']):
        ax.text((max(v, 0) + 0.003) if v >= 0 else (v - 0.003), yy + 0.34,
                f'{v:.3f}'.replace('.', ',').replace('-', '−'), fontsize=8, color='#222', va='center',
                ha='left' if v >= 0 else 'right')
plt.tight_layout()
plt.savefig(SALIDA + 'fig3_importancia_permutacion.png', dpi=200)
plt.close()

# ---- Figura 4. Dependencia parcial de Random Forest (reajustado con las 350 fichas)
d = pd.read_csv(R + 'resultados_para_powerbi.csv')
NUM = ['TOTAL_APRENDICES_MATRICULADOS', 'DURACION_PLANEADA_DIAS', 'DIAS_EXPOSICION_CORTE']
CAT = ['NIVEL_FORMACION', 'MODALIDAD_FORMACION']
X = d[NUM + CAT].copy()
X[NUM] = X[NUM].astype(float)
y = d['ALTA_DESERCION']
params = json.load(open(R + 'modelo_final.json'))['Random_Forest']['best_params']
pre = ColumnTransformer([('num', StandardScaler(), NUM),
                         ('cat', OneHotEncoder(handle_unknown='ignore'), CAT)])
pipe = Pipeline([('pre', pre),
                 ('model', RandomForestClassifier(class_weight='balanced', random_state=42, **params))]).fit(X, y)

NOMBRES = {'TOTAL_APRENDICES_MATRICULADOS': 'Aprendices matriculados', 'DURACION_PLANEADA_DIAS': 'Duración planeada (días)',
           'DIAS_EXPOSICION_CORTE': 'Días de exposición', 'NIVEL_FORMACION': 'Nivel de formación',
           'MODALIDAD_FORMACION': 'Modalidad de formación'}
CATEGORIAS = {'CURSO ESPECIAL': 'Curso Especial', 'TECNICO': 'Técnico', 'TECNOLOGO': 'Tecnólogo',
              'PRESENCIAL': 'Presencial', 'VIRTUAL': 'Virtual'}
def pdp_manual(modelo, X, v, categorica):
    """Dependencia parcial: probabilidad media predicha al fijar la variable v en cada valor de la rejilla."""
    if categorica:
        grid = sorted(X[v].unique())
    else:
        u = np.unique(X[v])
        grid = u if len(u) < 100 else np.linspace(*np.percentile(X[v], [5, 95]), 100)
    medias = []
    for g in grid:
        Xg = X.copy()
        Xg[v] = g
        medias.append(modelo.predict_proba(Xg)[:, 1].mean())
    return np.asarray(grid), np.asarray(medias)
fig, axs = plt.subplots(2, 3, figsize=(10, 6))
axs = axs.ravel()
for ax, v in zip(axs, NUM + CAT):
    g, a = pdp_manual(pipe, X, v, v in CAT)
    if v in CAT:
        ax.bar([CATEGORIAS[str(t)] for t in g], a, color='#1f5f8b', width=.55)
        ax.set_ylim(0, max(a) * 1.35)
        for i, vv in enumerate(a):
            ax.text(i, vv + max(a) * .03, f'{vv:.2f}'.replace('.', ','), ha='center', fontsize=8)
    else:
        ax.plot(g, a, color='#1f5f8b', lw=1.8)
    ax.set_title(NOMBRES[v], fontsize=10)
    ax.set_ylabel('Prob. media predicha')
axs[5].axis('off')
plt.tight_layout()
plt.savefig(SALIDA + 'fig4_dependencia_parcial.png', dpi=200)
plt.close()
print('Figuras guardadas en', SALIDA)
