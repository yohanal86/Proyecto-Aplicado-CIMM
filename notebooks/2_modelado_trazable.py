# %% [markdown]
# # 2. Modelado trazable CIMM (validación anidada, umbral, permutación, robustez)
# Este script produce TODAS las cifras de los capítulos 4 y 5. Se ejecuta desde la raíz del repositorio:
#   python notebooks/2_modelado_trazable.py main      # análisis principal (50 pliegues externos)
#   python notebooks/2_modelado_trazable.py grupos    # robustez: StratifiedGroupKFold por programa
#   python notebooks/2_modelado_trazable.py sens      # sensibilidad: umbral 12 % y fichas completas
#   python notebooks/2_modelado_trazable.py --quick ...  # prueba rápida (1 repetición)
# Entrada: data/resultados_predicciones_cimm.csv (derivado de la celda 3-4 del cuaderno 1).
# Salida: carpeta results/trazable/

# %%
import sys, json, time, warnings, os, pickle
from collections import Counter
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     StratifiedGroupKFold, GridSearchCV, cross_val_predict)
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             fbeta_score, precision_score, recall_score, confusion_matrix)
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_sample_weight
warnings.filterwarnings('ignore')

ARGS = [a for a in sys.argv[1:] if not a.startswith('--')]
QUICK = '--quick' in sys.argv
ETAPA = ARGS[0] if ARGS else 'main'
SALIDA = 'results/trazable' + ('_quick' if QUICK else '')
os.makedirs(SALIDA, exist_ok=True)

# %% [markdown]
# ## Datos y ventana de observación
# Fuente: datos.gov.co (u4ze-bi7k). Última actualización de los datos: 14-mar-2024.
# La ventana de exposición es 1-ene-2024 a 14-mar-2024 (o hasta la terminación planeada, si es anterior).

# %%
FECHA_CORTE = pd.Timestamp('2024-03-14')
INICIO_VENTANA = pd.Timestamp('2024-01-01')
df = pd.read_csv('data/resultados_predicciones_cimm.csv',
                 parse_dates=['FECHA_INICIO_FICHA', 'FECHA_TERMINACION_FICHA'])
a = df['FECHA_INICIO_FICHA'].clip(lower=INICIO_VENTANA)
b = df['FECHA_TERMINACION_FICHA'].clip(upper=FECHA_CORTE)
df['DIAS_EXPOSICION_CORTE'] = (b - a).dt.days.clip(lower=0)

NUM = ['TOTAL_APRENDICES_MATRICULADOS', 'DURACION_PLANEADA_DIAS', 'DIAS_EXPOSICION_CORTE']
CAT = ['NIVEL_FORMACION', 'MODALIDAD_FORMACION']
VARS = NUM + CAT
X_all = df[VARS].copy()

def preproc():
    return ColumnTransformer([('num', StandardScaler(), NUM),
                              ('cat', OneHotEncoder(handle_unknown='ignore'), CAT)])

# %% [markdown]
# ## Modelos y grillas (idénticas a las del cuaderno 1; Gradient Boosting ahora usa sample_weight balanceado)

# %%
def modelos_config(quick=False):
    C = [0.01, 0.1, 1.0, 10.0]; D = [3, 5, 7]; S = [2, 5, 10]
    E = [50, 100, 200]; LR = [0.01, 0.05, 0.1]
    if quick:
        C, D, S, E, LR = [0.1, 1.0], [3, 5], [2, 5], [50], [0.1]
    return {
        'Dummy': dict(model=DummyClassifier(strategy='stratified', random_state=42), params={}, sw=False),
        'Regresion_Logistica': dict(model=LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42),
                                    params={'model__C': C}, sw=False),
        'Arbol_Decision': dict(model=DecisionTreeClassifier(class_weight='balanced', random_state=42),
                               params={'model__max_depth': D, 'model__min_samples_split': S}, sw=False),
        'Random_Forest': dict(model=RandomForestClassifier(class_weight='balanced', random_state=42),
                              params={'model__n_estimators': E, 'model__max_depth': D}, sw=False),
        'Gradient_Boosting': dict(model=GradientBoostingClassifier(random_state=42),
                                  params={'model__n_estimators': E, 'model__learning_rate': LR, 'model__max_depth': D}, sw=True),
    }

# %% [markdown]
# ## Validación anidada con umbral elegido en el lazo interno (máx. F2)

# %%
UMBRALES = np.round(np.arange(0.02, 0.99, 0.01), 2)

def elegir_umbral(y, p):
    mejor, mejor_f = 0.5, -1
    for u in UMBRALES:
        f = fbeta_score(y, (p >= u).astype(int), beta=2, zero_division=0)
        if f > mejor_f + 1e-12:
            mejor, mejor_f = u, f
    return float(mejor)

def fit_params(cfg, ytr):
    return {'model__sample_weight': compute_sample_weight('balanced', ytr)} if cfg['sw'] else {}

def correr_anidado(X, y, splits, modelos, groups=None, permutacion=(), etiqueta=''):
    """splits: lista de (repeticion, idx_train, idx_test)."""
    filas, preds, hparams, perms = [], [], [], []
    t0 = time.time()
    ck = f'{SALIDA}/_checkpoint_{etiqueta}.pkl'   # permite reanudar si la ejecución se interrumpe
    hecho = -1
    if os.path.exists(ck):
        filas, preds, hparams, perms, hecho = pickle.load(open(ck, 'rb'))
        print(f'[{etiqueta}] reanudando tras el pliegue {hecho+1}', flush=True)
    for k, (rep, tr, te) in enumerate(splits):
        if k <= hecho:
            continue
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
        gtr = None if groups is None else groups.iloc[tr]
        if groups is None:
            inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + k)
        else:
            inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42 + k)
        for nombre, cfg in modelos.items():
            pipe = Pipeline([('pre', preproc()), ('model', clone(cfg['model']))])
            fp = fit_params(cfg, ytr)
            if cfg['params']:
                gs = GridSearchCV(pipe, cfg['params'], cv=inner, scoring='average_precision', n_jobs=1)
                gs.fit(Xtr, ytr, groups=gtr, **fp)
                best, bp = gs.best_estimator_, gs.best_params_
            else:
                best, bp = pipe.fit(Xtr, ytr), {}
            p = best.predict_proba(Xte)[:, 1]
            oof = cross_val_predict(clone(best), Xtr, ytr, cv=inner, groups=gtr,
                                    method='predict_proba', params=fp)[:, 1]
            u = elegir_umbral(ytr, oof)
            yhat = (p >= u).astype(int)
            tn, fp_, fn, tp = confusion_matrix(yte, yhat, labels=[0, 1]).ravel()
            filas.append(dict(modelo=nombre, pliegue=k, repeticion=rep, umbral=u,
                              AP=average_precision_score(yte, p),
                              ROC_AUC=roc_auc_score(yte, p), Brier=brier_score_loss(yte, p),
                              TP=tp, FP=fp_, FN=fn, TN=tn))
            preds.append(pd.DataFrame(dict(modelo=nombre, repeticion=rep, pliegue=k, idx=te, y=yte.values,
                                           prob=p, umbral=u, pred=yhat)))
            hparams.append(dict(modelo=nombre, pliegue=k, **{h.replace('model__', ''): v for h, v in bp.items()}))
            if nombre in permutacion:
                pi = permutation_importance(best, Xte, yte, scoring='average_precision', n_repeats=5,
                                            random_state=k, n_jobs=1)
                for j, v in enumerate(VARS):
                    perms.append(dict(modelo=nombre, pliegue=k, variable=v, importancia=pi.importances_mean[j]))
        pickle.dump((filas, preds, hparams, perms, k), open(ck, 'wb'))
        print(f'[{etiqueta}] pliegue {k+1}/{len(splits)}  {time.time()-t0:.0f}s', flush=True)
    return (pd.DataFrame(filas), pd.concat(preds, ignore_index=True),
            pd.DataFrame(hparams), pd.DataFrame(perms))

def splits_estratificados(y, reps):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=reps, random_state=42)
    return [(i // 5, tr, te) for i, (tr, te) in enumerate(cv.split(X_all, y))]

def splits_agrupados(y, groups, reps):
    out = []
    for r in range(reps):
        sg = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=r)
        out += [(r, tr, te) for tr, te in sg.split(X_all, y, groups)]
    return out

# %% [markdown]
# ## Resúmenes: métricas con IC, matriz de confusión fuera de pliegue, prueba de Nadeau y Bengio

# %%
def resumen_metricas(folds, preds, y):
    """AP/ROC-AUC/Brier por pliegue (media, DE, IC95 con varianza corregida) y métricas fuera de pliegue por repetición."""
    J = folds['pliegue'].nunique()
    n_te, n_tr = 0.25, 1.0   # proporción prueba/entrenamiento en 5 pliegues
    tcrit = stats.t.ppf(0.975, J - 1)
    filas = []
    for m, g in folds.groupby('modelo'):
        fila = dict(modelo=m)
        for met in ['AP', 'ROC_AUC', 'Brier']:
            v = g[met].values
            se = np.sqrt((1 / J + n_te / n_tr) * v.var(ddof=1))
            fila.update({f'{met}_media': v.mean(), f'{met}_DE': v.std(ddof=1),
                         f'{met}_IC95_inf': max(0.0, v.mean() - tcrit * se), f'{met}_IC95_sup': min(1.0, v.mean() + tcrit * se)})
        # métricas fuera de pliegue por repetición (cada ficha predicha una vez por repetición)
        pm = preds[preds['modelo'] == m]
        por_rep = []
        for r, gr in pm.groupby('repeticion'):
            yt, yp = gr['y'].values, gr['pred'].values
            tn, fp_, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
            por_rep.append(dict(TP=tp, FP=fp_, FN=fn, TN=tn,
                                Precision=precision_score(yt, yp, zero_division=0),
                                Sensibilidad=recall_score(yt, yp, zero_division=0),
                                F2=fbeta_score(yt, yp, beta=2, zero_division=0)))
        pr = pd.DataFrame(por_rep)
        for c in pr.columns:
            fila[f'OOF_{c}_media'] = pr[c].mean(); fila[f'OOF_{c}_DE'] = pr[c].std(ddof=1) if len(pr) > 1 else np.nan
        fila['umbral_mediano'] = g['umbral'].median()
        filas.append(fila)
    return pd.DataFrame(filas)

def nadeau_bengio(folds, a, b, met='AP'):
    J = folds['pliegue'].nunique()
    da = folds[folds.modelo == a].sort_values('pliegue')[met].values
    db = folds[folds.modelo == b].sort_values('pliegue')[met].values
    d = da - db
    se = np.sqrt((1 / J + 0.25) * d.var(ddof=1))
    t = d.mean() / se
    p = 2 * stats.t.sf(abs(t), J - 1)
    return dict(modelo_A=a, modelo_B=b, metrica=met, dif_media=d.mean(), t=t, gl=J - 1, p=p)

def cramer_v(a, b):
    ct = pd.crosstab(a, b)
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum(); k = min(ct.shape) - 1
    return ct, chi2, float(np.sqrt(chi2 / (n * k)))

def distribucion_hparams(hp):
    filas = []
    for m, g in hp.groupby('modelo'):
        for col in [c for c in g.columns if c not in ('modelo', 'pliegue')]:
            vc = g[col].dropna().value_counts()
            for val, n in vc.items():
                filas.append(dict(modelo=m, hiperparametro=col, valor=val, pliegues=int(n), porcentaje=100 * n / len(g)))
    return pd.DataFrame(filas)

# %% [markdown]
# ## Etapa principal

# %%
def etapa_main():
    y = df['ALTA_DESERCION']
    reps = 1 if QUICK else 10
    modelos = modelos_config(QUICK)
    splits = splits_estratificados(y, reps)
    folds, preds, hp, perms = correr_anidado(X_all, y, splits, modelos,
                                             permutacion=('Gradient_Boosting', 'Random_Forest'), etiqueta='main')
    folds.to_csv(f'{SALIDA}/metricas_por_pliegue.csv', index=False)
    preds.to_csv(f'{SALIDA}/predicciones_por_pliegue.csv', index=False)
    hp.to_csv(f'{SALIDA}/hiperparametros_por_pliegue.csv', index=False)
    distribucion_hparams(hp).to_csv(f'{SALIDA}/hiperparametros_distribucion.csv', index=False)
    perms.to_csv(f'{SALIDA}/permutacion_por_pliegue.csv', index=False)
    pr = perms.groupby(['modelo', 'variable'])['importancia'].agg(['mean', 'std']).reset_index()
    pr.to_csv(f'{SALIDA}/importancia_permutacion.csv', index=False)
    res = resumen_metricas(folds, preds, y); res.to_csv(f'{SALIDA}/resumen_modelos.csv', index=False)
    pares = [('Gradient_Boosting', 'Random_Forest'), ('Gradient_Boosting', 'Regresion_Logistica'),
             ('Random_Forest', 'Regresion_Logistica'), ('Gradient_Boosting', 'Arbol_Decision'),
             ('Gradient_Boosting', 'Dummy'), ('Random_Forest', 'Dummy')]
    pd.DataFrame([nadeau_bengio(folds, a_, b_) for a_, b_ in pares]).to_csv(f'{SALIDA}/nadeau_bengio.csv', index=False)
    # Cramér
    ct, chi2, v = cramer_v(df['NIVEL_FORMACION'], df['MODALIDAD_FORMACION'])
    ct.to_csv(f'{SALIDA}/tabla_nivel_modalidad.csv'); json.dump(dict(chi2=chi2, V_Cramer=v), open(f'{SALIDA}/cramer_v.json', 'w'))
    # Modelo final (todos los datos), PDP y exportación para Power BI
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    finales = {}
    for nombre in ['Gradient_Boosting', 'Random_Forest']:
        cfg = modelos[nombre]
        pipe = Pipeline([('pre', preproc()), ('model', clone(cfg['model']))])
        fp = fit_params(cfg, y)
        gs = GridSearchCV(pipe, cfg['params'], cv=inner, scoring='average_precision', n_jobs=1).fit(X_all, y, **fp)
        oof = cross_val_predict(clone(gs.best_estimator_), X_all, y, cv=inner, method='predict_proba', params=fp)[:, 1]
        finales[nombre] = dict(best_params={k.replace('model__', ''): v for k, v in gs.best_params_.items()},
                               cv_AP=gs.best_score_, umbral_F2_interno=elegir_umbral(y, oof))
        imp = gs.best_estimator_.named_steps['model'].feature_importances_
        nombres = gs.best_estimator_.named_steps['pre'].get_feature_names_out()
        pd.DataFrame(dict(variable=nombres, importancia_impureza=imp)).sort_values('importancia_impureza', ascending=False)\
            .to_csv(f'{SALIDA}/importancia_impureza_{nombre}.csv', index=False)
        try:
            import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
            from sklearn.inspection import PartialDependenceDisplay
            fig, ax = plt.subplots(2, 3, figsize=(14, 7))
            PartialDependenceDisplay.from_estimator(gs.best_estimator_, X_all, VARS, categorical_features=CAT,
                                                    response_method='predict_proba', ax=ax.ravel()[:5])
            ax.ravel()[5].axis('off'); fig.suptitle(f'Dependencia parcial - {nombre}'); fig.tight_layout()
            fig.savefig(f'{SALIDA}/pdp_{nombre}.png', dpi=200); plt.close(fig)
        except Exception as e:
            print('PDP no generado:', repr(e))
    json.dump(finales, open(f'{SALIDA}/modelo_final.json', 'w'), indent=2, default=str)
    # Exportación Power BI: probabilidad media fuera de pliegue y clase por voto entre repeticiones
    out = df.copy()
    for nombre in ['Gradient_Boosting', 'Random_Forest', 'Regresion_Logistica']:
        pm = preds[preds.modelo == nombre]
        g = pm.groupby('idx').agg(p=('prob', 'mean'), voto=('pred', 'mean'))
        out[f'PROB_OOF_{nombre}'] = g['p'].reindex(range(len(df))).values
        out[f'CLASE_OOF_{nombre}'] = (g['voto'].reindex(range(len(df))).values >= 0.5).astype(int)
    out.to_csv(f'{SALIDA}/resultados_para_powerbi.csv', index=False)
    print(res.round(3).T.to_string())

# %% [markdown]
# ## Robustez: validación agrupada por programa (StratifiedGroupKFold)

# %%
def etapa_grupos():
    y = df['ALTA_DESERCION']; g = df['CODIGO_PROGRAMA']
    reps = 1 if QUICK else 4
    modelos = modelos_config(QUICK)
    folds, preds, hp, _ = correr_anidado(X_all, y, splits_agrupados(y, g, reps), modelos, groups=g, etiqueta='grupos')
    folds.to_csv(f'{SALIDA}/grupos_metricas_por_pliegue.csv', index=False)
    resumen_metricas(folds, preds, y).to_csv(f'{SALIDA}/grupos_resumen_modelos.csv', index=False)
    pd.DataFrame([nadeau_bengio(folds, 'Gradient_Boosting', 'Random_Forest')]).to_csv(f'{SALIDA}/grupos_nadeau_bengio.csv', index=False)
    print('programas distintos:', g.nunique())

# %% [markdown]
# ## Sensibilidad: (a) fichas observadas completas, (b) umbral de 12 %

# %%
def etapa_sens():
    modelos = {k: v for k, v in modelos_config(QUICK).items()
               if k in ('Dummy', 'Regresion_Logistica', 'Random_Forest', 'Gradient_Boosting')}
    # (b) umbral 12 %
    y12 = df['ALTA_DESERCION_SENS']
    print('Críticas con umbral 12 %:', int(y12.sum()), 'de', len(y12))
    folds, preds, _, _ = correr_anidado(X_all, y12, splits_estratificados(y12, 1 if QUICK else 3), modelos, etiqueta='sens12')
    resumen_metricas(folds, preds, y12).to_csv(f'{SALIDA}/sens_umbral12_resumen.csv', index=False)
    # (a) fichas completas: descriptivo para distintos cortes de terminación
    filas = []
    for corte in ['2024-02-28', '2024-02-29', '2024-03-14']:
        d = df[df['FECHA_TERMINACION_FICHA'] <= pd.Timestamp(corte)]
        for mod, gg in d.groupby('MODALIDAD_FORMACION'):
            filas.append(dict(terminadas_hasta=corte, modalidad=mod, fichas=len(gg), criticas=int(gg['ALTA_DESERCION'].sum())))
    pd.DataFrame(filas).to_csv(f'{SALIDA}/sens_completas_descriptivo.csv', index=False)
    print(pd.DataFrame(filas).to_string())

# %%
if __name__ == '__main__':
    {'main': etapa_main, 'grupos': etapa_grupos, 'sens': etapa_sens}[ETAPA]()