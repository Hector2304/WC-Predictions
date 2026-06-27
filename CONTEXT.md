# Football Prediction Pipeline — Project Context

> Documento de referencia para nuevas sesiones. Cubre arquitectura, modelo activo (v5), estado actual, pendientes y cómo levantar todo.

---

## 1. Qué es este proyecto

Pipeline de predicción de resultados de fútbol internacional entrenado sobre el dataset `martj42/international_results` (~49 000 partidos desde 1872 hasta hoy). El objetivo es predecir los partidos del **Mundial 2026** (fase de grupos y eliminatoria).

Salida del modelo por partido: **P(H) / P(D) / P(A)** + marcador más probable.

---

## 2. Dataset

| Campo | Detalle |
|-------|---------|
| Archivo | `international_results-master/results.csv` |
| Fuente | github.com/martj42/international_results |
| Cobertura | 1872 – presente (se actualiza periódicamente) |
| Columnas clave | `date`, `home_team`, `away_team`, `home_score`, `away_score`, `tournament`, `neutral` |
| Nota Windows | La columna `neutral` llega como string `"TRUE"/"FALSE"` — `src/data/load.py` lo normaliza a bool |

**Nombres de equipos importantes:**
- `Czech Republic` (no `Czechia`)
- `Curaçao` (con cedilla — cuidado al agregar manualmente)
- `Ivory Coast` (no `Côte d'Ivoire`)

**Estado actual del dataset (26 junio 2026):**
- 64 partidos del Mundial 2026 con resultado (jornadas 1, 2 y 3 parcial — hasta 26 de junio)
- 8 fixtures sin resultado (2 del 26 de junio + 6 del 27 de junio — no jugados aún)
- Los partidos eliminatorios (R16 en adelante) no existen aún en el CSV

**Cómo agregar resultados nuevos:**
```
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
2026-06-29,Spain,Germany,2,1,FIFA World Cup,Miami,United States,TRUE
```
Actualizar directamente en `results.csv` y luego correr `py save_v5.py`.

---

## 3. Estructura del proyecto

```
Predictions/
│
├── international_results-master/
│   ├── results.csv                  # dataset fuente — editar con cuidado
│   └── mundial2026.csv              # resultados WC 2026 scrapeados manualmente
│
├── src/
│   ├── data/
│   │   └── load.py                  # carga y limpia results.csv
│   └── features/
│       ├── elo.py                   # sistema Elo con K-factor por torneo
│       ├── h2h.py                   # head-to-head con shrinkage bayesiano
│       ├── form.py                  # forma reciente (últimos N partidos)
│       ├── tournament.py            # is_world_cup + is_knockout (v5)
│       ├── wc_phases.py             # fechas de inicio de fase eliminatoria por edición WC
│       └── poisson.py               # clase PoissonDC (modelo central)
│
├── models/
│   ├── poisson_dc_v5.joblib         # modelo activo
│   └── v5_config.json               # config de v5 (thresholds, priors, hyperparams)
│
├── app.py                           # UI Streamlit
├── predict.py                       # CLI de predicciones
│
├── save_v5.py                       # entrena y guarda v5 (modelo activo)
├── calibrate_thresholds.py          # calibra theta_D y theta_D_knockout
│
├── compare_window.py                # comparación ventana temporal (full vs 1990+)
├── compare_v4_v5.py                 # comparación v4 vs v5
├── analyze_knockout_draws.py        # draw rates empíricos WC knockout por edición
├── diagnose_calibration.py          # diagnóstico calibración isotónica
│
└── requirements.txt
```

---

## 4. Arquitectura del pipeline

El pipeline es secuencial y **sin leakage**: cada paso usa solo información disponible antes del partido.

```
results.csv
    |
[load.py] --> df limpio
    |
[elo.py] --> df + elo_diff + final_ratings dict (ratings actuales)
    |
[h2h.py] --> df + h2h_home_goals_mu / h2h_away_goals_mu
    |
[form.py] --> df + form features (calculadas pero no usadas en v5)
    |
[tournament.py] --> df + is_world_cup + is_knockout
    |
[PoissonDC.fit(train)] --> modelo entrenado + rho
    |
[PoissonDC.predict_proba()] --> P(H), P(D), P(A) por partido
```

### Splits temporales

| Split | Rango | Uso |
|-------|-------|-----|
| Train | `< 2026-06-10` | Entrena PoissonDC (incluye 2023-2025) |
| Val | `2022-01-01 a 2023-01-01` | Calibra theta_D general |
| WC KO calib | WC knockout `< 2022-01-01` | Calibra theta_D_knockout (144 partidos, 1986-2018) |
| WC KO val | WC knockout 2022 | Valida theta_D_knockout (16 partidos) |

> El val set es un subconjunto del train. El modelo aprende sobre todos los datos < 2026-06-10; el val se usa solo para el sweep de threshold.

---

## 5. Sistema Elo (`src/features/elo.py`)

**Fórmula estándar** con K-factor variable por tipo de torneo:

| Torneo | K |
|--------|---|
| World Cup | 60 |
| Continental (Euros, Copa Am, etc.) | 50 |
| WC Qualifiers | 40 |
| Nations League | 45 |
| Friendly | 20 |
| Default | 30 |

- **HOME_ADVANTAGE:** 20 Elo points (el Poisson ya captura el efecto de cancha via `neutral`)
- **Rating inicial:** 1500 para equipos nuevos
- **Se calcula sobre el historial completo** (1872-hoy) para tener ratings precisos — aunque el Poisson solo entrena desde su train_cutoff

---

## 6. Features del modelo v5

### `elo_diff_scaled`
- `(elo_home - elo_away) / 400`
- División por 400 para escalar coeficientes a rango legible

### `neutral`
- `0` = partido en casa / `1` = cancha neutral
- Todos los partidos de WC son `neutral=1` (excepto locales del torneo: USA, Canadá, México)

### `h2h_home_goals_mu` / `h2h_away_goals_mu`
- Media con shrinkage bayesiano: `(sum_goals + k * global_avg) / (n + k)`
- `k = 5.0`
- Dirección: solo cuenta cuando ese par jugó en esa misma dirección (home vs away)
- Sin historial → prior global

### `is_world_cup`
- `1` si el partido es FIFA World Cup (excluye clasificatorias y amistosos)
- Coeficiente negativo → WC tiene menos goles que el promedio general

### `is_knockout`
- `1` si el partido es fase eliminatoria de WC
- Derivado de `wc_phases.py` — tabla de fechas de inicio de eliminatoria por edición (1986-2022)
- Coeficiente negativo adicional → eliminatoria tiene aún menos goles que fase de grupos WC

### Dixon-Coles `rho`
- Ajusta la matriz de probabilidades para partidos de pocos goles (0-0, 1-0, 0-1, 1-1)
- `rho = -0.0496` en v5

---

## 7. Modelo activo: v5

### Configuración (`models/v5_config.json`)

```json
{
  "version": "v5",
  "train_cutoff": "2026-06-10",
  "home_advantage": 20.0,
  "features": {
    "home": ["elo_diff_scaled", "neutral", "h2h_home_goals_mu", "is_world_cup", "is_knockout"],
    "away": ["elo_diff_scaled", "neutral", "h2h_away_goals_mu", "is_world_cup", "is_knockout"]
  },
  "priors": {
    "global_home_avg": 1.7248,
    "global_away_avg": 1.1604,
    "global_avg": 1.4426
  },
  "hyperparams": { "h2h_k": 5.0, "form_window": 5, "form_k": 3.0 },
  "theta_D": 0.26,
  "theta_D_knockout": 0.28,
  "rho": -0.049628
}
```

### Coeficientes v5

| Modelo | intercept | elo_diff_scaled | neutral | h2h_goals_mu | is_world_cup | is_knockout |
|--------|-----------|-----------------|---------|--------------|--------------|-------------|
| home | +0.1309 | +0.7237 | -0.0613 | +0.1939 | -0.0497 | -0.0870 |
| away | -0.2858 | -0.7811 | +0.2674 | +0.2522 | +0.0089 | -0.0967 |

### Sistema de dos umbrales

| Threshold | Valor | Cuándo se usa |
|-----------|-------|---------------|
| `theta_D` | 0.26 | Fase de grupos (y todos los partidos no-WC) |
| `theta_D_knockout` | 0.28 | Fase eliminatoria WC |

**Lógica de predicción:**
```
theta = theta_D_knockout si knockout=True, sino theta_D
si P(D) > theta  --> predice "D"
sino             --> predice argmax(P(H), P(A))
```

**Calibración:**
- `theta_D = 0.26` calibrado en val 2022 (954 partidos no-WC-KO), sweep F1-macro
- `theta_D_knockout = 0.28` calibrado en WC knockout 1986-2018 (144 partidos), validado en WC 2022 knockout (16 partidos): accuracy 68.8%, draw pred 25.0% vs real 31.2%

### Métricas v5 en WC 2026 grupo (48 partidos, test set)

| Métrica | Valor |
|---------|-------|
| Accuracy | 60.4% |
| F1-macro | 0.552 |
| Draw pred | 16.7% |
| Draw real | 29.2% |

---

## 8. `tournament.py` — corrección de falsos positivos

Versiones anteriores usaban substrings cortos (`"world cup"`, `"euro"`) que generaban falsos positivos:
- "Viva World Cup" (CONIFA) → taggeado como WC
- "Central European International Cup" → taggeado como Euros
- "West African Cup" → taggeado como AFCON

**Solución en v5:** frases exactas + exclusión de clasificatorias y amistosos:

```python
_RULES = [
    ("is_world_cup", ["fifa world cup"]),
    ("is_euros",     ["uefa euro"]),
    ...
]
_EXCLUDE = ["qual", "friendly"]

def _match(tournament, phrases):
    t = tournament.lower()
    if any(ex in t for ex in _EXCLUDE):
        return 0
    return int(any(ph in t for ph in phrases))
```

---

## 9. `wc_phases.py` — fechas de inicio eliminatoria

Tabla de fechas derivadas empíricamente del dataset (conteo de partidos de grupo por edición):

| Edición | Inicio eliminatoria |
|---------|---------------------|
| 1986 | 1986-06-15 |
| 1990 | 1990-06-23 |
| 1994 | 1994-07-02 |
| 1998 | 1998-06-27 |
| 2002 | 2002-06-15 |
| 2006 | 2006-06-24 |
| 2010 | 2010-06-26 |
| 2014 | 2014-06-28 |
| 2018 | 2018-06-30 |
| 2022 | 2022-12-03 |

WC 2026 no tiene fecha aún (eliminatoria por definirse cuando clasifiquen los 32 equipos).

---

## 10. Cómo levantar todo

### UI Streamlit

```bash
py -m streamlit run app.py
# Abre http://localhost:8501
```

Controles:
- Selectbox home/away team
- Checkbox "Neutral venue" (default True)
- Checkbox "Knockout phase" (default False) — activa theta_D_knockout=0.28

### CLI de predicciones

```bash
# Fase de grupos (default)
py predict.py "Argentina" "France"

# Fase eliminatoria
py predict.py "Argentina" "France" --knockout

# Múltiples partidos
py predict.py "Spain" "Germany" "Brazil" "England" --knockout

# Desde archivo (una línea = "TeamA,TeamB")
py predict.py --file r16_fixtures.txt --knockout

# Lista todos los equipos con Elo actual
py predict.py --teams

# Partido no neutral (local real)
py predict.py "Brazil" "Argentina" --not-neutral
```

### Volver a entrenar (cuando se actualizan datos)

```bash
# Re-entrenar v5 (actualiza Elo y coeficientes)
py save_v5.py

# Recalibrar thresholds theta_D y theta_D_knockout
py calibrate_thresholds.py
```

### Scripts de análisis

```bash
py analyze_knockout_draws.py   # draw rates empiricos WC knockout por edicion y ronda
py diagnose_calibration.py     # diagnostico calibracion isotonica
py compare_v4_v5.py            # comparacion v4 (dummies torneo) vs v5 (is_wc + is_ko)
py compare_window.py           # comparacion ventana temporal full vs 1990+
```

---

## 11. Ratings Elo actuales (top 25, post-jornada 2 WC 2026)

| # | Equipo | Elo |
|---|--------|-----|
| 1 | Spain | 2082 |
| 2 | Argentina | 2076 |
| 3 | France | 2043 |
| 4 | England | 1993 |
| 5 | Germany | 1972 |
| 6 | Colombia | 1968 |
| 7 | Morocco | 1961 |
| 8 | Mexico | 1954 |
| 9 | Portugal | 1953 |
| 10 | Brazil | 1952 |
| 11 | Netherlands | 1944 |
| 12 | Norway | 1942 |
| 13 | Japan | 1940 |
| 14 | Croatia | 1901 |
| 15 | United States | 1891 |
| 16 | Ecuador | 1888 |
| 17 | Paraguay | 1878 |
| 18 | Italy | 1873 |
| 19 | Switzerland | 1867 |
| 20 | Australia | 1861 |
| 21 | Turkey | 1861 |
| 22 | Belgium | 1852 |
| 23 | Uruguay | 1850 |
| 24 | South Korea | 1846 |
| 25 | Denmark | 1845 |

> Para ver la lista completa actualizada: `py predict.py --teams`

---

## 12. Plan fase KO — WC 2026

### Paso 0 — Versionar modelo experimental (ANTES de cualquier cambio)
Guardar el estado actual (v5 + form + FIFA stats experimental) como versión estable antes de tocar nada. Si los siguientes cambios empeoran el modelo, se puede volver aquí.

### Paso 1 — Fix técnico OBLIGADO: `wc_phases.py`
**Dato necesario:** fecha del primer partido del R32.

`WC_KNOCKOUT_START` no tiene 2026 → `is_wc_knockout()` retorna 0 para todos los partidos 2026. Al reentrenar con datos KO, esos partidos entrarán como `is_knockout=0`. Hay que agregar:
```python
2026: pd.Timestamp("2026-07-XX"),  # fecha real del primer R32
```
También actualizar `TRAIN_CUTOFF` en `save_v5.py` después de cada ronda.

### Paso 2 — Análisis R32: ¿threshold propio?
**Datos necesarios:** fixture R32 (16 emparejamientos) + criterio de clasificación de los 8 mejores terceros.

El modelo tiene `theta_D_knockout=0.28` calibrado sobre R16+QF+SF+Final (1986-2022).
Draw rates históricos por ronda: R16=19.6%, QF=32.1%, SF=14.3%. R32 probablemente < R16.
Con el fixture real, calcular Elo diffs para ver si los matchups son tan disparejos como se espera.
Decisión pendiente: usar el mismo `theta_D_knockout=0.28`, o un `theta_D_R32` más bajo (¿0.26?).

### Paso 3 — Stats FIFA actualizadas
**Datos necesarios:** archivos FIFA actualizados al cierre de fase de grupos (solo 32 clasificados).

Actualmente 9 indicadores. Con 3 partidos jugados los números son más representativos.
Ampliar columnas al momento de recibir los archivos — decidir qué agregar en ese momento.
Reemplazar los archivos en `DATOS FIFA/` y reajustar `src/features/fifa_stats.py`.

### Paso 4 — Stats de jugadores
El más complejo. Criterios acordados:
- Solo top ~100 jugadores más influyentes del torneo (no todos)
- Columnas a definir cuando lleguen los datos (goles, asistencias, minutos, shots en torneo, etc.)
- Usar como señal de ajuste suave encima del modelo base, no como feature del Poisson
- Riesgo a tener en cuenta: muestra pequeña (3 partidos) + lesiones/suspensiones en KO

### Orden de implementación
1. Versionar modelo actual
2. Recibir fixture R32 → fix `wc_phases.py` + análisis threshold
3. Recibir stats FIFA actualizadas → ampliar indicadores
4. Recibir datos de jugadores → definir columnas → integrar como ajuste

---

## 13. Pendientes y próximos pasos (operación inmediata)

### Datos por agregar (operación inmediata)

| Qué | Cuándo | Cómo |
|-----|--------|------|
| Resultados jornada 3 grupo (24-27 jun) | Cuando se jueguen | Agregar a results.csv + `py save_v5.py` |
| Fixture R16 (16 partidos) | Cuando se defina | `py predict.py --file r16.txt --knockout` |
| Resultados R16 | Tras jugarse | Agregar a results.csv + `py save_v5.py` |
| Fixture y resultados QF, SF, Final | Progresivamente | Mismo proceso |

### Retrain post-Mundial 2026

Una vez que termine el torneo:
1. Agregar todos los resultados de eliminatoria 2026 a `results.csv`
2. Incluir WC 2022 knockout en el set de calibración de `theta_D_knockout` (actualmente se excluye para usarlo como validación — cuando tengamos WC 2026 como nueva validación, ya no se necesita)
3. Correr `py save_v5.py` y luego `py calibrate_thresholds.py`
4. `theta_D_knockout` se recalibrará sobre 1986-2022 (160 partidos) y se validará en WC 2026 knockout

### Mejoras técnicas descartadas o diferidas

| Mejora | Estado | Razón |
|--------|--------|-------|
| Ventana temporal 1990+ | Descartada | Accuracy WC 2026 baja 60.4%→56.2%; coef is_knockout se invierte |
| Dixon-Coles completo (ratings ataque/defensa por equipo) | Diferida | Evaluar post-WC 2026 según resultado del modelo actual |
| Round-specific theta_D (R16 vs QF) | Diferida | Requiere columna de ronda en dataset (no existe) |
| Isotonic calibration | Descartada como mejora | Diagnóstico confirmó problema estructural, no de escala |
| Feature de paridad (ej. `1/(1+|elo_diff|/200)`) | **Descartada con evidencia** | Ver análisis draw gap abajo |
| Bajar theta_D a 0.24 | **Descartado con datos** | 2 TP ganados a costa de 12 FP; ratio 6:1; ver análisis draw gap |
| Bajar theta_D_knockout a 0.26 | **Descartado con datos** | 1 TP ganado (Argentina-Francia) a costa de 7 FP; ratio 7:1; ver análisis threshold knockout |

### Análisis del draw gap — WC 2026 fase de grupos (junio 2026)

**Hallazgo principal:** el gap empates (modelo predice 16.7%, real 29.2%) es varianza de upsets, no un defecto estructural del modelo ni una feature faltante.

**Metodología:** se corrió `analyze_draw_gap.py` con tres sweeps de threshold (f1_macro, f1_draw, draw_recall) sobre val_general 2022 y se evaluaron en los 48 partidos del grupo WC 2026. Se segmentaron los falsos negativos de empate por |elo_diff| y profundidad H2H.

**Resultados clave:**

| theta | acc | draw_pred | TP | FP | FN |
|-------|-----|-----------|----|----|-----|
| 0.26 (actual) | 60.4% | 16.7% | 3 | 5 | 11 |
| 0.24 (experimento) | 41.7% | 45.8% | 5 | 17 | 9 |
| 0.21 (f1_draw) | 35.4% | 68.8% | 8 | 25 | 6 |

La curva FP/TP en la banda [0.24, 0.26] es 6:1 — la banda tiene masa de P(D) donde el modelo correctamente asigna baja probabilidad de empate a matchups diferenciados.

**Segmentación de falsos negativos:**
- Empates con |elo_diff| > 185 (avg=288): **100% miss rate** — Spain-Cape Verde (+465), England-Ghana (+369), Ecuador-Curaçao (+324), Portugal-DR Congo (+218), etc. Upsets genuinos e impredecibles.
- Empates con |elo_diff| <= 185 (avg=78): **57% miss rate** — dos de ellos (Belgium-Egypt, Czech Republic-South Africa) tienen P(D)=0.255 y 0.250, justo bajo el umbral.

**P(D) de TP vs FN:**
- TP correctamente predichos: P(D) mean=0.266, rango [0.264, 0.267] — todos en matchups con |elo_diff| <= 24.
- FN fallados: P(D) mean=0.195, rango [0.085, 0.255] — el modelo les asigna correctamente probabilidades más bajas.

**Conclusión:** el modelo tiene ordenamiento correcto (sabe que Netherlands-Japan es más probable empate que Spain-Cape Verde). Los 11 FN se dividen en: 7 upsets de alto Elo irrecuperables por cualquier modelo, y 4 near-threshold donde el modelo tiene P(D) plausible pero ligeramente baja. Bajar theta no arregla los upsets y destruye precision en el resto.

**theta_D = 0.26 confirmado. Feature de paridad descartada con evidencia.**

### Análisis threshold knockout — WC 2022 knockout (16 partidos, junio 2026)

**Motivación:** `theta_D_knockout=0.28` no había pasado el mismo escrutinio empírico que `theta_D=0.26`. Se corrió `analyze_knockout_threshold.py` para aplicar la misma metodología.

**Hallazgo crítico — banda crítica:** P(D) de los 16 partidos de validación cae en [0.202, 0.290] con media 0.264 y mediana 0.274. El **81% (13 de 16) cae en la banda [0.25, 0.31]** — theta=0.28 es decisivo para casi todo el bracket, no marginal.

**La calibración aguanta:** sweep f1_macro sobre los 144 partidos de calibración (1986-2018) produce exactamente theta=0.28. El número no es arbitrario.

**Curva FP/TP:** f1_draw (theta=0.26) gana 1 TP (Argentina-Francia, P(D)=0.279) a costa de 7 FP. Ratio 7:1 — mismo patrón catastrófico que en grupos.

**Near-threshold miss:** Argentina vs France, P(D)=0.279, actual=D, pred=H. Solo 0.001 por debajo del umbral. Con n=16 es ruido, no señal.

**Limitación declarada:** n=16 → cada partido vale 6.25pp de accuracy. El 68.8% de accuracy podría ser 62.5% o 75% con otra realización. La calibración es internamente consistente pero estadísticamente débil. Se trata como prior confirmado, no como calibración robusta.

**theta_D_knockout = 0.28 confirmado con los mismos criterios que theta_D = 0.26.**

---

## 13. Decisiones de diseño relevantes

| Decisión | Alternativa descartada | Razón |
|----------|----------------------|-------|
| Solo `is_world_cup` + `is_knockout` (no dummies por Euros/AFCON/etc.) | 6 dummies de torneo (v4) | Producto es exclusivamente WC — granularidad adicional no aporta |
| `theta_D_knockout = 0.28` separado de `theta_D = 0.26` | Un solo threshold | Draw rate en eliminatoria WC (21-31%) es estructuralmente distinto al general. Sweep f1_macro sobre 144 partidos produce exactamente 0.28 — internamente consistente. Validado en 16 partidos (n pequeño, prior débil). Ver sección "Análisis threshold knockout" |
| Normalización por Elo en knockout neutral | Orden del archivo de input | `predict.py` y `app.py` reordenan automáticamente al equipo de mayor Elo como "home" para hacer el H2H lookup determinístico. La nota "(ko) = reordered" aparece en output cuando hay swap |
| Train cutoff `2026-06-10` (incluye 2023-2025) | `2023-01-01` (v3 original) | 3,598 partidos de fútbol moderno ganados; Euro 2024, Copa Am 2024, clasificatorias WC 2026 |
| Elo calculado sobre historial completo (1872-hoy) | Solo desde 1990 | Ratings más precisos para todos los equipos |
| Poisson entrenado sobre historial completo (no ventana 1990+) | Ventana 1990+ | Experimento mostró accuracy WC peor (56% vs 60%) y coef is_knockout se invierte |
| `HOME_ADVANTAGE = 20` | 100 (estándar Elo) | El feature `neutral` del Poisson ya captura el efecto de cancha |
| Calibración theta_D en val 2022 | Val más reciente | 2022 es el último año con suficientes partidos de todos los torneos antes del WC 2026 |
| Shrinkage bayesiana en H2H (`k=5.0`) | Sin shrinkage | 18% de test set tiene 0 historial H2H directo |
| Exclusión de `"qual"` y `"friendly"` en tournament matching | Solo frases exactas | "FIFA World Cup qualification" matcheaba "fifa world cup" sin esta exclusión |
