# Guía de Setup — Predicción de Resultados de Partidos Internacionales

Dataset base: `martj42/international_results` (GitHub)
Repo: https://github.com/martj42/international_results

Esta guía es de **setup y planeación**, sin código. El objetivo es que tengas un camino claro antes de empezar a programar.

---

## 1. Qué contiene el dataset

El repo tiene principalmente estos archivos:

- **`results.csv`** — ~49,000 partidos internacionales masculinos desde 1872 hasta la actualidad. Columnas:
  - `date` — fecha del partido
  - `home_team` / `away_team` — equipo local y visitante (usa el nombre **actual** del equipo, ej. partidos históricos de "Ireland" aparecen como "Northern Ireland")
  - `home_score` / `away_score` — goles de cada equipo
  - `tournament` — nombre del torneo (Mundial, Copa América, amistoso, etc.)
  - `city` / `country` — sede del partido
  - `neutral` — booleano: si el partido se jugó en cancha neutral (importante para el factor "local")

- **`shootouts.csv`** — registros de partidos que se decidieron por penales (con el ganador de la tanda).

- Es posible que el repo tenga también un archivo de goleadores (`goalscorers.csv`) — vale la pena revisar el listado actual del repo porque el dataset se sigue actualizando vía pull requests.

**Limitaciones a tener en cuenta:**
- No incluye Juegos Olímpicos en las reglas de inclusión recientes, aunque hay partidos históricos de Olimpiadas en el CSV — revisa si quieres filtrarlos.
- No incluye partidos B, sub-23 ni de selecciones de clubes.
- Los nombres de equipos están "normalizados" al nombre actual, así que cuidado si cruzas esto con otra fuente de datos (ratings Elo, rankings FIFA) que use nomenclatura distinta o histórica.

---

## 2. Estructura de carpetas recomendada

```
prediccion-partidos/
├── data/
│   ├── raw/              # CSVs originales sin tocar (results.csv, shootouts.csv)
│   ├── interim/          # datos limpios pero sin features (ej. nombres normalizados)
│   └── processed/        # dataset final con features, listo para entrenar
├── notebooks/            # exploración (EDA), prototipos de features y modelos
├── src/
│   ├── data/              # scripts de carga y limpieza
│   ├── features/          # cálculo de Elo, forma reciente, etc.
│   ├── models/             # entrenamiento y evaluación
│   └── utils/
├── models/                # modelos entrenados serializados (.pkl, .joblib)
├── reports/                # gráficas, métricas, resultados de validación
├── environment.yml  o  requirements.txt
└── README.md
```

Esto te da separación clara entre datos crudos, features y modelos — importante para no mezclar pasos y evitar fugas de información (data leakage).

---

## 3. Entorno de trabajo

Dado tu setup (Windows, PhpStorm/WebStorm como IDEs habituales, pero esto es un proyecto de Python puro):

1. **Gestor de entorno**: usa un entorno virtual dedicado (venv o conda) solo para este proyecto — nunca mezclar con tu entorno global de Python.
2. **IDE**: puedes abrir la carpeta en PyCharm/WebStorm con el plugin de Python, o usar VS Code si prefieres notebooks Jupyter integrados.
3. **Librerías núcleo a instalar** (sin código, solo la lista para tu `requirements.txt`):
   - `pandas`, `numpy` — manejo de datos
   - `scikit-learn` — modelos clásicos y métricas
   - `matplotlib`, `seaborn` — visualización exploratoria
   - `xgboost` o `lightgbm` — gradient boosting
   - `jupyter` o `jupyterlab` — para exploración interactiva
   - Opcional más adelante: `pymc` (modelos bayesianos), `statsmodels` (Poisson/Dixon-Coles)
4. **Control de versiones**: clona el repo de datos como submódulo o simplemente descarga el CSV y versiona tu propio repo aparte — no necesitas el historial de git del dataset, solo los archivos.

---

## 4. Plan de trabajo por fases

### Fase 1 — Ingesta y limpieza
- Descargar `results.csv` (y `shootouts.csv` si lo vas a usar).
- Verificar tipos de datos (fechas como fecha, no string; scores como enteros).
- Decidir si filtras por rango de fechas (ej. solo desde 1990 o desde 2000, porque el fútbol de hace 100 años se juega muy distinto y puede meter ruido).
- Decidir si filtras por tipo de torneo (¿incluyes amistosos? suelen tener menos intensidad competitiva y pueden sesgar el modelo).

### Fase 2 — Análisis exploratorio (EDA)
- Distribución de goles por partido (¿se ajusta a Poisson?).
- Ventaja de jugar en casa: diferencia de % de victorias local vs visitante vs neutral.
- Evolución de fuerza de selecciones a través del tiempo (¿tiene sentido dar más peso a partidos recientes?).
- Frecuencia de partidos por equipo (algunos equipos juegan muchos amistosos, otros casi nada — esto afecta cuántos datos tienes por equipo).

### Fase 3 — Ingeniería de features
Aquí es donde decides qué variables vas a construir, por ejemplo:
- Rating Elo dinámico por equipo (se recalcula partido a partido, en orden cronológico).
- Forma reciente (últimos N partidos: % victorias, goles a favor/contra).
- Historial head-to-head entre el par de equipos.
- Ventaja de local / neutral.
- Diferencia de ranking o de Elo entre los dos equipos del partido a predecir.
- Importancia del torneo (Mundial ≠ amistoso) como variable categórica.

**Punto crítico**: cualquier feature debe calcularse usando solo información disponible **antes** de la fecha del partido que se va a predecir. Si calculas un promedio usando todo el dataset (incluyendo partidos futuros), hay fuga de información y el modelo va a parecer mejor de lo que realmente es.

### Fase 4 — Modelado
- Empieza con un modelo simple (regresión logística o Elo puro) como **baseline**.
- Compara contra Dixon-Coles o Poisson para predicción de marcador.
- Prueba gradient boosting (XGBoost/LightGBM) con las features de la fase 3.
- Decide qué vas a predecir exactamente: ¿resultado (gana/pierde/empata), marcador exacto, o solo goles esperados por equipo? Cada opción cambia la métrica de evaluación y el tipo de modelo más adecuado.

### Fase 5 — Validación
- **No uses split aleatorio** — usa split temporal (ej. entrena con partidos hasta 2018, valida con 2019-2022, prueba con 2023-2024). Esto simula cómo se usaría el modelo en la realidad.
- Métricas según el tipo de predicción:
  - Clasificación de resultado: accuracy, log-loss, F1 por clase (ojo: las clases "empate" suelen estar desbalanceadas).
  - Predicción de goles: MAE o RMSE sobre goles esperados.
- Compara siempre contra el baseline simple — si tu modelo complejo no le gana al Elo básico por un margen claro, no vale la complejidad extra.

### Fase 6 — Documentación y reproducibilidad
- README con cómo correr el pipeline de principio a fin.
- Fijar semillas aleatorias (`random_state`) en cualquier paso con componente aleatorio.
- Guardar versión exacta de las librerías usadas (`requirements.txt` con versiones fijas, no solo nombres).

---

## 5. Próximos pasos sugeridos

Una vez que tengas el entorno armado y el dataset descargado, el primer entregable concreto debería ser un notebook de EDA que responda:
1. ¿Cuántos partidos por torneo/década tienes disponibles?
2. ¿Qué tan fuerte es la ventaja de local en este dataset específico?
3. ¿Qué rango de fechas vas a usar para entrenar (todo el histórico vs. los últimos 20-30 años)?

Con esas respuestas ya puedes decidir con más certeza si te conviene Elo+ML, Poisson/Dixon-Coles, o un híbrido — como vimos antes en la conversación.

¿Quieres que profundice en el diseño específico del sistema de rating Elo, o prefieres que armemos primero el checklist del EDA?
