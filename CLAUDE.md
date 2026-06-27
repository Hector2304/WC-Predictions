# Instrucciones para Claude Code — Football Prediction Pipeline

Ver `CONTEXT.md` para arquitectura completa del proyecto.

---

## Protocolo: agregar resultado de partido

Cuando el usuario da un marcador nuevo, actualizar **los tres archivos en orden**:

### 1. `results.csv` (fuente del modelo — PRIORIDAD)

El archivo ya tiene filas placeholder para todos los partidos del WC 2026 con venues correctos pero sin scores. Solo hay que llenar los scores:

```
# Buscar la fila existente y rellenar home_score y away_score
# Formato: scores como entero (1, no 1.0)
2026-06-26,Uruguay,Spain,0,1,FIFA World Cup,Zapopan,Mexico,True
```

**Nunca inventar venues** — ya están en las filas placeholder de `results.csv`. Usar siempre esos.

### 2. `mundial2026.csv` (usado por app.py)

Si la fila no existe, agregarla copiando venue y neutral de `results.csv`:

```
2026-06-26,Uruguay,Spain,0,1,FIFA World Cup,Zapopan,Mexico,TRUE
```

Nota: `results.csv` usa `True/False`, `mundial2026.csv` usa `TRUE/FALSE`.

### 3. `predictions_j3.csv`

Llenar `actual_home`, `actual_away` y `correct` (1 si acertó la predicción, 0 si no):

```
# pred=Spain, resultado=0-1 (Spain gana) → correct=1
2026-06-26,Uruguay,Spain,...,Spain,0,1,1
```

---

## Protocolo: auditoría de datos

Correr cuando:
- El usuario lo pida explícitamente
- Antes de reentrenar el modelo (`save_v5.py`)
- Al inicio de sesión si hubo resultados recientes

Script de auditoría:
```
C:\Users\hecto\AppData\Local\Temp\claude\...\scratchpad\full_audit.py
```

O recrearlo desde cero — hace tres chequeos:
1. Scores de `mundial2026.csv` coinciden con `results.csv`
2. Partidos jugados en `results.csv` tienen scores (no vacíos)
3. `predictions_j3.csv` tiene `actual_home/away/correct` para todos los partidos ya jugados

### Falsos positivos conocidos del audit

- **Curaçao**: `results.csv` usa `CuraÃ§ao` (mojibake) o `Curaçao`; `mundial2026.csv` usa `Curacao`. Es el mismo equipo — `load.py` lo normaliza. No es error.
- **Partidos futuros**: `results.csv` tiene filas sin score para partidos no jugados. El `dropna` de `load.py` los ignora. No es error.

---

## Comportamiento correcto del checkbox "Neutral venue" en la UI

- **Activado (True)** — todos los partidos WC donde ninguno de los dos equipos es sede (la gran mayoría). Uruguay en Zapopan → activado aunque Uruguay aparezca como Team 1.
- **Desactivado (False)** — solo cuando el Team 1 realmente juega en su propio país sede. Ejemplos: México en Ciudad de México, EE.UU. en cualquier ciudad de EE.UU., Canadá en Toronto/Vancouver.

No hay doble conteo: el `elo_diff` que entra al Poisson es el diff crudo sin home adjustment. El único mecanismo que cambia la predicción al desactivar la casilla es el coeficiente `neutral` del Poisson. El +20 de Elo se usa solo para calibrar los ratings durante entrenamiento, no se suma al input de predicción.

---

## Nombres de equipos — convenciones

| En `results.csv` | En `mundial2026.csv` |
|---|---|
| `CuraÃ§ao` / `Curaçao` | `Curacao` |
| `Czech Republic` | `Czech Republic` |
| `Ivory Coast` | `Ivory Coast` |

---

## Archivos clave

| Archivo | Uso |
|---|---|
| `results.csv` | Fuente del modelo (load.py lo lee) |
| `mundial2026.csv` | Solo para app.py (display) |
| `predictions_j3.csv` | Tracking de precisión jornada 3 |
| `save_v5.py` | Reentrenar modelo |
