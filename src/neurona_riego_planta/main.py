import numpy as np

X  = np.array([
    [80, 18], [70, 22], [65, 28], [55, 25], [50, 32],
    [40, 30], [35, 25], [30, 32], [20, 35], [10, 38]
], dtype=float)
 
y = np.array([
    [0], [0], [0], [0], [0],
    [1], [1], [1], [1], [1]
], dtype=float)


escala = np.array([100,50])

X_normalizado = X / escala

print("\nDatos normallizados: ")
print(X_normalizado)


def sigmoide (z):

    return 1/ (1 + np.exp(-z))

def entrenar(tasa_aprendizaje, epocas, semilla=7, intervalo=None):

    rng = np.random.default_rng(semilla)
    pesos = rng.normal(size=(2,1))
    sesgo = 0.0
    historial = []

    for epoca in range(epocas):
        
        z = X_normalizado @ pesos + sesgo
        probabilidades = sigmoide(z)
 
        
        error = probabilidades - y
        gradiente_z = 2 * error * probabilidades * (1 - probabilidades) / len(X)
 
        
        gradiente_pesos = X_normalizado.T @ gradiente_z
        gradiente_sesgo = np.sum(gradiente_z)
 
        
        pesos -= tasa_aprendizaje * gradiente_pesos
        sesgo -= tasa_aprendizaje * gradiente_sesgo
 
        perdida = np.mean(error ** 2)
        historial.append(perdida)
 
        if intervalo and epoca % intervalo == 0:
            print(f"Época {epoca:6} | error cuadrático medio: {perdida:.6f}")
 
    return pesos, sesgo, np.array(historial)

def predecir_probabilidad(entradas, pesos, sesgo):
    """Normaliza con la misma variable `escala` y devuelve la probabilidad."""
    entradas = np.atleast_2d(np.asarray(entradas, dtype=float))
    return sigmoide((entradas / escala) @ pesos + sesgo)
 
 
def a_binario(probabilidad, umbral=0.5):
    return (probabilidad >= umbral).astype(int)
 
 
def error_final(pesos, sesgo):
    """MSE calculado con los parámetros ya actualizados."""
    return float(np.mean((predecir_probabilidad(X, pesos, sesgo) - y) ** 2))
 
 
# ---------------------------------------------------------------------------
#  Configuración inicial (prueba base)
# ---------------------------------------------------------------------------

tasa_aprendizaje = 0.5
epocas = 10000
 
print("\n=== ENTRENAMIENTO BASE (tasa 0.5, 10000 épocas) ===")
pesos, sesgo, historial = entrenar(tasa_aprendizaje, epocas, intervalo=2000)
 
print(f"\nPeso de la humedad     : {pesos[0, 0]:.4f}")
print(f"Peso de la temperatura : {pesos[1, 0]:.4f}")
print(f"Sesgo                  : {sesgo:.4f}")
print(f"Error final (MSE)      : {error_final(pesos, sesgo):.6f}")
 
print("\nProbabilidad calculada para cada caso de entrenamiento:")
probabilidades = predecir_probabilidad(X, pesos, sesgo)
respuestas = a_binario(probabilidades)
for i, (entrada, p, r, esperado) in enumerate(
    zip(X, probabilidades.ravel(), respuestas.ravel(), y.ravel()), start=1
):
    print(f"Caso {i:2}: humedad {entrada[0]:3.0f} %, temp {entrada[1]:2.0f} °C "
          f"-> prob {p:.4f} -> respuesta {r} (esperado {int(esperado)})")
 
print("\nSigno de los pesos:")
print("- Peso de humedad", "NEGATIVO: más humedad -> menor probabilidad de regar."
      if pesos[0, 0] < 0 else "POSITIVO: más humedad -> mayor probabilidad de regar.")
print("- Peso de temperatura", "POSITIVO: más temperatura -> mayor probabilidad de regar."
      if pesos[1, 0] > 0 else "NEGATIVO: más temperatura -> menor probabilidad de regar.")
 
# ---------------------------------------------------------------------------
#  Pruebas con condiciones nuevas
# ---------------------------------------------------------------------------

nuevos = np.array([
    [75, 30], [45, 34], [25, 22], [50, 25], [30, 40]
], dtype=float)
 
print("\n=== PRUEBAS CON CONDICIONES NUEVAS (modelo base) ===")
print(f"{'Humedad':>8} {'Temp':>6} {'Probabilidad':>13} {'Decisión':>9}")
prob_nuevos = predecir_probabilidad(nuevos, pesos, sesgo).ravel()
for (h, t), p in zip(nuevos, prob_nuevos):
    decision = int(p >= 0.5)
    texto = "1 (regar)" if decision else "0 (no regar)"
    print(f"{h:7.0f}% {t:5.0f}° {p:13.4f}   {texto}")
 
# ---------------------------------------------------------------------------
#  Experimentos con los parámetros (se cambia un solo parámetro por prueba)
# ---------------------------------------------------------------------------
experimentos = [
    ("Prueba base",          10000, 0.5),
    ("Pocas épocas",           100, 0.5),
    ("Cantidad intermedia",   1000, 0.5),
    ("Más épocas",           20000, 0.5),
    ("Tasa pequeña",         10000, 0.01),
    ("Tasa moderada",        10000, 0.1),
    ("Tasa alta",            10000, 1.0),
    ("Tasa muy alta",        10000, 2.0),
]
 
 
META_ERROR = 0.05  # error considerado "suficientemente bajo" para comparar velocidad
 
 
def epocas_hasta_meta(historial):
    """Primera época en la que el error baja de META_ERROR (None si nunca)."""
    indices = np.where(historial < META_ERROR)[0]
    return int(indices[0]) if len(indices) else None
 
 
def clasificar(historial):
    """Describe el aprendizaje a partir de la curva de error."""
    
    if np.any(np.diff(historial) > 1e-9):
        return "inestable"
    meta = epocas_hasta_meta(historial)
    if meta is None:
        return "insuficiente"
    return "rápido" if meta <= 2000 else "lento"
 
 
print("\n=== EXPERIMENTOS ===")
print(f"{'Prueba':<20} {'Épocas':>6} {'Tasa':>5} {'Error final':>12} "
      f"{'Aciertos':>9} {'P(45%,34°C)':>12} {'Ép.<0.05':>9}  Aprendizaje")
resultados = []
for nombre, ep, tasa in experimentos:
    w, b, hist = entrenar(tasa, ep)
    aciertos = int(np.sum(a_binario(predecir_probabilidad(X, w, b)) == y))
    p_45_34 = float(predecir_probabilidad([45, 34], w, b)[0, 0])
    estado = clasificar(hist)
    resultados.append((nombre, ep, tasa, error_final(w, b), aciertos, p_45_34, estado, w, b, hist))
    meta = epocas_hasta_meta(hist)
    print(f"{nombre:<20} {ep:6} {tasa:5} {error_final(w, b):12.6f} "
          f"{aciertos:7}/10 {p_45_34:12.4f} {str(meta) if meta is not None else '-':>9}  {estado}")
 
# ---------------------------------------------------------------------------
#  Prueba adicional con el umbral (SIN reentrenar: se usan los pesos base)
# ---------------------------------------------------------------------------

print("\n=== UMBRALES 0.4 / 0.5 / 0.6 (mismos pesos del modelo base) ===")
todos = np.vstack([X, nuevos])
etiquetas = [f"Entrenamiento {i}" for i in range(1, 11)] + \
            [f"Nuevo {i}" for i in range(1, 6)]
prob_todos = predecir_probabilidad(todos, pesos, sesgo).ravel()
print(f"{'Caso':<18} {'Hum':>4} {'Temp':>5} {'Prob':>8} {'u=0.4':>6} {'u=0.5':>6} {'u=0.6':>6}")
cambian = []
for etiqueta, (h, t), p in zip(etiquetas, todos, prob_todos):
    r4, r5, r6 = (int(p >= u) for u in (0.4, 0.5, 0.6))
    marca = "  <- cambia" if len({r4, r5, r6}) > 1 else ""
    if marca:
        cambian.append(etiqueta)
    print(f"{etiqueta:<18} {h:4.0f} {t:5.0f} {p:8.4f} {r4:6} {r5:6} {r6:6}{marca}")
print("Casos que cambian según el umbral:", cambian if cambian else "ninguno")
print(f"Pesos tras cambiar el umbral (iguales): {pesos.ravel()}, sesgo {sesgo:.4f}")
