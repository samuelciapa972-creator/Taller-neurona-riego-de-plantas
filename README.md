Neurona artificial para el riego de plantas

Taller de IA Generativa, Universidad Santo Tomás (Tunja). La idea es construir una neurona artificial desde cero con NumPy que, mirando la humedad del suelo y la temperatura, decida si una planta hay que regarla o no. Los datos son didácticos, no sirven como recomendación agronómica real.

Qué hace

La neurona recibe dos entradas (humedad en % y temperatura en °C), las combina con dos pesos y un sesgo, pasa el resultado por una sigmoide y devuelve una probabilidad. Si la probabilidad es mayor o igual a 0.5, la salida es 1 (regar). Si no, es 0 (no regar).

Cómo ejecutarlo

Necesita uv y Python 3.12 o superior.

bash
git clone https://github.com/samuelciapa972-creator/Taller-neurona-riego-de-plantas
cd Taller-neurona-riego-de-plantas
cd src
cd neurona_riego_planta
uv sync
uv run main.py

Una sola ejecución imprime todo: el entrenamiento base, las cinco predicciones nuevas, los ocho experimentos y la prueba con umbrales. La semilla de los pesos iniciales es fija (default_rng(7)), así que a cualquiera le debe dar lo mismo que a mí.

Cómo está armado
Normalizo las entradas: X_normalizado = X / [100, 50].
Calculo z = X_normalizado @ pesos + sesgo.
Aplico la sigmoide para obtener la probabilidad.
Calculo el error cuadrático medio y sus gradientes.
Actualizo pesos y sesgo restando tasa_aprendizaje * gradiente.
Repito por cada época.

Los datos nuevos se dividen por la misma escala antes de entrar a la neurona, nunca se calcula una escala distinta.

Resultado del entrenamiento base

Con tasa de aprendizaje 0.5 y 10000 épocas:

Parámetro	Valor
Peso de la humedad	-19.5032
Peso de la temperatura	2.3101
Sesgo	7.4022
Error final (MSE)	0.019645
Caso	Humedad	Temp.	Esperado	Probabilidad	Respuesta
1	80 %	18 °C	0	0.0006	0
2	70 %	22 °C	0	0.0053	0
3	65 %	28 °C	0	0.0183	0
4	55 %	25 °C	0	0.1025	0
5	50 %	32 °C	0	0.2951	0
6	40 %	30 °C	1	0.7285	1
7	35 %	25 °C	1	0.8496	1
8	30 %	32 °C	1	0.9539	1
9	20 %	35 °C	1	0.9941	1
10	10 %	38 °C	1	0.9993	1

Qué significa el signo de cada peso. El de la humedad es negativo: mientras más húmedo está el suelo, menos probabilidad hay de regar. El de la temperatura es positivo: mientras más calor hace, más probabilidad hay de regar. Tiene sentido con lo que uno esperaría de una planta.

Predicciones con condiciones nuevas
Humedad	Temperatura	Probabilidad	Decisión
75 %	30 °C	0.0029	0 (no regar)
45 %	34 °C	0.5490	1 (regar)
25 %	22 °C	0.9719	1 (regar)
50 %	25 °C	0.2325	0 (no regar)
30 %	40 °C	0.9677	1 (regar)

Cuatro de las cinco son claras. La de 45 % y 34 °C no: da 0.549, casi en el límite, así que la neurona prácticamente "duda".

Experimentos

En cada prueba cambié solo un parámetro respecto a la base (10000 épocas y tasa 0.5). Para comparar qué tan rápido aprendía cada una, anoté en qué época el error baja de 0.05.

Prueba	Épocas	Tasa	Error final	Aciertos	P(45 %, 34 °C)	Época con error < 0.05	Aprendizaje
Base	10000	0.5	0.019645	10/10	0.5490	1977	Rápido
Pocas épocas	100	0.5	0.165623	10/10	0.5217	no llega	Insuficiente
Intermedia	1000	0.5	0.067481	10/10	0.5793	no llega	Insuficiente
Más épocas	20000	0.5	0.011705	10/10	0.5263	1977	Rápido
Tasa pequeña	10000	0.01	0.130048	10/10	0.5388	no llega	Insuficiente
Tasa moderada	10000	0.1	0.049728	10/10	0.5853	9884	Lento
Tasa alta	10000	1.0	0.011705	10/10	0.5263	988	Rápido
Tasa muy alta	10000	2.0	0.006566	10/10	0.5082	494	Rápido, sin inestabilidad

Algo que me llamó la atención: todas las pruebas clasifican bien los 10 casos, incluso la de 100 épocas. Lo que cambia es qué tan seguras son las probabilidades y cuánto error queda.

Prueba con umbrales

Sin volver a entrenar, usé los mismos pesos y probé los umbrales 0.4, 0.5 y 0.6.

Solo cambia un caso: el de 45 % y 34 °C (probabilidad 0.549). Con 0.4 y 0.5 da 1, y con 0.6 da 0. Los otros 14 casos no cambian porque sus probabilidades están lejos de la zona entre 0.4 y 0.6 (la más cercana es 0.2951 del caso 5 y 0.7285 del caso 6).

Los pesos no cambian porque el umbral se usa después de entrenar, solo para convertir la probabilidad en 0 o 1. No participa en el cálculo del error ni de los gradientes, así que no puede afectar lo que la neurona aprendió. Con los tres umbrales los pesos siguen siendo -19.50 y 2.31, y el sesgo 7.40.

Análisis

1. ¿Por qué normalizar? La humedad llega hasta 100 y la temperatura hasta unos 50. Si las dejo así, la humedad pesa más solo por tener números más grandes, z se dispara, la sigmoide se satura y los gradientes quedan desbalanceados. Dividiendo por 100 y 50 las dos variables quedan cerca de 0 a 1 y aportan de forma parecida.

2. ¿Dónde se usó X_normalizado y para qué se guardó X? X_normalizado se usa en la suma ponderada y en el gradiente de los pesos, y también para los datos nuevos. X se conserva con los valores originales para mostrar los resultados en % y °C, que es como se entienden.

3. ¿Qué pasó con 100 épocas? El error quedó en 0.1656 y los pesos quedaron chicos (-1.81 y 1.06). Las probabilidades están todas entre 0.29 y 0.69, o sea cerca de 0.5. Acierta los 10 casos, pero sin convicción: se quedó corta de entrenamiento.

4. ¿Más épocas siempre mejoran mucho? No. De 100 a 1000 épocas el error bajó bastante (0.166 a 0.067), pero de 10000 a 20000 solo pasó de 0.0196 a 0.0117. Y los aciertos eran 10/10 desde el principio, así que no mejoró la clasificación, solo la confianza. Se nota que cada vez rinde menos.

5. ¿Qué pasó con una tasa muy pequeña? Con 0.01 los pesos casi no se mueven. Después de 10000 épocas el error sigue en 0.130 y nunca baja de 0.05. Va en la dirección correcta, pero tan despacio que no alcanza.

6. ¿Qué pasó con una tasa alta o muy alta? Aprenden más rápido: con 1.0 baja de 0.05 en 988 épocas y con 2.0 en 494, contra 1977 con 0.5. En este problema no vi inestabilidad, el error nunca subió. Eso sí, con 2.0 los pesos se hacen muy grandes (-31.15 de humedad y 13.59 de sesgo) y las probabilidades quedan más extremas. También noté que tasa 1.0 con 10000 épocas da exactamente el mismo error que tasa 0.5 con 20000 (0.011705), lo que parece indicar que duplicar la tasa equivale, aquí, a duplicar las épocas.

7. ¿Qué representa el signo del peso de la humedad? Es negativo, entonces más humedad baja la probabilidad de regar y menos humedad la sube. Además es el peso más grande en valor absoluto, así que la humedad es la variable que más pesa en la decisión.

8. ¿Qué representa el signo del peso de la temperatura? Es positivo: más calor sube la probabilidad de regar. Su valor (2.31) es mucho menor que el de la humedad, por lo que influye menos. Hay que tener cuidado con esta lectura porque en los datos la temperatura sube justo cuando la humedad baja, y con solo 10 ejemplos la neurona no puede separar bien el efecto de cada una.

9. ¿Por qué convertir la probabilidad en 0 o 1 con un umbral? Porque la sigmoide da un valor continuo y lo que se necesita es una acción: regar o no regar. El umbral es la regla que traduce una cosa en la otra. 0.5 es lo normal, pero se podría mover según qué sea peor, que la planta se seque o que se riegue de más.

10. ¿Qué limitaciones tiene para representar el riego de una planta real? Usa solo dos variables y deja por fuera cosas como el tipo de planta, el suelo, la lluvia, el viento o la hora del día. Solo tiene 10 datos y no los separé en entrenamiento y prueba, así que no sé qué tan bien generaliza. La frontera de decisión es lineal. La escala fija de 100 y 50 deja de servir si la temperatura pasa de 50. Y la salida solo dice si regar o no, no cuánta agua poner.
