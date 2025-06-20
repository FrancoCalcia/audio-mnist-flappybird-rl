En [este dataset](https://www.tensorflow.org/datasets/catalog/spoken_digit?hl=es-419) se presenta un conjunto de datos que contiene clips de audio correspondientes a dígitos hablados del 0 al 9. Incluye un total de 2500 clips de audio correspondientes a 5 locutores distintos, 50 clips por dígito por locutor.

El objetivo fue construir un modelo de clasificación utilizando redes neuronales que pueda inferir con precisión el dígito correspondiente dado un clip de audio. Debimos entrenar y evaluar modelos utilizando técnicas adecuadas de validación y métricas de evaluación de clasificación.

Entrenamos dos modelos de distintas arquitecturas y comparamos los resultados:
- Modelo convolucional sobre los espectrogramas de los clips.
- Modelo recurrente sobre los espectrogramas de los clips.
