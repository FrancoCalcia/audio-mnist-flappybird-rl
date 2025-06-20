En [este dataset](https://www.tensorflow.org/datasets/catalog/spoken_digit?hl=es-419) se presenta un conjunto de datos que contiene clips de audio correspondientes a dígitos hablados del 0 al 9. Incluye un total de 2500 clips de audio correspondientes a 5 locutores distintos, 50 clips por dígito por locutor.

El objetivo fue construir y comparar dos modelos de clasificación utilizando redes neuronales que intenten inferir el dígito correspondiente dado un clip de audio. Debimos entrenar y evaluarlos utilizando técnicas adecuadas de validación y métricas de evaluación de clasificación.

Los modelos que realizamos fueron:
- Modelo convolucional sobre los espectrogramas de los clips - CNN.
- Modelo recurrente sobre los espectrogramas de los clips - RNN (LSTM). 
