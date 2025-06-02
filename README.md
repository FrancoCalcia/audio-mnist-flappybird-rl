# Ej 2 - TP2 - Q-Learning
El objetivo de este laboratorio es entrenar agentes para resolver videojuegos sencillos usando Q-Learning y la librería PLE.

## Preparación del entorno

1. Crear el entorno virtual:
```bash
python3 -m venv env
```

2. Activar el entorno:
```bash
source env/bin/activate
```

3. Instalar dependencias:
```bash
pip3 install -r requirements.txt
```

## Estructura del proyecto
- `test_agent.py`: Script principal para probar agentes en FlappyBird.
- `agentes/`: Carpeta con implementaciones de agentes.
    - `base.py`: Clase base para todos los agentes.
    - `random_agent.py`: Agente que toma acciones aleatorias.
    - `manual_agent.py`: Agente que permite jugar manualmente usando la barra espaciadora.

## Uso

Ejecuta un agente especificando la ruta completa de la clase:

```bash
python test_agent.py --agent agentes.random_agent.RandomAgent
```

Para jugar manualmente (salta con la barra espaciadora):

```bash
python test_agent.py --agent agentes.manual_agent.ManualAgent
```

Puedes crear tus propios agentes en la carpeta `agentes/` siguiendo la interfaz de la clase base.

## Notas
- El entorno está configurado para FlappyBird por defecto.
- Los agentes reciben la instancia del juego y la lista de acciones posibles al inicializarse.
- Para agregar un nuevo agente, crea un archivo en `agentes/` y define una clase que herede de `Agent`.
