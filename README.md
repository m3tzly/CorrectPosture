#  Detección de Postura con YOLO11n-Pose
Este proyecto implementa un sistema en **Python + OpenCV** para la **detección y monitoreo de
postura en tiempo real** utilizando el modelo **YOLO11n-pose**.
El flujo está diseñado para ser **interactivo y listo para producción**, con interfaz visual, mensajes en
pantalla y alertas sonoras/visuales cuando la postura es incorrecta.
---
##  Características principales
-  Captura de postura de referencia con cuenta regresiva.
-  Panel lateral con carrusel de imágenes (navegable con clic o teclas `a`/`s`).
-  Botón físico en pantalla para reiniciar la postura de referencia en cualquier momento.
-  Advertencias sonoras y visuales cuando se detecta una mala postura.
-  Anti-rebote de alertas para evitar repeticiones excesivas.
-  Visualización en tiempo real con keypoints del modelo YOLO.
-  Diseño adaptable con panel lateral informativo.
---
##  Requisitos
Antes de ejecutar el proyecto, asegúrate de tener instalado lo siguiente:
- Python **3.8+**
- Librerías:
pip install -r requirements.txt
- Windows (para las alertas sonoras con winsound)
> En Linux/Mac, la función de beep está protegida con try/except y no genera error.
- Modelo **YOLO11n-pose**
El script descarga el modelo automáticamente si no está en el directorio.
---
##  Ejecución
1. Clona este repositorio o copia el código:
git clone https://github.com/m3tzly/CorrectPosture
cd posture-yolo
2. Coloca tus imágenes de ejemplo para el carrusel en el mismo directorio del script, nombradas como:
1.jpg, 2.jpg, 3.jpg
> Si no existen, se mostrarán placeholders generados por el programa.
3. Ejecuta el programa:
python CorrectPosture.py
---
##  Controles
- Teclas:
- q -> Salir
- a -> Imagen anterior del carrusel
- s -> Imagen siguiente del carrusel
- Botones en pantalla:
- Anterior / Siguiente -> Navegar carrusel
- Reiniciar Ref. -A Capturar una nueva postura de referencia
---
##  Flujo del programa
1. Pantalla de preparación (15s por defecto): tiempo para colocarse frente a la cámara.
2. Cuenta regresiva (3s por defecto) antes de capturar la postura de referencia.
3. Captura de referencia: se registran los puntos clave detectados por YOLO.
4. Monitoreo en tiempo real:
- Se comparan los keypoints actuales contra la referencia.
- Si hay desviaciones mayores al umbral, se emite una advertencia sonora/visual.
- El panel lateral muestra el estado (correcto/incorrecto) de cabeza, hombros y postura general.
---
##  Configuración
Los parámetros principales se encuentran al inicio del script:
- Umbrales de tolerancia (px):
THRESHOLD_GENERAL = 50
THRESHOLD_HOMBRO = 30
- Tiempos (s):
TIEMPO_PREPARACION = 15
CUENTA_REGRESIVA_INICIAL = 3
INTERVALO_ADVERTENCIA = 3
INTERVALO_CARRUSEL = 6
- Dimensiones de ventana/panel y colores de la UI.
---
##  Estructura recomendada
- CorrectPosture
    - CorrectPosture.py
    - requirements.txt
    - 1.jpg
    - 2.jpg
    - 3.jpg
---
##  Mejoras futuras

- Entrenamiento personalizado con datasets específicos.
---
##  Autor
Proyecto desarrollado por Metzli Yunnue Domínguez Bautista  — Estudiante de Ingeniería de Software