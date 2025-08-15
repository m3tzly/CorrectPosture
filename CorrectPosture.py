# -*- coding: utf-8 -*-
"""
Detección de postura con YOLO11n-pose
- Flujo completo refactorizado y listo para producción.
- Panel lateral con carrusel (click y teclas 'a'/'s'), mensajes y estado.
- Captura de postura de referencia con cuenta regresiva.
- Advertencias sonoras/visuales con anti-rebote.
"""

from ultralytics import YOLO
from imutils.video import VideoStream
import numpy as np
import cv2
import time
import winsound
import os
from typing import Optional, Tuple, List, Dict, Union

# ===================== CONFIGURACIÓN =====================
# Umbrales (px) — ajusta según tu distancia a cámara
THRESHOLD_GENERAL = 50
THRESHOLD_HOMBRO = 30

# Tiempos (s)
TIEMPO_PREPARACION = 15           # Pantalla inicial antes de la cuenta regresiva
CUENTA_REGRESIVA_INICIAL = 3      # Para capturar postura de referencia
INTERVALO_ADVERTENCIA = 3         # Anti-rebote de advertencias
INTERVALO_CARRUSEL = 6            # Autoplay del carrusel

# Ventana / Layout
ANCHO_FRAME = 800
ALTO_FRAME = 720
ANCHO_PANEL = 400
ALTO_PANEL = 720
TITULO_VENTANA = "Deteccion de Postura"

# UI / Estilos
COLOR_BOTON = (79, 55, 138)
COLOR_BOTON2 = (162, 28, 165)
COLOR_TEXTO_BOTON = (255, 255, 255)
COLORES_PUNTOS = [(255, 0, 255), (0, 0, 255), (0, 255, 0), (64, 128, 255), (128, 64, 255)]
RADIO_PUNTO = 4  # puntos de keypoints

# Rutas carrusel 
CARRUSEL_ITEMS = [f"{i}.jpg" for i in range(1, 4)]


# ===================== UTILIDADES =====================
def safe_beep(freq=1000, ms=500):
    try:
        winsound.Beep(freq, ms)
    except Exception:
        pass


def fit_frame(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    """Asegura BGR 3 canales y resize."""
    if frame is None:
        return None
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return cv2.resize(frame, (w, h))


def group_color(idx: int) -> Tuple[int, int, int]:
    """Colorea keypoints por grupos (más estable visualmente)."""
    return COLORES_PUNTOS[min(idx // 2, len(COLORES_PUNTOS) - 1)]


def l2_mean_diff(ref: np.ndarray, cur: np.ndarray, indices: Union[slice, List[int], None] = None) -> float:
    r = ref[indices] if indices is not None else ref
    c = cur[indices] if indices is not None else cur
    return float(np.linalg.norm(r - c, axis=1).mean()) if len(r) > 0 else 0.0


# ===================== APLICACIÓN =====================
class PostureApp:
    def __init__(self):
        # Modelo
        self.model = YOLO("yolo11n-pose.pt")

        # Cámara
        self.cam = VideoStream(src=0).start()
        time.sleep(2.0)

        # Estado postura
        self.postura_ref: Optional[np.ndarray] = None
        self.ultima_advertencia = 0.0

        # Carrusel
        self.carrusel_imgs: List[np.ndarray] = self._load_carousel_images(CARRUSEL_ITEMS)
        self.carrusel_idx = 0
        self.carrusel_t = time.time()
        self.btn_prev: Optional[Tuple[int, int, int, int]] = None
        self.btn_next: Optional[Tuple[int, int, int, int]] = None

        # Ventana
        cv2.namedWindow(TITULO_VENTANA, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(TITULO_VENTANA, ANCHO_FRAME + ANCHO_PANEL, ALTO_FRAME)
        cv2.setMouseCallback(TITULO_VENTANA, self.on_mouse)

        #Boton recaptura
        self.btn_reset_ref: Optional[Tuple[int, int, int, int]] = None
        self.en_bucle_principal = False  # Estado para mostrar/ocultar botón

    # ---------- Inicialización ----------
    @staticmethod
    def _load_carousel_images(paths: List[str]) -> List[np.ndarray]:
        imgs = []
        for i, p in enumerate(paths, start=1):
            if os.path.exists(p):
                img = cv2.imread(p)
                img = cv2.resize(img, (300, 300))
            else:
                img = np.zeros((300, 300, 3), dtype=np.uint8)
                cv2.putText(img, f"Ejemplo postura {i}", (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            imgs.append(img)
        return imgs

    # ---------- UI ----------
    @staticmethod
    def _rounded_button(img, text, pos, size, bg, fg) -> Tuple[int, int, int, int]:
        x, y = pos
        w, h = size
        r = 15
        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), bg, -1)
        cv2.rectangle(img, (x, y + r), (x + w, y + h - r), bg, -1)
        for cx, cy in [(x + r, y + r), (x + w - r, y + r), (x + r, y + h - r), (x + w - r, y + h - r)]:
            cv2.circle(img, (cx, cy), r, bg, -1)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(img, text, (x + (w - tw) // 2, y + (h + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, fg, 2)
        return (x, y, w, h)

    def _panel(self, titulo: str, subtitulo: str = "", posture_info: Optional[Dict[str, str]] = None) -> np.ndarray:
        panel = np.ones((ALTO_PANEL, ANCHO_PANEL, 3), dtype=np.uint8) * 240

        # Color dinámico del título si la postura es incorrecta
        color_title = (0, 0, 255) if (posture_info and "incorrecta" in posture_info.get("Postura general", "").lower()) else (0, 0, 0)

        # --- TÍTULO ---
        y_offset = 40
        for linea in titulo.split("\n"):
            cv2.putText(panel, linea, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, color_title, 2)
            y_offset += 40  # Espaciado entre líneas

        # --- SUBTÍTULO ---
        if subtitulo:
            for linea in subtitulo.split("\n"):
                cv2.putText(panel, linea, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                y_offset += 25  # Espaciado entre líneas

        # --- INFO DE POSTURA ---
        if posture_info:
            y_offset += 10
            for k, v in posture_info.items():
                col = (0, 255, 0) if "correcta" in v.lower() else (0, 0, 255)
                cv2.putText(panel, f"{k}: {v}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1)
                y_offset += 30

            y_offset += 20
            cv2.line(panel, (10, y_offset), (ANCHO_PANEL - 10, y_offset), (210, 210, 210), 2)
            y_offset += 20

        # --- CARRUSEL ---
        img = self.carrusel_imgs[self.carrusel_idx]
        x_center = (ANCHO_PANEL - img.shape[1]) // 2
        panel[y_offset:y_offset + img.shape[0], x_center:x_center + img.shape[1]] = img
        y_offset += img.shape[0] + 20

        # --- BOTONES DEL CARRUSEL ---
        self.btn_prev = self._rounded_button(panel, "Anterior", (50, y_offset), (120, 50), COLOR_BOTON, COLOR_TEXTO_BOTON)
        self.btn_next = self._rounded_button(panel, "Siguiente", (ANCHO_PANEL - 170, y_offset), (120, 50), COLOR_BOTON, COLOR_TEXTO_BOTON)
        y_offset += 70

        # --- BOTÓN DE REINICIAR REFERENCIA (solo en bucle principal) ---
        if self.en_bucle_principal:
            self.btn_reset_ref = self._rounded_button(panel, "Reiniciar Ref.", 
                                                    (ANCHO_PANEL // 2 - 80, y_offset), (160, 50),
                                                    COLOR_BOTON2, COLOR_TEXTO_BOTON)
        else:
            self.btn_reset_ref = None

        return panel


    def _warning(self, mensaje: str):
        win = np.zeros((200, 600, 3), dtype=np.uint8)
        cv2.putText(win, "ADVERTENCIA", (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(win, mensaje, (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Advertencia", win)
        cv2.waitKey(1200)
        cv2.destroyWindow("Advertencia")
        safe_beep(1000, 400)

    def _stack(self, frame: np.ndarray, panel: np.ndarray) -> np.ndarray:
        # Autoplay carrusel
        if time.time() - self.carrusel_t > INTERVALO_CARRUSEL:
            self.carrusel_idx = (self.carrusel_idx + 1) % len(self.carrusel_imgs)
            self.carrusel_t = time.time()
        return np.hstack((frame, panel))

    # ---------- Eventos ----------
    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        px = x - ANCHO_FRAME
        py = y

        # Botones carrusel
        if self.btn_prev and (self.btn_prev[0] <= px <= self.btn_prev[0] + self.btn_prev[2]) and (self.btn_prev[1] <= py <= self.btn_prev[1] + self.btn_prev[3]):
            self.carrusel_idx = (self.carrusel_idx - 1) % len(self.carrusel_imgs)
            self.carrusel_t = time.time()
        elif self.btn_next and (self.btn_next[0] <= px <= self.btn_next[0] + self.btn_next[2]) and (self.btn_next[1] <= py <= self.btn_next[1] + self.btn_next[3]):
            self.carrusel_idx = (self.carrusel_idx + 1) % len(self.carrusel_imgs)
            self.carrusel_t = time.time()

        # Botón de reinicio de referencia
        elif self.btn_reset_ref and (self.btn_reset_ref[0] <= px <= self.btn_reset_ref[0] + self.btn_reset_ref[2]) and (self.btn_reset_ref[1] <= py <= self.btn_reset_ref[1] + self.btn_reset_ref[3]):
            print("Reiniciando referencia...")
            self.reiniciar_referencia()

    # ---------- Flujo ----------
    def pantalla_preparacion(self, segundos: int = TIEMPO_PREPARACION) -> bool:
        t0 = time.time()
        while time.time() - t0 < segundos:
            frame = fit_frame(self.cam.read(), ANCHO_FRAME, ALTO_FRAME)
            frame = cv2.flip(frame, 1)
            if frame is None:
                print("Error: No se pudo obtener el frame de la cámara.")
                return False
            rem = max(0, segundos - int(time.time() - t0))
            p = self._panel("Preparando para \ncapturar postura \nde referencia",
                            f"El programa comenzara en \n{rem} segundos...", None)
            out = self._stack(frame, p)
            cv2.imshow(TITULO_VENTANA, out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return False
            elif key == ord('a'):
                self.carrusel_idx = (self.carrusel_idx - 1) % len(self.carrusel_imgs); self.carrusel_t = time.time()
            elif key == ord('s'):
                self.carrusel_idx = (self.carrusel_idx + 1) % len(self.carrusel_imgs); self.carrusel_t = time.time()
        return True
    
    def reiniciar_referencia(self):
        """Vuelve a capturar la postura de referencia con cuenta regresiva."""
        if not self.cuenta_regresiva_referencia(CUENTA_REGRESIVA_INICIAL):
            return
        for _ in range(10):
            if self.capturar_referencia():
                print("Nueva referencia capturada.")
                return
            time.sleep(0.1)
        print("No fue posible capturar nueva referencia.")

    def cuenta_regresiva_referencia(self, segundos: int = CUENTA_REGRESIVA_INICIAL) -> bool:
        ultimo = -1
        t0 = time.time()
        while True:
            frame = fit_frame(self.cam.read(), ANCHO_FRAME, ALTO_FRAME)
            frame = cv2.flip(frame, 1)
            if frame is None:
                print("Error: No se pudo obtener el frame de la cámara.")
                return False
            trans = time.time() - t0
            rem = segundos - int(trans)
            if rem <= 0:
                return True
            if rem != ultimo:
                ultimo = rem
                print(f"Se tomara pose de referencia en {rem} segundos...")

            p = self._panel("Asegurate de tener \nuna postura correcta",
                            f"Se tomara pose de referencia en \n{rem} segundos...", None)
            out = self._stack(frame, p)
            cv2.imshow(TITULO_VENTANA, out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False

    def capturar_referencia(self) -> bool:
        """Toma un frame, corre YOLO y fija la postura de referencia."""
        frame = fit_frame(self.cam.read(), ANCHO_FRAME, ALTO_FRAME)
        frame = cv2.flip(frame, 1)
        if frame is None:
            print("Error: No se pudo obtener el frame de la camara.")
            return False
        results = self.model.predict(frame, verbose=False)
        for r in results:
            kps = r.keypoints
            if kps is None or kps.xy is None:
                continue
            xy = kps.xy.cpu().numpy()
            if xy.shape[0] == 0:
                continue
            self.postura_ref = xy[0]  # (num_kps, 2)
            return True
        print("No se detectaron puntos para referencia. Intenta de nuevo.")
        return False

    def bucle_principal(self):
        self.en_bucle_principal = True  # Mostrar botón de reinicio
        fps_t0, fps_frames = time.time(), 0
        while True:
            frame = fit_frame(self.cam.read(), ANCHO_FRAME, ALTO_FRAME)
            frame = cv2.flip(frame, 1)
            if frame is None:
                print("Error: No se pudo obtener el frame de la camara.")
                break

            results = self.model.predict(frame, verbose=False)
            posture_info = {
                "Postura cabeza": "correcta",
                "Postura hombro derecho": "correcta",
                "Postura hombro izquierdo": "correcta",
                "Postura general": "correcta"
            }

            for r in results:
                kps = r.keypoints
                if kps is None or kps.xy is None:
                    continue
                xy = kps.xy.cpu().numpy()
                if xy.shape[0] == 0:
                    continue
                pts = xy[0]

                # Dibuja keypoints
                for i, (x, y) in enumerate(pts):
                    cv2.circle(frame, (int(x), int(y)), RADIO_PUNTO, group_color(i), -1)

                if self.postura_ref is None:
                    self.postura_ref = pts.copy()
                    continue

                dif_general = l2_mean_diff(self.postura_ref, pts)
                dif_cabeza = l2_mean_diff(self.postura_ref, pts, slice(0, 3))
                dif_hombro_der = l2_mean_diff(self.postura_ref, pts, [4, 6])
                dif_hombro_izq = l2_mean_diff(self.postura_ref, pts, [3, 5])

                posture_info["Postura general"] = "correcta" if dif_general <= THRESHOLD_GENERAL else "incorrecta"
                posture_info["Postura cabeza"] = "correcta" if dif_cabeza <= THRESHOLD_GENERAL else "incorrecta"
                posture_info["Postura hombro derecho"] = "correcta" if dif_hombro_der <= THRESHOLD_HOMBRO else "incorrecta"
                posture_info["Postura hombro izquierdo"] = "correcta" if dif_hombro_izq <= THRESHOLD_HOMBRO else "incorrecta"

                if (dif_general > THRESHOLD_GENERAL or dif_cabeza > THRESHOLD_GENERAL) and (time.time() - self.ultima_advertencia > INTERVALO_ADVERTENCIA):
                    self._warning("Endereza tu postura")
                    self.ultima_advertencia = time.time()

            fps_frames += 1
            if time.time() - fps_t0 >= 1.0:
                fps = fps_frames / (time.time() - fps_t0)
                fps_t0 = time.time()
                fps_frames = 0
                cv2.putText(frame, f"FPS ~ {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 200, 30), 2)

            panel = self._panel("Postura correcta" if posture_info["Postura general"] == "correcta" else "Postura incorrecta",
                                "", posture_info)
            out = self._stack(frame, panel)
            cv2.imshow(TITULO_VENTANA, out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                self.carrusel_idx = (self.carrusel_idx - 1) % len(self.carrusel_imgs); self.carrusel_t = time.time()
            elif key == ord('s'):
                self.carrusel_idx = (self.carrusel_idx + 1) % len(self.carrusel_imgs); self.carrusel_t = time.time()

        self.en_bucle_principal = False

    # ---------- Run ----------
    def run(self):
        try:
            if not self.pantalla_preparacion(TIEMPO_PREPARACION):
                return
            if not self.cuenta_regresiva_referencia(CUENTA_REGRESIVA_INICIAL):
                return
            # Intentos para fijar referencia (por si el primer frame falla)
            for _ in range(10):
                if self.capturar_referencia():
                    break
                time.sleep(0.1)
            if self.postura_ref is None:
                print("No fue posible capturar postura de referencia. Saliendo.")
                return
            self.bucle_principal()
        finally:
            cv2.destroyAllWindows()
            try:
                self.cam.stop()
            except Exception:
                pass


if __name__ == "__main__":
    app = PostureApp()
    app.run()