import time
from datetime import datetime
import cv2
from ultralytics import YOLO, solutions

IP      = "10.207.201.70"      # confirma la IP actual en IP Webcam
PUERTO  = 8080
OBJETOS = []                   # [] = contar todas las clases; luego puedes filtrar
LINEA   = "horizontal"         # "vertical" | "horizontal"
MODELO  = "yolo26n.pt"
ANCHO, ALTO = 640, 360

FPS_SALIDA   = 20              # ajústalo a los FPS reales que ves en consola
GUARDAR_RAW  = True            # también guarda el video sin anotar

cap = cv2.VideoCapture(f"http://{IP}:{PUERTO}/video")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
ok, frame = cap.read()
assert ok, "No se pudo conectar"

w, h = ANCHO, ALTO
region = [(w // 2, 0), (w // 2, h)] if LINEA == "vertical" else [(0, h // 2), (w, h // 2)]

nombres = YOLO(MODELO).names
ids = [i for i, n in nombres.items() if n in OBJETOS] or None

counter = solutions.ObjectCounter(
    show=False, region=region, model=MODELO,
    classes=ids, conf=0.15, line_width=2, verbose=False,
)

# --- Grabación ---
sello = datetime.now().strftime("%Y%m%d_%H%M%S")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_det = cv2.VideoWriter(f"conteo_{sello}.mp4", fourcc, FPS_SALIDA, (w, h))
out_raw = cv2.VideoWriter(f"raw_{sello}.mp4", fourcc, FPS_SALIDA, (w, h)) if GUARDAR_RAW else None

print("Presiona 'q' en la ventana para salir.")
n, t0 = 0, time.time()
try:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            print("Se perdió la conexión.")
            break

        frame = cv2.resize(frame, (w, h))
        if out_raw is not None:
            out_raw.write(frame)               # original, antes de dibujar

        results = counter(frame)
        annotated = results.plot_im            # SIN paréntesis

        cv2.line(annotated, region[0], region[1], (0, 0, 255), 2)
        texto = f"IN {counter.in_count}  OUT {counter.out_count}"
        cv2.putText(annotated, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out_det.write(annotated)               # video con detecciones y conteo

        n += 1
        if n % 30 == 0:
            print(f"FPS: {n / (time.time() - t0):.1f} | {texto}")

        cv2.imshow("Conteo con YOLO", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    out_det.release()                          # sin esto el .mp4 queda corrupto
    if out_raw is not None:
        out_raw.release()
    cv2.destroyAllWindows()
    print("Entradas:", counter.in_count, "| Salidas:", counter.out_count)
    print("Por clase:", counter.classwise_count)
    print(f"Guardado: conteo_{sello}.mp4")