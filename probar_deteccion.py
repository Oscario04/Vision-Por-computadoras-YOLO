import cv2
from ultralytics import YOLO

IP = "10.207.201.70"
cap = cv2.VideoCapture(f"http://{IP}:8080/video")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

model = YOLO("yolo26n.pt")

# Descarta unos frames viejos y toma uno actual
for _ in range(10):
    ok, frame = cap.read()
assert ok, "No se pudo leer el video"

frame = cv2.resize(frame, (640, 360))
cv2.imwrite("snapshot.jpg", frame)          # revisa este archivo

res = model(frame, conf=0.10, verbose=False)[0]   # todas las clases, confianza baja
print("Detectó", len(res.boxes), "objetos:")
for b in res.boxes:
    print(" -", model.names[int(b.cls)], f"{float(b.conf):.2f}")

cv2.imwrite("deteccion.jpg", res.plot())
cap.release()