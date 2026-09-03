import cv2
import numpy as np
from ultralytics import YOLO

# 1. Cargar el modelo base de segmentación de YOLOv8
# Se descargará automáticamente 'yolov8n-seg.pt' la primera vez que lo ejecutes
print("Cargando modelo de IA...")
model = YOLO('yolov8n-seg.pt')

# 2. Iniciar la cámara (0 suele ser la cámara web principal del PC)
cap = cv2.VideoCapture(0)

# Factor de calibración simulado (Píxeles a Kilos)
# En producción para PorciTech, esto será reemplazado por tu modelo de regresión
FACTOR_PIXELES_A_KILOS = 0.0005 

print("Iniciando cámara... Presiona 'q' en la ventana de video para salir.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara. Revisa los permisos en tu sistema.")
        break

    # 3. Ejecutar inferencia en el frame actual
    # verbose=False evita que la terminal se llene de texto en cada frame
    resultados = model(frame, verbose=False)

    for resultado in resultados:
        # Verificar si el modelo logró segmentar alguna silueta (máscaras)
        if resultado.masks is not None:
            
            # Iterar sobre cada objeto detectado en pantalla
            for i, mask in enumerate(resultado.masks.xy):
                # Convertir los puntos de la máscara a un formato compatible con OpenCV
                contorno = np.array(mask, dtype=np.int32)
                
                # 4. EXTRACCIÓN DE MÉTRICAS (Matemática visual)
                # Calcular el Área en píxeles de la silueta
                area_pixeles = cv2.contourArea(contorno)
                
                # Filtrar objetos muy pequeños (ruido visual)
                if area_pixeles > 5000:
                    # Calcular la caja delimitadora para obtener Largo y Ancho
                    x, y, w, h = cv2.boundingRect(contorno)
                    
                    # 5. SIMULACIÓN DEL PESO
                    peso_estimado = area_pixeles * FACTOR_PIXELES_A_KILOS
                    
                    # 6. DIBUJAR EN PANTALLA (Interfaz)
                    # Dibujar el contorno exacto de la silueta en color verde
                    cv2.polylines(frame, [contorno], isClosed=True, color=(0, 255, 0), thickness=2)
                    
                    # Dibujar la caja delimitadora en azul
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    
                    # Mostrar los datos en tiempo real
                    texto_area = f"Area: {int(area_pixeles)} px"
                    texto_peso = f"Peso est: {peso_estimado:.2f} kg"
                    
                    cv2.putText(frame, texto_area, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(frame, texto_peso, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Mostrar el video con los cálculos superpuestos
    cv2.imshow('PorciTech - Prueba de Vision (Presiona Q para salir)', frame)

    # Detener el script al presionar la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos de hardware
cap.release()
cv2.destroyAllWindows()