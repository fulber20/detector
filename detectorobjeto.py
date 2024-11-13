import cv2
import numpy as np

class_translation = {
    'person': 'persona',
    'bicycle': 'bicicleta',
    'car': 'coche',
    'motorbike': 'motocicleta',
    'aeroplane': 'avión',
    'bus': 'autobús',
    'train': 'tren',
    'truck': 'camión',
    'boat': 'barco',
    'traffic light': 'semáforo',
    'fire hydrant': 'hidrante de fuego',
    'stop sign': 'señal de pare',
    'parking meter': 'parquímetro',
    'bench': 'banco',
    'bird': 'pájaro',
    'cat': 'gato',
    'dog': 'perro',
    'horse': 'caballo',
    'sheep': 'oveja',
    'cow': 'vaca',
    'elephant': 'elefante',
    'bear': 'oso',
    'zebra': 'cebra',
    'giraffe': 'jirafa',
    'backpack': 'mochila',
    'umbrella': 'sombrilla',
    'handbag': 'bolso',
    'tie': 'corbata',
    'suitcase': 'maleta',
    'frisbee': 'frisbee',
    'skis': 'esquís',
    'snowboard': 'tabla de snowboard',
    'sports ball': 'pelota deportiva',
    'kite': 'cometa',
    'baseball bat': 'bate de béisbol',
    'baseball glove': 'guante de béisbol',
    'skateboard': 'patineta',
    'surfboard': 'tabla de surf',
    'tennis racket': 'raqueta de tenis',
    'bottle': 'botella',
    'wine glass': 'copa de vino',
    'cup': 'taza',
    'fork': 'tenedor',
    'knife': 'cuchillo',
    'spoon': 'cucharita',
    'bowl': 'tazón',
    'banana': 'plátano',
    'apple': 'manzana',
    'sandwich': 'sándwich',
    'orange': 'naranja',
    'broccoli': 'brócoli',
    'carrot': 'zanahoria',
    'hot dog': 'perrito caliente',
    'pizza': 'pizza',
    'donut': 'rosquilla',
    'cake': 'pastel',
    'chair': 'silla',
    'sofa': 'sofá',
    'pottedplant': 'planta en maceta',
    'bed': 'cama',
    'diningtable': 'mesa de comedor',
    'toilet': 'inodoro',
    'tvmonitor': 'monitor de TV',
    'laptop': 'computadora portátil',
    'mouse': 'ratón',
    'remote': 'control remoto',
    'keyboard': 'teclado',
    'cell phone': 'teléfono móvil',
    'microwave': 'microondas',
    'oven': 'horno',
    'toaster': 'tostadora',
    'sink': 'fregadero',
    'refrigerator': 'nevera',
    'book': 'libro',
    'clock': 'reloj',
    'vase': 'florero',
    'scissors': 'tijeras',
    'teddy bear': 'oso de peluche',
    'hair drier': 'secador de pelo',
    'toothbrush': 'cepillo de dientes',
    'Marker': 'marcador',
    'Remote control': 'Control remoto',
    'Headphones': 'auriculares',
    'clothes iron':'plancha de ropar',
}

net = cv2.dnn.readNet('colaca la ruta  del archiovo decargado yolov3.weights', 'yolov3.cfg')

layer_names = net.getLayerNames()

output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

classes = []

with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

translated_classes = [class_translation.get(cls, cls) for cls in classes]

cap = cv2.VideoCapture(0)  

while True:

    ret, frame = cap.read()

    if not ret:
        print("No se puede capturar el frame. Verifica la cámara.")
        break

    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5: 
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indices) > 0:
        indices = indices.flatten()  

        for i in indices:
            x, y, w, h = boxes[i]
            label = translated_classes[class_ids[i]] 
            confidence = confidences[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence*100:.2f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Real-Time Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
