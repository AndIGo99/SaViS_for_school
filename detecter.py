import cv2
from utils.general import non_max_suppression
from models.experimental import attempt_load
from utils.torch_utils import select_device
import torch
from playsound import playsound  # Для воспроизведения звука

# Загрузка модели
weights = 'runs/train/exp/weights/best.pt'
device = select_device('')
model = attempt_load(weights, map_location=device)

# Функция для воспроизведения звукового сигнала
def play_alarm_sound():
    playsound('alarm_sound.mp3')  # Укажите путь к файлу со звуком

# Открытие видеопотока
cap = cv2.VideoCapture(0)  # 0 для веб-камеры

# Флаг для предотвращения повторного воспроизведения звука
alarm_triggered = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Предобработка изображения
    img = cv2.resize(frame, (640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  # Normalize
    img = img.unsqueeze(0)

    # Детекция
    with torch.no_grad():
        pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    # Сброс флага тревоги
    alarm_triggered = False

    # Визуализация результатов
    for det in pred:
        if det is not None and len(det):
            h, w, _ = frame.shape  # Размеры исходного кадра
            quarter_height = h / 4  # Четверть высоты кадра (для сравнения размера объектов)

            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)  # Координаты bounding box
                obj_width = x2 - x1
                obj_height = y2 - y1

                # Определение типа объекта
                class_name = model.names[int(cls)]
                dangerous_from_coco = ['knife','scissors']
                dangerous_from_imagenet = ['acoustic guitar','assault rifle, assault gun', 'bulletproof vest','chainsaw', 'hammer', 'hatchet', 'military uniform','mortar', 'revolver', 'rifle']
                dangerous = dangerous_from_coco+dangerous_from_imagenet
                dangerous.append("weapon")
                is_dangerous = class_name in dangerous  #  Опасные классы
                is_large = obj_height >= quarter_height  # Большой объект
                is_small = obj_height < quarter_height  # Маленький объект

                # Выбор цвета и толщины рамки
                if conf >= 0.75:
                    if is_dangerous:
                        color = (0, 0, 255)  # Красный
                        thickness = 4  # Толстая рамка
                        alarm_triggered = True  # Тревога
                    else:
                        color = (0, 255, 0)  # Зеленый
                        thickness = 2  # Тонкая рамка
                elif 0.5 <= conf < 0.75:
                    if is_dangerous:
                        color = (0, 0, 255)  # Красный
                        thickness = 2  # Тонкая рамка
                        alarm_triggered = True  # Тревога
                    else:
                        color = (0, 255, 255)  # Желтый
                        thickness = 2  # Тонкая рамка
                else:  # conf < 0.5
                    if is_large:
                        color = (0, 165, 255)  # Оранжевый
                        thickness = 4  # Толстая рамка
                    else:
                        color = (255, 0, 0)  # Синий
                        thickness = 4  # Толстая рамка

                # Рисуем рамку
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # Добавляем текст с названием класса и уровнем уверенности
                label = f'{class_name} {conf:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Если обнаружен опасный объект, воспроизводим звуковой сигнал
    if alarm_triggered:
        play_alarm_sound()

    # Добавляем предупреждение на экран
    if alarm_triggered:
        warning_text = "WARNING: POTENTIALLY DANGEROUS OBJECT DETECTED!"
        cv2.putText(frame, warning_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Отображение кадра
    cv2.imshow('YOLOv5 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 
