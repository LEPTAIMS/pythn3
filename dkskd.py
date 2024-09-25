from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import serial

# Инициализация Flask приложения
app = Flask(__name__)
ser = serial.Serial("COM7", 9600)
strWrite = "|" + str(95) + "  " + str(95) + "  " + "0"
ser.write(strWrite.encode())

# Инициализация камеры
camera = cv2.VideoCapture(1)

# Инициализация YOLO модели
model = YOLO('C:/Users/sahal/Development/technopark/yolov8_3k_n/yolov8_cus/weights/best.pt')

# Включаем OpenCL для аппаратного ускорения
cv2.ocl.setUseOpenCL(True)
x = 95
y = 95
x_angles = {
    0: 18, 1: 15, 2: 12, 3: 9, 4: 6, 5: 3, 6: 0, 7: -3, 8: -6, 9: -9,
    10: -12, 11: -15, 12: -18, 13: -21, 14: -24, 15: -27
}
y_angles = {
    0: 19, 1: 15, 2: 13.5, 3: 12, 4: 10.5, 5: 9, 6: 7.5, 7: 6, 8: 4.5, 9: 3,
    10: 1.5, 11: 0, 12: -1.5, 13: -3, 14: -4.5, 15: -7
}

def get_angles(get_x, get_y):
    global x, y
    global x_index, y_index
    # Определение индекса квадратика в сетке по координатам
    x_index = get_x // 40  # Разделяем на 40, чтобы получить индекс по горизонтали
    y_index = get_y // 40  # Разделяем на 40, чтобы получить индекс по вертикали
    print(x_index, y_index)
    return x_angles.get(x_index, 0), y_angles.get(y_index, 0)


def ugl(x_center, y_center):
    global x
    global y
    global x_index, y_index
    x_center = round(x_center)
    y_center = round(y_center)

    # Получаем углы для текущих координат
    left_angle, right_angle = get_angles(x_center, y_center)
    print(get_angles(x_center, y_center))
    x = x + left_angle
    y = y + right_angle
    if x_index == 6 and y_index == 11:
        strWrite = f"|{x}  {y}  1"
        print(strWrite)
        ser.write(strWrite.encode())
    else:
        strWrite = f"|{x}  {y}  0"
        print(strWrite)
        ser.write(strWrite.encode())

    # Формируем строку для отправки в порт


# Функция для обработки кадра и детекции огня
def process_frame(frame):
    try:
        # Выполнить предсказание на текущем кадре
        results = model.predict(source=frame, conf=0.30)

        max_score = 0
        best_detection = None

        # Проверить, обнаружен ли огонь
        for result in results:
            for detection in result.boxes:
                label = int(detection.cls[0])  # ID класса
                score = detection.conf[0].cpu().numpy()  # Вероятность обнаружения
                if label == 0 and score > max_score:  # ID класса "fire" обычно равен 0
                    # Координаты и размеры обнаруженного огня
                    x_center, y_center, width, height = detection.xywh[0].cpu().numpy()

                    max_score = score
                    best_detection = (x_center, y_center, width, height)

        if best_detection is not None:
            x_center, y_center, width, height = best_detection

            # Преобразуем центр и размеры в координаты углов
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # Отобразить прямоугольник вокруг обнаруженного огня
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{max_score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # Рисуем красную точку в центре огня
            cv2.circle(frame, (int(x_center), int(y_center)), 5, (0, 0, 255), -1)
            ugl(x_center=x_center, y_center=y_center)
        return frame




    except Exception as e:
        print(f"Произошла ошибка при обработке кадра: {e}")
        return frame


# Генерация кадров для видеопотока
def gen_frames():
    frame_counter = 0  # Счётчик кадров
    while True:
        success, frame = camera.read()
        frame = cv2.resize(frame, (640, 640))
        # i = 40
        # while i <= 600:
        #     cv2.line(frame, (i, 0), (i, 640), (255, 255, 255), 1)
        #     i = i + 40
        # r = 40
        # while r <= 600:
        #     cv2.line(frame, (0, r), (640, r), (255, 255, 255), 1)
        #     r = r + 40
        # cv2.line(frame, (240, 440), (240, 480), (255, 0, 0), 2)
        # cv2.line(frame, (240, 440,), (280, 440), (255, 0, 0), 2)
        # cv2.line(frame, (280, 440), (280, 480), (255, 0, 0), 2)
        # cv2.line(frame, (240, 480), (280, 480), (255, 0, 0), 2)
        if not success:
            break
        else:
            # Каждые 150 кадров выполняем детекцию огня
            if frame_counter % 90 == 0:
                frame = process_frame(frame)
            frame_counter += 1

            # Кодирование кадра в JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Передача изображения как видеопоток
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Маршрут для главной страницы
@app.route('/')
def index():
    return render_template('cam1.html')  # Возвращаем HTML страницу


# Маршрут для видеопотока
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Запуск Flask сервера
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)