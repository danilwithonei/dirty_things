from ultralytics import YOLO

class YOLOInference:
    def __init__(self, model_path):
        # Загружаем модель
        self.model = YOLO(model_path)

    def predict(self, image_path):
        # Выполняем инференс на изображении
        results = self.model(image_path)
        return results

    def show_results(self, results):
        # Отображаем результаты
        for result in results:
            result.show()

# Пример использования
if __name__ == "__main__":
    # Создаем экземпляр класса
    yolo_inference = YOLOInference("yolo11n.pt")
    
    # Выполняем предсказание
    results = yolo_inference.predict("path/to/image.jpg")
    
    # Отображаем результаты
    yolo_inference.show_results(results)