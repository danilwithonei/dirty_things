from ultralytics import YOLO
import cv2
import numpy as np
from typing import Any, List, Tuple


from utils import blur_detection
from utils import preprocessor


class DirtDetector:
    def __init__(self, model_path: str = "yolo11n.pt",
                 blur_threshold: int = 1330,
                 blur_img_size: Tuple[int, int] = (120, 120)):
        """
        Инициализация детектора грязи.

        :param model_path: Путь к файлу модели YOLO.
        :param blur_threshold: Пороговое значение для определения размытости изображения.
        :param blur_img_size: Размер изображения для анализа размытости.
        """
        # Загружаем модель
        self.model = YOLO(model_path)
        self.preprocessor = preprocessor.normalize
        self.blur_threshold = blur_threshold
        self.blur_img_size = blur_img_size
        self.blur_classifier = blur_detection.is_image_blured

    def predict(self, img: Any) -> Any:
        """
        Выполняет инференс на изображении.

        :param img: Исходное изображение.
        :return: Новый маска или результаты модели.
        """
        # Предобработка изображения
        img_preproc = self.preprocessor(img)

        # Проверка на размытость
        is_blured, new_mask = self.blur_classifier(
            img_preproc, self.blur_threshold, self.blur_img_size)
        if is_blured:
            return new_mask

        # Выполнение инференса с помощью модели YOLO
        results = self.model(img)
        return results

    def show_results(self, results: Any) -> None:
        """
        Отображает результаты на изображении.

        :param results: Результаты инференса.
        """

        if isinstance(results, np.ndarray):
            cv2.imshow('Result', results)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            for result in results:
                result.show()


# Пример использования
if __name__ == "__main__":
    import os

    inference = DirtDetector()

    path = 'dataset'
    for img_name in os.listdir(path):
        img = cv2.imread(os.path.join(path, img_name))

        if img is not None:
            result = inference.predict(img)

            inference.show_results(result)
        else:
            print(f"Не удалось загрузить изображение: {img_name}")
