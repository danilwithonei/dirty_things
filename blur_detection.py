import cv2
import numpy as np
import os
from typing import Tuple, Union, List


def detect_blured(img: np.ndarray, size: Tuple[int, int] = (120, 120)) -> float:
    """
    Определяет уровень размытия изображения.
    
    :param img: Исходное изображение в формате numpy.ndarray.
    :param size: Размер для изменения изображения перед анализом (ширина, высота).
    :return: Вариация Лапласиана, показывающая уровень размытия.
    """
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, size)
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    variance = laplacian.var()
    return variance


def check_full_blur(mask: np.ndarray) -> bool:
    """
    Проверяет, является ли маска полностью черной.
    
    :param mask: Маска изображения в формате numpy.ndarray.
    :return: True, если маска полностью черная, иначе False.
    """
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    black_pixels_mask = gray_mask == 0
    black_pixel_count = np.sum(black_pixels_mask)
    return black_pixel_count == 0


def is_image_blured(img: np.ndarray, threshold: float, size: Tuple[int, int] = (120, 120)) -> Tuple[bool, Union[np.ndarray, None]]:
    """
    Определяет, является ли изображение размытым по заданному порогу.
    
    :param img: Исходное изображение в формате numpy.ndarray.
    :param threshold: Пороговое значение для определения размытия.
    :param size: Размер для изменения изображения перед анализом (ширина, высота).
    :return: Tuple, где первый элемент - True, если изображение размыто, иначе False, 
             второй элемент - маска, если изображение размыто, иначе None.
    """
    height, width = img.shape[:2]
    score = detect_blured(img, size=size)
    
    if score < threshold:
        mask = np.ones((height, width), dtype=np.uint8) * 255
        return True, mask
    else:
        return False, None


if __name__ == '__main__':
    # Проверка гипотезы
    imgs_path = 'open_img'
    masks_path = 'open_msk'

    full: List[float] = []
    not_full: List[float] = []

    counter = 0
    THRESHOLD = 1330
    SHOW = False

    for img_name in os.listdir(imgs_path):
        img_name = '.'.join(img_name.split('.')[:-1])

        img = cv2.imread(f'{imgs_path}/{img_name}.jpg')
        mask = cv2.imread(f'{masks_path}/{img_name}.png')

        if mask is None or img is None:
            print(f'Ошибка загрузки изображения: {img_name}')
            continue

        is_full_mask = check_full_blur(mask)
        blur_value = detect_blured(img)

        is_blured, new_mask = is_image_blured(img, THRESHOLD, size=(120, 120))

        if (blur_value < THRESHOLD and not is_full_mask) or (blur_value >= THRESHOLD and is_full_mask):
            print(f'Уровень размытия: {blur_value}, Полная маска: {is_full_mask}')
            counter += 1
            if SHOW:
                cv2.imshow('img', img)
                cv2.imshow('mask', mask)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        if is_full_mask:
            full.append(blur_value)
        else:
            not_full.append(blur_value)

    if full:
        full.sort()
        print(max(full), min(full), sum(full) / len(full), full[len(full) // 2], len(full))
    if not_full:
        not_full.sort()
        print(max(not_full), min(not_full), sum(not_full) / len(not_full), not_full[len(not_full) // 2], len(not_full))
    print('Счетчик ошибочных классификаций:', counter)
