import os
import random
import argparse

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Функция для трансформации изображения: изменение размера и поворот
def transform_img(img: Image.Image, background: Image.Image) -> Image.Image:
    bg_width, bg_height = background.size
    # Изменяем размер изображения до 40% от размера фона
    img = img.resize((int(bg_width * 0.4), int(bg_height * 0.4)))
    angle = random.randint(0, 360)
    # Поворачиваем изображение на случайный угол
    img = img.rotate(angle, expand=True)
    return img

# Функция для вставки изображения в случайное место на фоне
def paste_in_random_place(
    background: Image.Image, paste_img: Image.Image
) -> Image.Image:
    bg_width, bg_height = background.size
    png_width, png_height = paste_img.size

    # Генерация случайных координат для вставки изображения
    x = random.randint(0, bg_width - png_width)
    y = random.randint(0, bg_height - png_height)

    # Вставка PNG-изображения на основное изображение с учетом альфа-канала
    background.paste(paste_img, (x, y), paste_img)
    return background

# Функция для получения маски, показывающей разницу между фоном и фоном с вставленным изображением
def get_mask(
    background: Image.Image, background_with_paste: Image.Image
) -> Image.Image:
    # Преобразуем изображения в массивы NumPy для обработки
    background = np.asarray(background)
    background_with_paste = np.asarray(background_with_paste)

    # Преобразуем изображения в оттенки серого
    gray_image2 = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    gray_image1 = cv2.cvtColor(background_with_paste, cv2.COLOR_BGR2GRAY)

    # Вычисляем разницу между изображениями: 0 - одинаковые пиксели, 255 - разные
    difference = np.where(gray_image1 == gray_image2, 0, 255).astype(np.uint8)
    mask = Image.fromarray(difference)
    return mask

# Основная функция для обработки изображений
def main(
    images_dir_path: str,
    paste_images_dir_path: str,
    path_to_save_images: str,
    path_to_save_masks: str,
):
    # Список для хранения вставляемых изображений
    paste_images: list[Image.Image] = []
    print("Загрузка изображений")
    # Загружаем все изображения из директории вставляемых изображений
    for img_name in tqdm(os.listdir(paste_images_dir_path)):
        img_path = os.path.join(paste_images_dir_path, img_name)
        paste_img = Image.open(img_path)
        paste_images.append(paste_img)

    print("Генерация датасета")
    # Проходим по всем изображениям в основной директории
    for img_name in tqdm(os.listdir(images_dir_path)):
        img_path = os.path.join(images_dir_path, img_name)
        background = Image.open(img_path)  # Загружаем фоновое изображение
        paste_img = random.choice(paste_images)  # Случайно выбираем изображение для вставки
        paste_img = transform_img(paste_img.copy(), background.copy())  # Трансформируем изображение
        # Вставляем изображение в случайное место на фоне
        background_with_paste = paste_in_random_place(
            background=background.copy(),
            paste_img=paste_img.copy(),
        )
        # Получаем маску, показывающую разницу между фоном и фоном с вставленным изображением
        mask = get_mask(
            background=background.copy(),
            background_with_paste=background_with_paste.copy(),
        )
        # Сохраняем изображение с вставленным элементом
        background_with_paste_path = os.path.join(path_to_save_images, img_name)
        # Сохраняем маску
        mask_path = os.path.join(path_to_save_masks, img_name.replace(".jpg","") + ".png")
        background_with_paste.save(background_with_paste_path)
        mask.save(mask_path)

# Проверяем, запущен ли скрипт напрямую
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Скрипт для обработки изображений и масок."
    )

    # Добавляем аргументы
    parser.add_argument(
        "--images_dir_path",
        type=str,
        required=True,
        help="Путь к директории с изображениями.",
    )
    parser.add_argument(
        "--paste_images_dir_path",
        type=str,
        required=True,
        help="Путь к директории для вставляемых изображений.",
    )
    parser.add_argument(
        "--path_to_save_images",
        type=str,
        required=True,
        help="Путь для сохранения обработанных изображений.",
    )
    parser.add_argument(
        "--path_to_save_masks",
        type=str,
        required=True,
        help="Путь для сохранения масок.",
    )

    # Парсим аргументы
    args = parser.parse_args()

    images_dir_path = args.images_dir_path
    paste_images_dir_path = args.paste_images_dir_path
    path_to_save_images = args.path_to_save_images
    path_to_save_masks = args.path_to_save_masks

    if not os.path.exists(path_to_save_images):
        os.makedirs(path_to_save_images, exist_ok=True)
        print(f"Изображения сохраняются в:{path_to_save_images}")

    if not os.path.exists(path_to_save_masks):
        os.makedirs(path_to_save_masks, exist_ok=True)
        print(f"Маски сохраняются в:{path_to_save_masks}")

    main(
        images_dir_path,
        paste_images_dir_path,
        path_to_save_images,
        path_to_save_masks,
    )
# python dataset_generator.py --images_dir_path just_images --paste_images_dir_path paste_imgs --path_to_save_images test_image --path_to_save_masks test_masks