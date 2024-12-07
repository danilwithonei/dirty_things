import cv2
import numpy as np


def prepare_shadows_masks(img_name: str, out_folder: str, img: np.ndarray, img_no_shadow: np.ndarray) -> None:
    """
    Подготавливает маски теней для входного изображения и сохраняет их.

    :param img_name: Имя входного изображения.
    :param out_folder: Папка для сохранения полученной маски.
    :param img: Исходное изображение с тенями.
    :param img_no_shadow: Изображение без теней.
    """
    
    gray_image1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(img_no_shadow, cv2.COLOR_BGR2GRAY)
    
    difference = np.where(gray_image1 == gray_image2, 0, gray_image1)
    difference = np.where(difference != 0, 255, difference)

    kernel = np.ones((3, 3), np.uint8) 
    difference = cv2.erode(difference, kernel, iterations=10)

    difference = cv2.GaussianBlur(difference, (3, 3), 0)

    cv2.imwrite('{}/{}'.format(out_folder, img_name), difference)


def blur_original_images(img_name: str, out_folder: str, img: np.ndarray, mask: np.ndarray) -> None:
    """
    Размывает оригинальное изображение на основе маски и сохраняет результат.

    :param img_name: Имя входного изображения.
    :param out_folder: Папка для сохранения размытого изображения.
    :param img: Оригинальное изображение.
    :param mask: Маска, определяющая области размытия.
    """
    
    new_mask = cv2.GaussianBlur(mask, (51, 51), 0)
    blurred_img = cv2.GaussianBlur(img, (21, 21), 0)

    blurred = np.where(new_mask > 50, blurred_img, img)
    blurred = cv2.GaussianBlur(blurred, (3, 3), 0)

    cv2.imwrite('{}/{}'.format(out_folder, img_name), blurred)


if __name__ == '__main__':
    import os

    # Пути к папкам с изображениями и масками
    paths = ['bup_2/images', 'bup_2/noshadows', 'bup_2/masks', 'bup_2/masks_shadows', 'bup_2/images_blured']

    for img_name in os.listdir(paths[0]):
        print(img_name)

        # Чтение входного изображения и связанных масок
        img = cv2.imread('{}/{}'.format(paths[0], img_name))
        img_no_shadow = cv2.imread('{}/{}'.format(paths[1], img_name))
        mask = cv2.imread('{}/{}'.format(paths[2], img_name))

        # Подготовка масок теней и размытие оригинальных изображений
        prepare_shadows_masks(img_name, paths[3], img, img_no_shadow)
        blur_original_images(img_name, paths[4], img, mask)
