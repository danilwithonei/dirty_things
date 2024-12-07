import cv2
import numpy as np


def normalize(img: np.ndarray) -> np.ndarray:
    """
    Нормализует входное изображение и усиливает его градиенты.

    :param img: Исходное изображение в формате numpy.ndarray.
    :return: Усиленное изображение.
    """


    image_normalized = cv2.normalize(img.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    gray_image = cv2.cvtColor(image_normalized, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
    

    magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
    angle_rounded = np.round(angle / 45) * 45

    grad_x_aligned, grad_y_aligned = cv2.polarToCart(magnitude, angle_rounded, angleInDegrees=True)

    aligned_gradient = cv2.magnitude(grad_x_aligned, grad_y_aligned)
    aligned_gradient = cv2.normalize(aligned_gradient, None, 0, 1, cv2.NORM_MINMAX)
    enhanced_image = cv2.addWeighted(gray_image, 0.8, aligned_gradient, 0.2, 0)

    return enhanced_image


def experimental(img: np.ndarray, threshold: int = 200) -> np.ndarray:
    """
    Обрабатывает изображение и визуализирует размытые участки на тепловой карте.

    :param img: Исходное изображение в формате numpy.ndarray.
    :param threshold: Пороговое значение для определения размытости.
    :return: Карта тепла с обозначением размытых участков.
    """
    
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_map = cv2.absdiff(gray_image, cv2.GaussianBlur(gray_image, (11, 11), 0))
    mean_value = np.mean(blur_map)

    heatmap = cv2.applyColorMap(np.uint8(255 * (blur_map / mean_value)), cv2.COLORMAP_JET)
    mask = cv2.cvtColor(np.uint8(blur_map > threshold), cv2.COLOR_GRAY2BGR)
    result = cv2.addWeighted(heatmap, 0.7, mask * 255, 0.3, 0)
         
    return result


if __name__ == '__main__':
    # Проверка гипотезы на нескольких изображениях
    import os
    
    imgs_path = 'open_img'
    masks_path = 'open_msk'  

    for img_name in os.listdir(imgs_path):
        img_name = '.'.join(img_name.split('.')[:-1])

        img = cv2.imread('{}/{}.jpg'.format(imgs_path, img_name))
        mask = cv2.imread('{}/{}.png'.format(masks_path, img_name))

        if mask is None or img is None:
            print(img_name)
            continue

        new_img = normalize(img)
        
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cv2.imshow('img', img)
        cv2.imshow('new_img', new_img)
        cv2.imshow('gray_image', gray_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
