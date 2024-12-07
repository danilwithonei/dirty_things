import os
import argparse
import cv2
from ultralytics import YOLO
from utils import preprocessor


def prepare_dataset(input: str, output: str) -> None:
    """
    Подготавливает набор данных, нормализуя изображения.

    :param input: Путь к директории с исходными изображениями.
    :param output: Путь к директории, куда будут сохранены нормализованные изображения.
    """
    if not os.path.isdir(output):
        os.mkdir(output)

    for img_name in os.listdir(input):
        img = cv2.imread(f'{input}/{img_name}')
        if img is not None:
            img = preprocessor.normalize(img)
            cv2.imwrite(f'{output}/{img_name}', img)


def main(args) -> None:
    """
    Основная функция для обучения модели YOLO.

    :param args: Аргументы командной строки.
    """
    # Загрузка модели
    model = YOLO(args.model)

    # Обучение модели
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        name=args.name,
        device=args.device,
        mosaic=args.mosaic,
        val=args.val,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a YOLO model.")

    parser.add_argument(
        "--model", type=str, default="yolo11n-seg.pt", help="Path to the model file."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data.yaml",
        help="Path to the data configuration file.",
    )
    parser.add_argument(
        "--epochs", type=int, default=1000, help="Number of training epochs."
    )
    parser.add_argument(
        "--imgsz", type=int, default=992, help="Image size for training."
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolo_n_992_no_mosaic",
        help="Name of the training run.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help='Device to train on (e.g., "0" for GPU 0).',
    )
    parser.add_argument(
        "--mosaic", type=int, default=0, help="Use mosaic augmentation (0 or 1)."
    )
    parser.add_argument(
        "--val",
        type=bool,
        default=False,
        help="Whether to validate the model during training.",
    )
    parser.add_argument(
        "--prepare_dataset_input",
        type=str,
        default="",
        help="Path to the input dataset for preparation.",
    )
    parser.add_argument(
        "--prepare_dataset_output",
        type=str,
        default="",
        help="Path to the output dataset after preparation.",
    )

    args = parser.parse_args()

    # Подготовка набора данных, если указаны соответствующие аргументы
    if len(args.prepare_dataset_input) > 0 and len(args.prepare_dataset_output) > 0:
        prepare_dataset(input=args.prepare_dataset_input,
                        output=args.prepare_dataset_output)

    # Обучение модели
    main(args)
