import argparse
from ultralytics import YOLO


def main(args):
    # Load a model
    model = YOLO(args.model)

    # Train the model
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

    args = parser.parse_args()
    main(args)
# python train.py --model yolo11n-seg.pt --data data.yaml --epochs 1000 --imgsz 992 --name test_test --device 0 --mosaic 0 --val False