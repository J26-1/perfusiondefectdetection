#main.py
from preprocessing.data_loader import PerfusionDataset
from training.train import train_model
from inference.predictor import predict_dataset
import multiprocessing


DEV_MODE = True

CONFIG = {
    "dev": {
        "img_size": 128,
        "batch_size": 8,
        "epochs": 35,
        "lr": 5e-4,
        "mc_samples": 3,
        "save_uncertainty": True,
        "full_dataset_outputs": False,   # only selected 10 UI slices get advanced maps
        "selected_ui_slices": 10,
        "max_train_samples": 2000,        # optional cap for speed
        "max_infer_samples": 500         # optional cap for speed
    },
    "final": {
        "img_size": 128,
        "batch_size": 8,
        "epochs": 120,
        "lr": 5e-4,
        "mc_samples": 10,
        "save_uncertainty": True,
        "full_dataset_outputs": True,
        "selected_ui_slices": 10,
        "max_train_samples": None,
        "max_infer_samples": None
    }
}


def main():
    cfg = CONFIG["dev"] if DEV_MODE else CONFIG["final"]

    dataset = PerfusionDataset(
        dicom_dir="data/raw/DICOM",
        mask_dir="data/raw/NIfTI",
        img_size=cfg["img_size"],
        max_samples=cfg["max_train_samples"] if DEV_MODE else None
    )
    print("Total samples:", len(dataset))

    model = train_model(
        dataset,
        batch_size=cfg["batch_size"],
        epochs=cfg["epochs"],
        lr=cfg["lr"]
    )

    predict_dataset(
        model,
        dataset,
        output_dir="outputs",
        mc_samples=cfg["mc_samples"],
        save_uncertainty=cfg["save_uncertainty"],
        full_dataset_outputs=cfg["full_dataset_outputs"],
        selected_ui_slices=cfg["selected_ui_slices"],
        max_samples=cfg["max_infer_samples"] if DEV_MODE else None
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()