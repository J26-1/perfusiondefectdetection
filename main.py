from preprocessing.data_loader import PerfusionDataset
from training.train import train_model
from inference.predictor import predict_dataset
import multiprocessing

def main():
    # -----------------------------
    # LOAD DATASET
    # -----------------------------
    dataset = PerfusionDataset(
        dicom_dir="data/raw/DICOM",
        mask_dir="data/raw/NIfTI"
    )
    print("Total samples:", len(dataset))

    # -----------------------------
    # TRAIN OR LOAD MODEL
    # -----------------------------
    model = train_model(dataset)

    # -----------------------------
    # FULL INFERENCE (ONE LINE)
    # -----------------------------
    predict_dataset(
        model,
        dataset,
        output_dir="outputs",
        mc_samples=10,       # Monte Carlo dropout samples
        save_uncertainty=True
    )

if __name__ == "__main__":
    multiprocessing.freeze_support()  # Required for Windows
    main()