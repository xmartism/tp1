import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pytorch_forecasting import TimeSeriesDataSet, DeepAR
from pytorch_forecasting.metrics import NormalDistributionLoss
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch import seed_everything
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

DATASET_MODE = "WEATHER"  # "SINE" or "WEATHER"
WEATHER_FILE = "weather.csv"
MAX_EPOCHS = 30

def get_data(mode):
    if mode == "SINE":
        # Simple sine wave
        timesteps = 1000
        time_idx = np.arange(timesteps)
        target = np.sin(0.05 * time_idx) + np.random.normal(0, 0.02, timesteps)
        df = pd.DataFrame({
            "time_idx": time_idx,
            "target": target,
            "series_id": "0"
        })
        dataset_name = "sine_wave"
        return df, dataset_name
    
    else:  # WEATHER mode
        # Read weather data and clean ALL column names
        df = pd.read_csv(WEATHER_FILE)
        
        # Clean ALL column names
        df.columns = [col.replace('.', '_').replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_').replace('%', 'pct') for col in df.columns]
        
        # Basic preprocessing
        df["date"] = pd.to_datetime(df["date"])
        df["time_idx"] = np.arange(len(df))
        df["series_id"] = "0"
        
        # Use temperature as target
        df["target"] = df["T_degC"]
        
        # Simple NaN handling
        df["target"] = df["target"].ffill().bfill()
        
        # Add time features
        df["hour"] = df["date"].dt.hour.astype(str).astype("category")
        df["day_of_week"] = df["date"].dt.dayofweek.astype(str).astype("category")
        
        dataset_name = "weather_data"
        return df, dataset_name

def create_datasets(data, max_prediction_length=24):
    max_encoder_length = 168  # 7 days * 24 hours
    
    # Training cutoff
    training_cutoff = data["time_idx"].max() - max_prediction_length
    
    # Weather features for multivariate prediction
    weather_features = []
    for feature in ['p_mbar', 'rh_pct', 'Tdew_degC', 'VPact_mbar']:
        if feature in data.columns:
            # Clean feature
            data[feature] = data[feature].ffill().bfill().fillna(0)
            weather_features.append(feature)
    
    print(f"Using weather features: {weather_features}")  
  
    # Create training dataset
    training = TimeSeriesDataSet(
        data[data.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="target",
        group_ids=["series_id"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["series_id"],
        time_varying_known_categoricals=["hour", "day_of_week"],
        time_varying_known_reals=weather_features,
        time_varying_unknown_reals=["target"],
        add_relative_time_idx=True,
        add_target_scales=True
    )
    
    # Create validation dataset
    validation = TimeSeriesDataSet.from_dataset(
        training, 
        data, 
        predict=True, 
        stop_randomization=True
    )
    
    return training, validation


def train_new_model(training, validation, max_epochs=MAX_EPOCHS):
    # Set seed for reproducibility
    seed_everything(42, workers=True)
    
    # Create model
    model = DeepAR.from_dataset(
        training,
        learning_rate=0.0001,
        hidden_size=64,
        rnn_layers=2,
        loss=NormalDistributionLoss(),
        dropout=0.1
    )
    
    # Setup callbacks
    early_stop = EarlyStopping(
        monitor="val_loss", 
        patience=5, 
        mode="min",
        verbose=True
    )

    # Create trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        gradient_clip_val=0.1,
        callbacks=[early_stop],
        enable_progress_bar=True
    )
    
    # Create data loaders
    train_loader = training.to_dataloader(train=True, batch_size=32)
    val_loader = validation.to_dataloader(train=False, batch_size=32)
    
    # Train the model
    print("Starting training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    return model

def evaluate_and_save(model, validation, dataset_name):

    val_loader = validation.to_dataloader(train=False, batch_size=64)
    predictions = model.predict(val_loader, return_y=True)

    y_true = predictions.y[0].cpu().numpy().flatten() if isinstance(predictions.y, tuple) else predictions.y.cpu().numpy().flatten()
    y_pred = predictions.output.mean(dim=0).cpu().numpy().flatten()

    min_length = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:min_length], y_pred[:min_length]

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print(f"\nResults for {dataset_name}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")

    pd.DataFrame({"actual": y_true, "predicted": y_pred}).to_csv(
        f"{dataset_name}_results.csv", index=False
    )

    plt.figure(figsize=(12,6))
    plt.plot(y_true[:200], label="Actual")
    plt.plot(y_pred[:200], "--", label="Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_plot.png")
    plt.show()

    torch.save(model.state_dict(), f"{dataset_name}_model.pth")

    return mse, mae

def main():
    print(f"Running in {DATASET_MODE} mode...")
    
    # Get data
    df, name = get_data(DATASET_MODE)
    print(f"Data shape: {df.shape}")
    print(f"Target range: [{df['target'].min():.2f}, {df['target'].max():.2f}]")
    print(f"Target mean: {df['target'].mean():.2f} +/- {df['target'].std():.2f}")  
    
    # Create datasets
    training, validation = create_datasets(df)
    print(f"Training samples: {len(training)}")
    print(f"Validation samples: {len(validation)}")

    print("Training model...")
    model = train_new_model(training, validation, max_epochs=MAX_EPOCHS)
    
    # Evaluate and save results
    evaluate_and_save(model, validation, name)

if __name__ == "__main__":
    main()