# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pytorch_forecasting import TimeSeriesDataSet, DeepAR
from pytorch_forecasting.metrics import NormalDistributionLoss
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch import seed_everything
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import os

# Add safe globals for PyTorch 2.6 checkpoint loading
try:
    from torch.serialization import add_safe_globals
    import pytorch_forecasting.data.encoders
    
    # Add pytorch_forecasting classes
    add_safe_globals([pytorch_forecasting.data.encoders.EncoderNormalizer])
    add_safe_globals([pytorch_forecasting.data.encoders.NaNLabelEncoder])
    add_safe_globals([pytorch_forecasting.data.encoders.TorchNormalizer])
    
    # Add numpy classes (required based on your error)
    import numpy._core.multiarray
    add_safe_globals([numpy._core.multiarray.scalar])
    
    print("? Added PyTorch 2.6+ safe globals")
except ImportError as e:
    print(f"Note: PyTorch 2.6+ compatibility not needed: {e}")
# ============================================

warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIG ---
DATASET_MODE = "WEATHER"  # "SINE" or "WEATHER"
WEATHER_FILE = "weather.csv"
LOAD_FROM_CHECKPOINT = True  # Set to True to load existing model, False to train new
CHECKPOINT_PATH = "./checkpoints/best_model.ckpt"  # Path to saved checkpoint
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


def load_model_from_checkpoint(checkpoint_path, training_dataset):
    """Load a trained model from checkpoint file - MUST MATCH ARCHITECTURE"""
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        return None
    
    try:
        # CRITICAL: Create model with EXACT SAME architecture as checkpoint
        # Based on error, checkpoint has: hidden_size=64, rnn_layers=2
        model = DeepAR.from_dataset(
            training_dataset,
            learning_rate=0.0001,  # Learning rate doesn't affect architecture
            hidden_size=64,        # MUST MATCH checkpoint: 64
            rnn_layers=2,          # MUST MATCH checkpoint: 2
            loss=NormalDistributionLoss(),
            dropout=0.1
        )
        
        # Load the checkpoint
        model = DeepAR.load_from_checkpoint(
            checkpoint_path,
            map_location="cpu"
        )
        print("? Model loaded successfully from checkpoint")
        return model
        
    except Exception as e:
        print(f"? Failed to load checkpoint: {e}")
        
        # Try loading just state dict with exact architecture
        print("\nTrying to infer architecture from checkpoint...")
        try:
            # Load checkpoint to inspect
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            
            # Check what architecture was used
            print("Checkpoint keys:", list(checkpoint.keys()))
            
            if 'hyper_parameters' in checkpoint:
                hparams = checkpoint['hyper_parameters']
                print(f"Original model hyperparameters: {hparams}")
                
                # Create model with original hyperparameters
                model = DeepAR.from_dataset(
                    training_dataset,
                    learning_rate=hparams.get('learning_rate', 0.0001),
                    hidden_size=hparams.get('hidden_size', 64),
                    rnn_layers=hparams.get('rnn_layers', 2),
                    loss=NormalDistributionLoss(),
                    dropout=hparams.get('dropout', 0.1)
                )
                
                # Load state dict
                model.load_state_dict(checkpoint['state_dict'])
                model.eval()
                print("? Model loaded with inferred architecture")
                return model
                
        except Exception as e2:
            print(f"? Could not infer architecture: {e2}")
            
        return None

def train_new_model(training, validation, max_epochs=MAX_EPOCHS):
    """Train a new model from scratch"""
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
    
    # Model checkpoint callback - saves the best model
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best_model",
        dirpath="./checkpoints",
        save_last=False,
        verbose=True
    )
    
    # Create trainer with checkpoint callback
    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        gradient_clip_val=0.1,
        callbacks=[early_stop, checkpoint_callback],
        enable_progress_bar=True
    )
    
    # Create data loaders
    train_loader = training.to_dataloader(train=True, batch_size=32)
    val_loader = validation.to_dataloader(train=False, batch_size=32)
    
    # Train the model
    print("Starting training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # Get the best model (try checkpoint first, fall back to trained model)
    best_model = model  # Default to trained model
    
    if checkpoint_callback.best_model_path and os.path.exists(checkpoint_callback.best_model_path):
        try:
            print(f"\nLoading best checkpoint: {checkpoint_callback.best_model_path}")
            best_model = DeepAR.load_from_checkpoint(checkpoint_callback.best_model_path)
            print(f"? Loaded best model with val_loss: {checkpoint_callback.best_model_score:.4f}")
        except:
            print("? Could not load checkpoint, using trained model")
    
    return best_model

def evaluate_and_save(model, validation, dataset_name):
    # Make predictions
    val_loader = validation.to_dataloader(train=False, batch_size=64)
    predictions = model.predict(val_loader, return_y=True)
    
    # Debug: Check the structure of predictions
    print(f"Predictions type: {type(predictions)}")
    print(f"Predictions output shape: {predictions.output.shape}")
    
    # Check if predictions.y is a tuple or tensor
    if isinstance(predictions.y, tuple):
        print(f"Predictions y is a tuple with {len(predictions.y)} elements")
        # Usually y is the first element of the tuple
        y_true = predictions.y[0].cpu().numpy().flatten()
    else:
        print(f"Predictions y shape: {predictions.y.shape}")
        y_true = predictions.y.cpu().numpy().flatten()
    
    # Get predictions
    # predictions.output shape: [1, 24] means 1 prediction with 24 time steps
    y_pred = predictions.output.mean(dim=0).cpu().numpy().flatten()  # Take mean across batch dimension
    
    print(f"y_pred shape: {y_pred.shape}, y_true shape: {y_true.shape}")
    
    # Ensure shapes match
    min_length = min(len(y_true), len(y_pred))
    y_true = y_true[:min_length]
    y_pred = y_pred[:min_length]
    
    # Calculate metrics - ONLY MSE and MAE
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"\n{'='*50}")
    print(f"Results for {dataset_name}:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Samples compared: {len(y_true)}")
    print(f"{'='*50}")
    
    # Save results with only actual and predicted
    results_df = pd.DataFrame({
        "actual": y_true,
        "predicted": y_pred
    })
    results_df.to_csv(f"{dataset_name}_results.csv", index=False)
    
    # Simple plot - just predictions vs actual (first 200 if available)
    plot_length = min(200, len(y_true))
    plt.figure(figsize=(12, 6))
    plt.plot(y_true[:plot_length], label="Actual", alpha=0.8, linewidth=1)
    plt.plot(y_pred[:plot_length], label="Predicted", alpha=0.8, linestyle='--', linewidth=1)
    plt.title(f"DeepAR Forecast: {dataset_name}")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_plot.png", dpi=100, bbox_inches='tight')
    plt.show()
    
    # Save the model
    torch.save(model.state_dict(), f"{dataset_name}_model.pth")
    print(f"Model saved as {dataset_name}_model.pth")
    
    return mse, mae

def main():
    print(f"Running in {DATASET_MODE} mode...")
    print(f"LOAD_FROM_CHECKPOINT: {LOAD_FROM_CHECKPOINT}")
    
    # Create checkpoint directory if needed
    if not LOAD_FROM_CHECKPOINT:
        os.makedirs("./checkpoints", exist_ok=True)
    
    # Get data
    df, name = get_data(DATASET_MODE)
    print(f"Data shape: {df.shape}")
    print(f"Target range: [{df['target'].min():.2f}, {df['target'].max():.2f}]")
    print(f"Target mean: {df['target'].mean():.2f} +/- {df['target'].std():.2f}")  
    
    # Create datasets
    training, validation = create_datasets(df)
    print(f"Training samples: {len(training)}")
    print(f"Validation samples: {len(validation)}")
    
    # Get model - either load from checkpoint or train new
    if LOAD_FROM_CHECKPOINT:
        # Try to load existing model
        model = load_model_from_checkpoint(CHECKPOINT_PATH, training)
        
        if model is None:
            print("Failed to load checkpoint, training new model...")
            model = train_new_model(training, validation, max_epochs=MAX_EPOCHS)
        else:
            print("Using pre-trained model from checkpoint")
    else:
        # Train new model
        print("Training new model...")
        model = train_new_model(training, validation, max_epochs=MAX_EPOCHS)
    
    # Evaluate and save results
    evaluate_and_save(model, validation, name)

if __name__ == "__main__":
    main()