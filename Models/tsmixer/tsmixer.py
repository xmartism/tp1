import argparse
import glob
import logging
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import json

# Lokálny data_loader (nová verzia prijímajúca cesty k CSV)
from data_loader import TSFDataLoader
import models

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.FATAL)

DATASET_HPARAMS = {
    "ETTm2":          dict(lr=0.001,  n_block=2, dropout=0.9, ff_dim=64),
    "weather":        dict(lr=0.0001, n_block=4, dropout=0.3, ff_dim=32),
    "electricity":    dict(lr=0.0001, n_block=4, dropout=0.7, ff_dim=64),
    "traffic":        dict(lr=0.0001, n_block=8, dropout=0.7, ff_dim=64),
    "SinTwentyWaves": dict(lr=0.001,  n_block=2, dropout=0.3, ff_dim=64),
}

DEFAULT_HPARAMS = dict(lr=0.0001, n_block=2, dropout=0.1, ff_dim=64)

def parse_args():
    parser = argparse.ArgumentParser(
        description="TSMixer – pipeline-compatible wrapper"
    )

    # --- Povinné argumenty z pipeline ---
    parser.add_argument("--train-dataset", required=True, help="Cesta k train CSV")
    parser.add_argument("--val-dataset",   required=True, help="Cesta k val CSV")
    parser.add_argument("--test-dataset",  required=True, help="Cesta k test CSV")
    parser.add_argument("--target",        required=True, help="Názov cieľového stĺpca")
    parser.add_argument("--date",          default="date", help="Názov stĺpca s dátumom")
    parser.add_argument("--horizon",       type=int, required=True, help="Dĺžka predikcie")
    parser.add_argument("--output",        required=True, help="Výstupný súbor (CSV)")

    # --- Voliteľné hyperparametre (override) ---
    parser.add_argument("--dataset-name",  default=None,
                        help="Názov datasetu pre výber hyperparametrov (napr. 'weather')")
    parser.add_argument("--seq-len",       type=int, default=512)
    parser.add_argument("--batch-size",    type=int, default=32)
    parser.add_argument("--train-epochs",  type=int, default=100)
    parser.add_argument("--patience",      type=int, default=5)
    parser.add_argument("--seed",          type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="./checkpoints/")
    parser.add_argument("--norm-type",     default="B", choices=["L", "B"])
    parser.add_argument("--activation",    default="relu", choices=["relu", "gelu"])
    parser.add_argument("--model",         default="tsmixer_rev_in")

    return parser.parse_args()

def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    hparams = DATASET_HPARAMS.get(args.dataset_name, DEFAULT_HPARAMS)
    lr       = hparams["lr"]
    n_block  = hparams["n_block"]
    dropout  = hparams["dropout"]
    ff_dim   = hparams["ff_dim"]

    print(f"[TSMixer] dataset_name={args.dataset_name or 'unknown'} "
          f"| lr={lr} n_block={n_block} dropout={dropout} ff_dim={ff_dim}")

    data_loader = TSFDataLoader(
        train_path=args.train_dataset,
        val_path=args.val_dataset,
        test_path=args.test_dataset,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        pred_len=args.horizon,
        target=args.target,
        date_col=args.date,
    )

    train_data = data_loader.get_train()
    val_data   = data_loader.get_val()
    test_data  = data_loader.get_test(shuffle=False)

    build_model = getattr(models, args.model).build_model
    model = build_model(
        input_shape=(args.seq_len, data_loader.n_feature),
        pred_len=args.horizon,
        norm_type=args.norm_type,
        activation=args.activation,
        dropout=dropout,
        n_block=n_block,
        ff_dim=ff_dim,
        target_slice=data_loader.target_slice,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, "tsmixer_best")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            verbose=1,
        ),
    ]

    t0 = time.time()
    model.fit(
        train_data,
        epochs=args.train_epochs,
        validation_data=val_data,
        callbacks=callbacks,
        verbose=1,
    )
    print(f"[TSMixer] Tréning dokončený za {time.time() - t0:.1f}s")

    model.load_weights(ckpt_path)

    all_preds  = []
    all_actual = []

    for x_batch, y_batch in test_data:
        preds = model.predict_on_batch(x_batch)
        all_preds.append(preds)
        all_actual.append(y_batch.numpy())

    all_preds  = np.concatenate(all_preds,  axis=0)
    all_actual = np.concatenate(all_actual, axis=0)

    preds_flat = all_preds[-1, :, 0].tolist()
    # actual_flat = all_actual[:, 0, 0]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(preds_flat, f)
    # out_df = pd.DataFrame({"actual": actual_flat, "predicted": preds_flat})
    # out_df.to_csv(args.output, index=False)
    print(f"[TSMixer] Predikcie uložené do: {args.output}")

    for f in glob.glob(ckpt_path + "*"):
        os.remove(f)


if __name__ == "__main__":
    main()