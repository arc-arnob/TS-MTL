import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import numpy as np
import torch
import random
import os
import json
from hydra.utils import get_original_cwd

from .models import MODEL_REGISTRY
from .trainers import TRAINER_REGISTRY
from .utils.data_preparation import (
    create_client_datasets_with_id,
    combine_client_datasets,
    create_tsdiff_datasets
)


@hydra.main(config_path="../../configs", config_name="default")
def main(cfg: DictConfig):
    
    # ────────────── Set seeds ──────────────
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # ────────────────────────────────────────

    # Check if this is a TSDiff model or MTL model
    is_tsdiff = cfg.model.name == "ts_diff"
    
    if is_tsdiff:
        return run_tsdiff_pipeline(cfg)
    else:
        return run_mtl_pipeline(cfg)


def run_mtl_pipeline(cfg: DictConfig):
    """Run the Multi-Task Learning pipeline for hard_sharing, etc."""
    # 1) Build file‐pairs for each client (HF and LF)
    base   = cfg.data.base_path
    sites  = cfg.data.sites
    hf_suf = cfg.data.hf_suffix
    lf_suf = cfg.data.lf_suffix

    client_data_pairs = [
        (f"{base}/{site}{hf_suf}", f"{base}/{site}{lf_suf}")
        for site in sites
    ]

    # 2) Compute client IDs
    client_ids = list(range(1, len(sites) + 1))

    # 3) Create datasets (train+test split per client)
    client_datasets = create_client_datasets_with_id(
        client_data_pairs=client_data_pairs,
        features=cfg.data.features,
        target=cfg.data.target,
        client_ids=client_ids,
        min_date=cfg.data.min_date,
        max_date=cfg.data.max_date,
        hf_lookback=cfg.data.hf_lookback,
        lf_lookback=cfg.data.lf_lookback,
        forecast_horizon=cfg.data.forecast_horizon,
        freq_ratio=cfg.data.freq_ratio,
        train_ratio=cfg.data.train_ratio,
    )

    # 4) Global train loader
    train_loader = DataLoader(
        combine_client_datasets(client_datasets, mode="train"),
        batch_size=cfg.data.batch_size,
        shuffle=True
    )

    # 5) Instantiate model & trainer
    ModelCls   = MODEL_REGISTRY[cfg.model.name]
    TrainerCls = TRAINER_REGISTRY[cfg.trainer.name]

    model = ModelCls(
        hf_input_dim=cfg.model.params.hf_input_dim,
        lf_input_dim=cfg.model.params.lf_input_dim,
        lf_output_dim=cfg.model.params.lf_output_dim,
        hidden_dim=cfg.model.params.hidden_dim,
        client_ids=client_ids,
        num_layers=cfg.model.params.num_layers,
        dropout=cfg.model.params.dropout,
    )

    trainer = TrainerCls(
        model,
        learning_rate=cfg.trainer.params.learning_rate,
        device=cfg.trainer.params.device,
    )

    # 6) Train
    print(f"\nTraining for {cfg.train.epochs} epochs…")
    trainer.fit(train_loader, epochs=cfg.train.epochs, verbose=True)

    # 7) Evaluate per client
    print("\nEvaluating per‐client on test splits…")
    client_results = {}
    for cid, client_data in client_datasets.items():
        test_loader = DataLoader(
            client_data['test_dataset'],
            batch_size=cfg.data.batch_size,
            shuffle=False
        )
        metrics = trainer.evaluate(test_loader)
        client_results[cid] = metrics

        print(f"\nClient {cid} metrics:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  MSE:  {metrics['mse']:.4f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")

    # 8) Overall averages
    overall = {
        m: np.mean([res[m] for res in client_results.values()])
        for m in ['loss','mae','mse','mape']
    }
    print("\nOverall metrics (average across clients):")
    print(f"  Loss: {overall['loss']:.4f}")
    print(f"  MAE:  {overall['mae']:.4f}")
    print(f"  MSE:  {overall['mse']:.4f}")
    print(f"  MAPE: {overall['mape']:.2f}%")
    
    return {"overall_metrics": overall, "client_results": client_results}


def run_tsdiff_pipeline(cfg: DictConfig):
    """Run the TSDiff pipeline for diffusion models."""
    # ─── 1) Build HF/LF file‐pairs ─────────────────────────
    repo_root = get_original_cwd()
    base      = os.path.join(repo_root, cfg.data.base_path)
    hf_suf, lf_suf = cfg.data.hf_suffix, cfg.data.lf_suffix

    client_data_pairs = [
        (f"{base}/{site}{hf_suf}", f"{base}/{site}{lf_suf}")
        for site in cfg.data.sites
    ]
    client_ids = list(range(1, len(client_data_pairs) + 1))

    # ─── 2) Create TSDiff train/test splits ───────────────
    train_ds, test_ds, client_info = create_tsdiff_datasets(
        client_data_pairs  = client_data_pairs,
        features            = cfg.data.features,
        target              = cfg.data.target,
        client_ids          = client_ids,
        min_date            = cfg.data.min_date,
        max_date            = cfg.data.max_date,
        hf_lookback         = cfg.data.hf_lookback,
        lf_lookback         = cfg.data.lf_lookback,
        forecast_horizon    = cfg.data.forecast_horizon,
        freq_ratio          = cfg.data.freq_ratio,
        train_ratio         = cfg.data.train_ratio,
        debug               = cfg.get("debug", False),
    )

    train_loader = DataLoader(train_ds,
                              batch_size=cfg.data.batch_size,
                              shuffle=True)
    test_loader  = DataLoader(test_ds,
                              batch_size=cfg.data.batch_size,
                              shuffle=False)

    # ─── 3) Instantiate model & trainer ───────────────────
    # total input dim = HF_interp + LF dims
    first    = next(iter(client_info.values()))
    hf_feats = first["hf_data_shape"][1]
    lf_feats = first["lf_data_shape"][1]
    input_dim = hf_feats + lf_feats

    device = cfg.trainer.params.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU instead")

    # pick TSDiffModel and TSDiffTrainer from registry
    ModelCls   = MODEL_REGISTRY[cfg.model.name]
    TrainerCls = TRAINER_REGISTRY[cfg.trainer.name]

    model = ModelCls(
        input_dim    = input_dim,
        hidden_dim   = cfg.model.params.hidden_dim,
        num_layers   = cfg.model.params.num_layers,
        time_emb_dim = cfg.trainer.params.time_emb_dim
    ).to(device)

    diffusion = MODEL_REGISTRY["diffusion_process"](
        num_timesteps = cfg.diffusion.num_timesteps,
        beta_start    = cfg.diffusion.beta_start,
        beta_end      = cfg.diffusion.beta_end
    ).to(device)

    trainer = TrainerCls(
        model             = model,
        diffusion_process = diffusion,
        learning_rate     = cfg.trainer.params.learning_rate,
        device            = device
    )

    # ─── 4) Train ─────────────────────────────────────────
    print(f"\nTraining TSDiff for {cfg.train.epochs} epochs on {device}…")
    history = trainer.fit(
        train_loader = train_loader,
        val_loader   = test_loader,
        epochs       = cfg.train.epochs,
        verbose      = True
    )

    # ─── 5) Evaluate at multiple horizons ─────────────────
    print("\n=== Forecast Horizon Evaluation ===")
    horizon_results = {}
    for H in cfg.train.eval_horizons:
        direct = trainer.evaluate_multihorizon(test_loader, H, autoregressive=False)
        auto   = trainer.evaluate_multihorizon(test_loader, H, autoregressive=True)
        horizon_results[H] = {"direct": direct, "autoregressive": auto}
        print(f"H={H:2d}  direct MAE={direct['mae']:.4f}, autoreg MAE={auto['mae']:.4f}")

    # ─── 6) Log horizon=1 preds/targets ───────────────────
    res = trainer.predict_multihorizon(
        test_loader,
        forecast_horizon=1,
        guidance_scale=1.0,
        autoregressive=False
    )
    preds = res["predictions"].squeeze(1).tolist()
    targs = res["targets"].squeeze(1).tolist()

    out_path = os.path.join(repo_root, "preds_targets_log.txt")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "a") as f:
        f.write(f"# Model: {cfg.model.name}, Horizon=1\n")
        f.write("predictions: " + json.dumps(preds) + "\n")
        f.write("targets:     " + json.dumps(targs) + "\n\n")
    print(f"\nAppended horizon=1 results to {out_path}")

    # ─── 7) Summary for main horizon ───────────────────────
    main_H = cfg.train.eval_horizons[-1]
    main_m = trainer.evaluate_multihorizon(test_loader, main_H, autoregressive=False)
    print(f"\nMain eval (H={main_H}): MAE={main_m['mae']:.4f}, MSE={main_m['mse']:.4f}, MAPE={main_m['mape']:.2f}%")

    return {
        "train_loss": history["train_loss"][-1],
        "main_eval": main_m,
        "horizon_results": horizon_results
    }


if __name__ == "__main__":
    main()



