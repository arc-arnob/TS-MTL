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
    create_tsdiff_datasets,
    create_client_datasets
)
from .models.federated.per_fed_avg import (
    train_simple_pfedavg_system,
    train_personalized_fedavg_system,
    train_private_fedprox_system,
    train_private_scaffold_system,
)
from .models.federated.per_fed_conf import train_private_federated_system
from .models.non_fed_baselines.arimax_independent import run_arimax_independent
from .models.non_fed_baselines.arimax_global import run_arimax_global


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

    # Check if this is a TSDiff model or MTL model or federated model or ARIMAX
    is_tsdiff = cfg.model.name == "ts_diff"
    is_federated = cfg.get("federated", False)
    is_arimax = cfg.model.name.startswith("arimax")

    if is_tsdiff:
        return run_tsdiff_pipeline(cfg)
    elif is_federated:
        return run_federated_pipeline(cfg)
    elif is_arimax:
        return run_arimax_dispatch(cfg)
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


def run_federated_pipeline(cfg: DictConfig):

    """Standard or personalized FedAvg pipeline (simple vs Per‐FedAvg)."""
    base, hf_suf, lf_suf = cfg.data.base_path, cfg.data.hf_suffix, cfg.data.lf_suffix
    pairs = [(f"{base}/{s}{hf_suf}", f"{base}/{s}{lf_suf}") for s in cfg.data.sites]

    # Use the federated-specific data preparation function
    client_datasets = create_client_datasets(
        client_data_pairs=pairs,
        features=cfg.data.features,
        target=cfg.data.target,
        min_date=cfg.data.min_date,
        max_date=cfg.data.max_date,
        hf_lookback=cfg.data.hf_lookback,
        lf_lookback=cfg.data.lf_lookback,
        forecast_horizon=cfg.data.forecast_horizon,
        freq_ratio=cfg.data.freq_ratio,
        train_ratio=cfg.data.train_ratio,
        debug=cfg.get("debug", False)
    )

    print(f"\nStarting federated learning with {len(client_datasets)} clients...")
    print(f"Trainer: {cfg.trainer.name}")
    print(f"Device: {cfg.trainer.params.device}")

    # pick which training function based on trainer.name
    if cfg.trainer.name == "simple_pfedavg":
        system, history = train_simple_pfedavg_system(
            client_datasets,
            hidden_dim=cfg.model.params.hidden_dim,
            num_layers=cfg.model.params.num_layers,
            dropout=cfg.model.params.dropout,
            batch_size=cfg.data.batch_size,
            learning_rate=cfg.trainer.params.learning_rate,
            personalization_lr=cfg.trainer.params.get("personalization_lr", 1e-5),
            rounds=cfg.train.epochs,
            local_epochs=cfg.trainer.params.get("local_epochs", 1),
            personalization_epochs=cfg.trainer.params.get("personalization_epochs", 10),
            device=cfg.trainer.params.device,
        )
    
    elif cfg.trainer.name == "secured_fedavg":
        system, history = train_private_federated_system(
            client_datasets,
            hidden_dim=cfg.model.params.hidden_dim,
            num_layers=cfg.model.params.num_layers,
            dropout=cfg.model.params.dropout,
            batch_size=cfg.data.batch_size,
            learning_rate=cfg.trainer.params.learning_rate,
            personalization_lr=cfg.trainer.params.get("personalization_lr", 1e-4),
            rounds=cfg.train.epochs,
            local_epochs=cfg.trainer.params.get("local_epochs", 5),
            personalization_epochs=cfg.trainer.params.get("personalization_epochs", 3),
            noise_scale=cfg.trainer.params.get("noise_scale", 0.1),
            clip_norm=cfg.trainer.params.get("clip_norm", 1.0),
            encoder_noise_scale=cfg.trainer.params.get("encoder_noise_scale", 0.05),
            enable_secure_agg=cfg.trainer.params.get("enable_secure_agg", True),
            device=cfg.trainer.params.device,
        )
    
    elif cfg.trainer.name == "secured_fedprox":
        system, history = train_private_fedprox_system(
            client_datasets,
            hidden_dim=cfg.model.params.hidden_dim,
            num_layers=cfg.model.params.num_layers,
            dropout=cfg.model.params.dropout,
            batch_size=cfg.data.batch_size,
            learning_rate=cfg.trainer.params.learning_rate,
            personalization_lr=cfg.trainer.params.get("personalization_lr", 1e-4),
            rounds=cfg.train.epochs,
            local_epochs=cfg.trainer.params.get("local_epochs", 5),
            personalization_epochs=cfg.trainer.params.get("personalization_epochs", 3),
            noise_scale=cfg.trainer.params.get("noise_scale", 0.1),
            clip_norm=cfg.trainer.params.get("clip_norm", 1.0),
            encoder_noise_scale=cfg.trainer.params.get("encoder_noise_scale", 0.05),
            fedprox_mu=cfg.trainer.params.get("fedprox_mu", 0.01),
            enable_secure_agg=cfg.trainer.params.get("enable_secure_agg", True),
            device=cfg.trainer.params.device,
        )
    
    elif cfg.trainer.name == "secured_scaffold":
        system, history = train_private_scaffold_system(
            client_datasets,
            hidden_dim=cfg.model.params.hidden_dim,
            num_layers=cfg.model.params.num_layers,
            dropout=cfg.model.params.dropout,
            batch_size=cfg.data.batch_size,
            learning_rate=cfg.trainer.params.learning_rate,
            personalization_lr=cfg.trainer.params.get("personalization_lr", 1e-4),
            rounds=cfg.train.epochs,
            local_epochs=cfg.trainer.params.get("local_epochs", 5),
            personalization_epochs=cfg.trainer.params.get("personalization_epochs", 3),
            noise_scale=cfg.trainer.params.get("noise_scale", 0.1),
            clip_norm=cfg.trainer.params.get("clip_norm", 1.0),
            encoder_noise_scale=cfg.trainer.params.get("encoder_noise_scale", 0.05),
            enable_secure_agg=cfg.trainer.params.get("enable_secure_agg", True),
            device=cfg.trainer.params.device,
        )
    
    else:  # "personalized_fedavg"
        system, history = train_personalized_fedavg_system(
            client_datasets,
            hidden_dim=cfg.model.params.hidden_dim,
            num_layers=cfg.model.params.num_layers,
            dropout=cfg.model.params.dropout,
            batch_size=cfg.data.batch_size,
            learning_rate=cfg.trainer.params.learning_rate,
            meta_learning_rate=cfg.trainer.params.get("meta_learning_rate", 1e-2),
            epochs=cfg.train.epochs,
            local_epochs=cfg.trainer.params.get("local_epochs", 1),
            personalization_steps=cfg.trainer.params.get("personalization_steps", 10),
            device=cfg.trainer.params.device,
        )

    # Evaluate final global vs personalized
    test_loaders = [
        DataLoader(cd["test_dataset"], batch_size=cfg.data.batch_size)
        for cd in client_datasets.values()
    ]
    final = system.evaluate(test_loaders)
    
    print("\nFinal evaluation per client:")
    for i, m in enumerate(final):
        client_num = i + 1  # Start from 1 instead of 0
        print(
            f"Client {client_num}:\n"
            f"  Global   → Loss: {m['global_loss']:.4f}, MAE: {m['global_mae']:.4f}, MAPE: {m['global_mape']:.2f}%\n"
            f"  Personal → Loss: {m['personalized_loss']:.4f}, MAE: {m['personalized_mae']:.4f}, MAPE: {m['personalized_mape']:.2f}%"
        )
        
    
    # Calculate overall averages
    overall_global = {
        metric: np.mean([m[f'global_{metric}'] for m in final])
        for metric in ['loss', 'mae', 'mse', 'mape']
    }
    overall_personalized = {
        metric: np.mean([m[f'personalized_{metric}'] for m in final])
        for metric in ['loss', 'mae', 'mse', 'mape']
    }
    
    print("Overall metrics (average across all clients):")
    print(f"  Global model:")
    print(f"    Loss: {overall_global['loss']:.4f}, MAE: {overall_global['mae']:.4f}, MAPE: {overall_global['mape']:.2f}%")
    print(f"  Personalized model:")
    print(f"    Loss: {overall_personalized['loss']:.4f}, MAE: {overall_personalized['mae']:.4f}, MAPE: {overall_personalized['mape']:.2f}%")
    
    avg_loss_improvement = (overall_global['loss'] - overall_personalized['loss']) / overall_global['loss'] * 100
    avg_mape_improvement = (overall_global['mape'] - overall_personalized['mape']) / overall_global['mape'] * 100
    print(f"  Average improvement: Loss: {avg_loss_improvement:.2f}%, MAPE: {avg_mape_improvement:.2f}%")
    
    return {
        "system": system, 
        "history": history, 
        "final_metrics": final,
        "overall_global": overall_global,
        "overall_personalized": overall_personalized
    }


def run_arimax_dispatch(cfg: DictConfig):
    """Dispatch to the appropriate ARIMAX pipeline based on model name."""
    if cfg.model.name == "arimax_independent":
        return run_arimax_independent_pipeline(cfg)
    elif cfg.model.name == "arimax_global":
        return run_arimax_global_pipeline(cfg)
    else:
        raise ValueError(f"Unknown ARIMAX model: {cfg.model.name}")


def run_arimax_independent_pipeline(cfg: DictConfig):
    """Run the ARIMAX Independent pipeline."""
    # Extract parameters from config
    data_params = {
        "data_base_path": cfg.data.base_path,
        "sites": cfg.data.sites,
        "hf_suffix": cfg.data.hf_suffix,
        "lf_suffix": cfg.data.lf_suffix,
        "features": cfg.data.features,
        "target": cfg.data.target,
        "min_date": cfg.data.min_date,
        "max_date": cfg.data.max_date,
        "train_ratio": cfg.data.train_ratio,
    }
    
    # ARIMAX-specific parameters (with defaults)
    arimax_params = {
        "lookback_days": cfg.model.params.get("lookback_days", 32),
        "forecast_horizon": cfg.data.get("forecast_horizon", 16),
        "save_plots": cfg.model.params.get("save_plots", False),
    }
    
    # Combine parameters
    all_params = {**data_params, **arimax_params}
    
    print(f"\nRunning ARIMAX Independent model...")
    print(f"Data path: {all_params['data_base_path']}")
    print(f"Sites: {all_params['sites']}")
    print(f"Features: {all_params['features']}")
    print(f"Target: {all_params['target']}")
    print(f"Lookback days: {all_params['lookback_days']}")
    print(f"Forecast horizon: {all_params['forecast_horizon']}")
    
    # Run the ARIMAX independent model
    results = run_arimax_independent(**all_params)
    
    if results is not None:
        overall_results = results.get("overall_metrics", {})
        site_results = results.get("site_results", [])
        
        print(f"\nARIMAX Independent results:")
        print(f"Overall MAE: {overall_results.get('mae', 0.0):.4f}")
        print(f"Overall MSE: {overall_results.get('mse', 0.0):.4f}")
        print(f"Overall MAPE: {overall_results.get('mape', 0.0):.2f}%")
        
        # Print per-site results
        if site_results:
            print("\nPer-site results:")
            for site_result in site_results:
                site = site_result.get('site', 'Unknown')
                mae = site_result.get('mae', 0.0)
                mse = site_result.get('mse', 0.0)
                mape = site_result.get('mape', 0.0)
                print(f"  {site}: MAE={mae:.4f}, MSE={mse:.4f}, MAPE={mape:.2f}%")
    
    return results


def run_arimax_global_pipeline(cfg: DictConfig):
    """Run the ARIMAX Global pipeline."""
    # Extract parameters from config
    data_params = {
        "data_base_path": cfg.data.base_path,
        "sites": cfg.data.sites,
        "hf_suffix": cfg.data.hf_suffix,
        "lf_suffix": cfg.data.lf_suffix,
        "features": cfg.data.features,
        "target": cfg.data.target,
        "min_date": cfg.data.min_date,
        "max_date": cfg.data.max_date,
        "train_ratio": cfg.data.train_ratio,
    }
    
    # ARIMAX-specific parameters (with defaults)
    arimax_params = {
        "lookback_days": cfg.model.params.get("lookback_days", 32),
        "forecast_horizon": cfg.data.get("forecast_horizon", 16),
        "save_plots": cfg.model.params.get("save_plots", False),
    }
    
    # Combine parameters
    all_params = {**data_params, **arimax_params}
    
    print(f"\nRunning ARIMAX Global model...")
    print(f"Data path: {all_params['data_base_path']}")
    print(f"Sites: {all_params['sites']}")
    print(f"Features: {all_params['features']}")
    print(f"Target: {all_params['target']}")
    print(f"Lookback days: {all_params['lookback_days']}")
    print(f"Forecast horizon: {all_params['forecast_horizon']}")
    
    # Run the ARIMAX global model
    results = run_arimax_global(**all_params)
    
    if results is not None:
        overall_results = results.get("overall_metrics", {})
        per_site_mae = results.get("per_site_mae", {})
        
        print(f"\nARIMAX Global results:")
        print(f"Overall MAE: {overall_results.get('mae', 0.0):.4f}")
        print(f"Overall MSE: {overall_results.get('mse', 0.0):.4f}")
        print(f"Overall MAPE: {overall_results.get('mape', 0.0):.2f}%")
        
        # Print per-site MAE
        if per_site_mae:
            print("\nPer-site MAE:")
            for site, mae in per_site_mae.items():
                print(f"  {site}: {mae:.4f}")
    
    return results

if __name__ == "__main__":
    main()



