import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import numpy as np
import torch
import random
from .models import MODEL_REGISTRY
from .trainers import TRAINER_REGISTRY
from .utils.data_preparation import (
    create_client_datasets_with_id,
    combine_client_datasets
)


@hydra.main(config_path="../../configs", config_name="default")
def main(cfg: DictConfig):
    
    # ────────────── Set seeds ──────────────
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # If you’re using CUDA:
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # ────────────────────────────────────────

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

if __name__ == "__main__":
    main()
