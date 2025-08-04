#!/bin/bash

python -m TS_MTL.cli \
    federated=true \
    trainer.name=secured_scaffold \
    trainer.params.learning_rate=0.001 \
    trainer.params.personalization_lr=0.0001 \
    trainer.params.local_epochs=5 \
    trainer.params.personalization_epochs=3 \
    trainer.params.noise_scale=0.1 \
    trainer.params.clip_norm=1.0 \
    trainer.params.encoder_noise_scale=0.05 \
    trainer.params.enable_secure_agg=true \
    trainer.params.device="cpu" \
    model.params.hidden_dim=64 \
    model.params.num_layers=2 \
    model.params.dropout=0.2 \
    data.base_path="src/TS_MTL/data/air_quality" \
    data.sites='[station-1,station-2,station-3,station-4,station-5,station-6]' \
    data.hf_suffix="-hf.csv" \
    data.lf_suffix="-lf.csv" \
    data.features='[PM2.5,NO2,PM10]' \
    data.target='[CO]' \
    data.min_date="2014-09-01" \
    data.max_date="2014-11-12" \
    data.batch_size=16 \
    data.hf_lookback=128 \
    data.lf_lookback=32 \
    data.forecast_horizon=16 \
    data.freq_ratio=4 \
    data.train_ratio=0.8 \
    train.epochs=5 \
    hydra.run.dir=.
