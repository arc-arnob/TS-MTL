#!/bin/bash
python -m TS_MTL.cli \
    federated=true \
    trainer.name=simple_pfedavg \
    trainer.params.learning_rate=0.001 \
    trainer.params.personalization_lr=0.00005 \
    trainer.params.local_epochs=20 \
    trainer.params.personalization_epochs=10 \
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