#!/usr/bin/env bash
# scripts/run_ts_diff_pc_grad.sh

python -m TS_MTL.cli \
  model.name=ts_diff \
  model.params.hidden_dim=64 \
  model.params.num_layers=3 \
  trainer.name=ts_diff_pc_grad_trainer \
  trainer.params.learning_rate=0.001 \
  trainer.params.device="cpu" \
  trainer.params.time_emb_dim=128 \
  diffusion.num_timesteps=1000 \
  diffusion.beta_start=0.0001 \
  diffusion.beta_end=0.02 \
  data.base_path="src/TS_MTL/data/air_quality" \
  data.sites='[station-1,station-2,station-3,station-4,station-5,station-6]' \
  data.hf_suffix="-hf.csv" \
  data.lf_suffix="-lf.csv" \
  data.features='[PM2.5,NO2,PM10]' \
  data.target='[CO]' \
  data.min_date="2014-09-01" \
  data.max_date="2014-11-12" \
  data.hf_lookback=64 \
  data.lf_lookback=32 \
  data.forecast_horizon=1 \
  data.freq_ratio=2 \
  data.batch_size=16 \
  data.train_ratio=0.8 \
  train.epochs=30 \
  train.eval_horizons='[1,3,6,12]' \
  hydra.run.dir=.
