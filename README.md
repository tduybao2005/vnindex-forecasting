# VN-Index 1-hour forecasting

Predicting the next 1-hour close of the VN-Index with an LSTM, against ARIMA
and Prophet baselines.

The interesting part of this project is not the error figure. It is what the
error figure fails to prove, which is written out in
[Limitations](#limitations) below.

| | |
|---|---|
| Hourly candles | **17,875**, Feb 2012 to Jun 2026 |
| Daily candles | 3,606, Jan 2012 to Jun 2026 |
| Session hours kept | 9, 10, 11, 13, 14 (ICT) |
| Split | 70 / 15 / 15, in time order |

---

## Pipeline

```
01_scrape      vnstock, daily VN-Index                 → data/VNINDEX_daily_data.csv
02_build_1h    Kaggle 2012–2024 + vnstock 2024–2026    → data/VNINDEX_1h_data.csv
03_preprocess  UTC → ICT, session hours, dedupe        → data/VNINDEX_1h_clean.csv
04_baselines   ARIMA + Prophet, daily                  → results/baseline_*.json
05_lstm        LSTM, hourly                            → results/lstm_*.json
```

Two things in step 3 are worth naming, because both were silent errors rather
than crashes:

**Timezone.** The Kaggle half of the series carries UTC timestamps; the vnstock
half is already in ICT. Concatenated as-is, the same trading hour appears under
two different labels and the series develops a seven-hour seam in mid-2024.
`03_preprocess.py` detects the UTC rows by their hour and shifts them forward.

**Session hours.** HOSE trades five hours a day. After the shift, every row is
reindexed onto 9, 10, 11, 13 and 14 ICT, so a "1-hour candle" always means the
same thing.

---

## Result

Read from [`results/lstm_metrics.json`](results/lstm_metrics.json):

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| LSTM | 20.32 | 28.69 | **1.26 %** |

Configuration: lookback 150 hourly candles (30 trading sessions), features OHLC,
hidden size 128, 2 layers, dropout 0.2, batch 64, Adam at 1e-4, early stopping
with patience 20.

---

## Limitations

**A 1.26 % MAPE is not evidence the model is useful.** The VN-Index moves very
little from one hour to the next, so a model that learns nothing more than "the
next candle is close to this one" already scores a low percentage error. To know
whether this LSTM beat that, two things are needed and neither exists yet:

- a **naive baseline** that predicts the previous close, scored on the same test
  window
- a **directional accuracy** figure, because getting the direction right is what
  a forecast is for, and MAPE does not measure it

**The ARIMA and Prophet baselines do not settle it either.** They run on *daily*
data with an 85/15 split, while the LSTM runs on *hourly* data with 70/15/15.
Errors at different frequencies are not comparable numbers, so putting them in
one table would be misleading. That is why only the LSTM appears above.

**Gaps are forward-filled.** Missing candles take the previous value, so a small
number of rows in the hourly series are repeats rather than observations.

**No fixed seed.** Results move between runs; see
[docs/reproduce.md](docs/reproduce.md).

---

## Repository map

| Path | |
|---|---|
| [`src/`](src) | The five pipeline stages, numbered in run order |
| [`data/`](data) | Cleaned hourly and daily series |
| [`results/`](results) | Saved LSTM metrics and predictions |
| [`docs/reproduce.md`](docs/reproduce.md) | How to run it again |
| [`docs/dataset.md`](docs/dataset.md) | Notes on the ten-year dataset |
