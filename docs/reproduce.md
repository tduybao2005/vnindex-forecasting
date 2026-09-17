# Running the pipeline again

The scripts are numbered in the order they run and each one writes what the
next one reads, so the whole chain is:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python src/01_scrape.py      # daily VN-Index from vnstock  → data/VNINDEX_daily_data.csv
python src/02_build_1h.py    # Kaggle archive + vnstock     → data/VNINDEX_1h_data.csv
python src/03_preprocess.py  # timezone fix, session hours  → data/VNINDEX_1h_clean.csv
python src/04_baselines.py   # ARIMA + Prophet, daily       → results/baseline_*.json
python src/05_lstm.py        # LSTM, hourly                 → results/lstm_*.json
```

Steps 1 and 2 reach the network. The cleaned data they produce is committed
under `data/`, so steps 4 and 5 run on their own against what is in the
repository.

`02_build_1h.py` additionally needs `data/HOSEVNINDEXH1.csv`, the Kaggle hourly
archive covering 2012 to 2024. It is not redistributed here; the cleaned output
built from it is, in `data/VNINDEX_1h_clean.csv`.

## Expected result

`src/05_lstm.py` should finish with numbers close to those in
`results/lstm_metrics.json`:

```json
{ "LSTM": { "MAE": 20.322, "RMSE": 28.6946, "MAPE": 1.2639 } }
```

No random seed is fixed, so weight initialisation and shuffling of the training
batches move the result between runs. Treat the figures as the scale of the
error rather than as values to reproduce exactly.
