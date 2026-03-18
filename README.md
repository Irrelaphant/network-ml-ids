# ML-IDS — Machine Learning Intrusion Detection System

A lightweight, ML-powered network intrusion detection system that scores network flow traffic and generates ranked alerts. Built on the CIC-IDS-2017 dataset using a Random Forest classifier. Designed to run on standard hardware or a Raspberry Pi 4.

---

## How it works

Instead of inspecting raw packets, ML-IDS analyzes **flow metadata** — summarized connection statistics like packet rate, duration, TCP flag counts, and port numbers. Each flow gets a maliciousness probability score (0–1), and flows above a configurable threshold trigger alerts.

```
Raw CSVs → build → Processed dataset → train → Model → scan → Alerts CSV
```

---

## Requirements

- Python 3.9+
- ~2GB RAM minimum (4GB recommended for full dataset training)
- Dependencies:

```bash
pip install -r requirements.txt
```

> **Raspberry Pi note:** Tested on Pi 4 (4GB). The `build` and `train` steps take 10–20 minutes on a Pi — run them once and save the model. Inference (`scan`) is fast and runs in real time.

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/Irrelaphant/network-ml-ids.git
cd network-ml-ids
```

### 2. Add your dataset

Place your CIC-IDS-2017 CSV files into `data/raw/`. The filenames should contain the day name (e.g. `Monday`, `Tuesday`, ... `Friday`) — this is used to split train vs. test data.

```
data/
  raw/
    Monday-WorkingHours.pcap_ISCX.csv
    Tuesday-WorkingHours.pcap_ISCX.csv
    Wednesday-WorkingHours.pcap_ISCX.csv
    Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
    Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
    ...
```

> Download CIC-IDS-2017 from: https://www.unb.ca/cic/datasets/ids-2017.html

### 3. Configure once

Open `config.yaml` and set your paths and preferences. The defaults work out of the box if your folder structure matches above:

```yaml
raw_dir: "data/raw"
model_tag: "v1"
threshold: 0.30        # default scan sensitivity
test_day_keyword: "Friday"
```

### 4. Build the dataset

Combines and cleans all raw CSVs into one ML-ready dataset:

```bash
python ids.py build
```

### 5. Train the model

Trains a Random Forest classifier on Mon–Thu traffic, held out Friday for evaluation:

```bash
python ids.py train
```

### 6. Scan traffic

Score a CSV file and generate a ranked alerts report:

```bash
python ids.py scan --input data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

Output is saved to `reports/alerts.csv`, sorted most suspicious to least.

---

## Sensitivity presets

Use `--sensitivity` to control alert volume without touching raw numbers:

```bash
python ids.py scan --input traffic.csv --sensitivity low        # high-confidence only
python ids.py scan --input traffic.csv --sensitivity medium     # balanced
python ids.py scan --input traffic.csv --sensitivity high       # default, catches more
python ids.py scan --input traffic.csv --sensitivity aggressive # maximum catch rate
```

| Preset | Threshold | Best for |
|---|---|---|
| `low` | 0.70 | Production — minimal false positives |
| `medium` | 0.50 | General use — balanced |
| `high` | 0.30 | Investigation — wider net |
| `aggressive` | 0.10 | Triage — catch everything |

You can still pass a raw value if you want precise control:

```bash
python ids.py scan --input traffic.csv --threshold 0.45
```

---

## Check system state

See which files are ready at a glance:

```bash
python ids.py status
```

```
System status
────────────────────────────────────────
  ✓  Raw CSV directory         data\raw (8 CSVs)
  ✓  Processed dataset         data\processed\dataset_v1.csv (1,200,000 rows)
  ✓  Trained model             models\rf_v1.joblib
  ✓  Feature schema            models\features_v1.json
  ✓  Metrics report            reports\metrics_v1.json
```

---

## Example scan output

```
══════════════════════════════════════════════
  ML-IDS Scan Report
══════════════════════════════════════════════
  Input   : Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
  Model   : models\rf_v1.joblib
  Output  : reports\alerts.csv
──────────────────────────────────────────────
  Flows scored  :    225,745
  Alerts fired  :     81,381  (36.05%)
  Threshold     :       0.30
──────────────────────────────────────────────
    HIGH     MED     LOW
       9  81,372  144,364
──────────────────────────────────────────────

  Top 10 results:

  SEV    PROB BAR                        SCORE    Dst Port
  HIGH   [████████████████████] 100.0%   1.0000   80
  HIGH   [████████████████████]  99.9%   0.9992   80
  MED    [█████████████████   ]  82.7%   0.8267   80
```

---

## Project structure

```
network-ml-ids/
├── ids.py                        # Main CLI entrypoint
├── config.yaml                   # User configuration
├── requirements.txt
├── scripts/
│   ├── 01_build_dataset.py       # Cleans and combines raw CSVs
│   ├── 02_train.py               # Trains the Random Forest model
│   └── 03_predict.py             # Scores traffic and generates alerts
├── data/
│   ├── raw/                      # Place CIC-IDS CSV files here
│   └── processed/                # Auto-generated combined dataset
├── models/                       # Saved model artifacts (.joblib, .json)
└── reports/                      # Scan output CSVs and training metrics
```

---

## Limitations & roadmap

- **No source/destination IPs** — CIC-IDS-2017 strips IP addresses. CIC-IDS-2018 includes them and is the planned next dataset upgrade.
- **Static model** — the model is trained offline and doesn't update on new traffic. Online learning is a future goal.
- **pfSense integration** — planned for Checkpoint 3: live traffic ingestion via nfstream/CICFlowMeter pointed at a pfSense firewall export.
- **Dataset scope** — one week of lab traffic. More diverse training data will improve generalization.

---

## Checkpoint progress

- [x] Checkpoint 1 — Dataset pipeline, baseline Random Forest, threshold-based alerting
- [x] Checkpoint 2 — CLI wrapper (`ids.py`), sensitivity presets, polished scan output, quickstart README
- [ ] Checkpoint 3 — CIC-IDS-2018 integration, pfSense live traffic, per-attack-type metrics