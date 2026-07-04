# Auction Bot Detection (FNN)

A feedforward neural network that detects automated bidding bots in online auctions from raw bid-level data. Built for WID3011 Deep Learning at Universiti Malaya.

**Validation AUC-ROC: 0.904** · **Best threshold: 0.70 (F1: 0.455)**

---

## Problem

Online auction platforms are increasingly compromised by automated bots that place rapid, precise bids beyond human capability, undermining fair competition and degrading user trust. Detecting them is hard because:

- **Severe class imbalance** — bots make up only ~5% of bidder accounts
- **Behavioral mimicry** — sophisticated bots resemble human bidding patterns
- **Asymmetric costs** — false positives damage user trust; false negatives let bots persist

This project frames bot detection as a binary classification task using engineered behavioral features and a deep feedforward network.

## Approach

Raw bid-level records (`bids.csv`) are transformed into per-bidder profiles across five feature categories:

| Category | What it captures |
|---|---|
| **Activity** | Bid volume and auction spread (`bids_count`, `auction_count`, `mean_bids_per_auction`) |
| **Time** | Rhythm and consistency of bidding intervals (`tdiff_*`, `tdiff_ip`) |
| **Diversity** | Spread across countries, IPs, devices, URLs (`ip_entropy`, `url_entropy`, `device_cnt`) |
| **Price** | Bid position within auctions (`price_max/min/mean/median/std`) |
| **Response** | Reaction speed to competing bids (`response_*`) |

Bidders with no bidding history are treated as humans and filled with zero-valued features.

## Model

A 5-layer feedforward neural network (4 hidden + 1 output), progressively compressing from 256 → 128 → 64 → 32 neurons:

- **GELU** activations for smoother gradient flow
- **Batch normalization** after every dense layer for training stability
- **Dropout** (0.4 for first 3 layers, 0.3 for the 4th) to prevent co-adaptation
- **Sigmoid** output for probability-based threshold tuning

Trained with Adam (lr=0.001), binary cross-entropy loss, class weighting to offset the 5% bot imbalance, and early stopping / ReduceLROnPlateau callbacks.

## Results

| Metric | Score |
|---|---|
| Validation AUC-ROC | 0.904 |
| AUPR (vs random ≈0.052) | 0.403 |
| Best threshold | 0.70 |
| Precision @ threshold | 0.33 |
| Recall @ threshold | 0.71 |
| F1 | 0.455 |

**Confusion matrix** (403 validation samples):

|  | Predicted Human | Predicted Bot |
|---|---|---|
| **Actual Human** (382) | 352 | 30 |
| **Actual Bot** (21) | 6 | 15 |

At this threshold, the model catches 71% of bots while flagging roughly 1 in 3 accounts correctly — suited as a **screening tool for manual fraud review**, not automatic banning.

## Limitations & Future Work

- **Small dataset**: ~2,000 training samples, only ~103 confirmed bots limits how much bot diversity the model can learn.
- **No temporal modeling**: features are static per-bidder; an **LSTM** could capture bidding rhythm changes over time.
- **No classical ML baseline**: XGBoost/Random Forest may perform competitively on this tabular, small-data setting and are worth benchmarking.
- More labeled bot data from the platform would be the single highest-impact improvement.

## Repo Structure

```
├── src/
│   ├── preprocess.py      # Feature engineering pipeline
│   └── train.py            # Model training script
├── model/
│   ├── FNN_09039.keras     # Trained model
│   ├── scaler_09039.pkl    # StandardScaler fit on training data
│   └── threshold_09039.pkl # Optimal decision threshold
├── data/
│   ├── train_features.csv
│   └── test_features.csv
└── requirements.txt
```

## Usage

```bash
pip install -r requirements.txt

# Generate features from raw bid data
python src/preprocess.py

# Train the model
python src/train.py
```

## Tech Stack

Python · TensorFlow / Keras · scikit-learn · pandas · scipy

## Team

Group project for WID3011 Deep Learning, Faculty of Computer Science & Information Technology, Universiti Malaya.

- Dennis Aimin Oon bin Jeffrey Oon (Me)
- Muhammad Imran bin Ilias
- Muhammad Imran bin Shuhanizal
- Muhammad Aiman bin Sharuddin
- Ahmad Syakir Izzuan bin Hashim
