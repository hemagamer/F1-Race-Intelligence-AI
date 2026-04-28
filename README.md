# 🏎️ F1 2026 Miami GP — Podium Prediction Project

A data science and deep learning project that predicts podium finishers for the **2026 Formula 1 Miami Grand Prix** using real telemetry and lap data collected via the [FastF1](https://theoehrly.github.io/Fast-F1/) API.

> **Part of an ongoing F1 analytics series** — see the companion project:
> [🔗 F1 Abu Dhabi Analysis](https://github.com/hemagamer/f1-abudhabi-analysis)

---

## 📌 Project Overview

This notebook aggregates lap-by-lap race data from **all 2026 F1 races preceding the Miami GP**, engineers driver and team performance features, runs exploratory data analysis, and trains a **Keras binary classifier** to estimate each driver's probability of finishing on the podium.

---

## 🗂️ Notebook Structure

| Step | Description |
|------|-------------|
| **1** | Environment setup — install libraries, enable FastF1 cache |
| **2** | Load the 2026 F1 race schedule |
| **3** | Identify all races before the Miami GP |
| **4** | Collect lap-level data for every pre-Miami race |
| **5** | Clean and filter key columns (lap time, tyre compound, position, etc.) |
| **6–7** | EDA — team and driver average pace |
| **8** | Save dataset to CSV |
| **9–17** | Extended EDA — tyre degradation, consistency, race-by-race evolution, correlation matrix, radar chart |
| **18** | Pre-Miami Team Strength Index (weighted pace + position score) |
| **19–21** | Feature engineering — rolling averages, lag features, podium labels |
| **22** | Deep learning model — binary classification with Early Stopping |
| **23–24** | Miami GP podium probability prediction + visualization |

---

## 🧠 Model Architecture

```
Input: [PrevFinish, PrevPace, RecentFinish3, TyreLife, TeamEnc]
         ↓
Dense(32, relu) + L2 Regularization
Dropout(0.30)
         ↓
Dense(16, relu) + L2 Regularization
Dropout(0.20)
         ↓
Dense(1, sigmoid) → Podium Probability
```

- **Loss:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Callbacks:** EarlyStopping (patience=8, restore best weights)  
- **Train/Test split:** 80/20, stratified when class balance allows  

---

## 📊 Features Used

| Feature | Description |
|---------|-------------|
| `PrevFinish` | Driver's average finishing position in the previous race |
| `PrevPace` | Driver's average lap time in the previous race |
| `RecentFinish3` | Rolling 3-race average finishing position |
| `TyreLife` | Average tyre life across stints |
| `TeamEnc` | Label-encoded team identifier |

---

## 📈 EDA Highlights

- **Lap time distribution** — filters out pit laps and safety car anomalies (< 200 sec threshold)
- **Team pace heatmap** — race-by-race breakdown per constructor
- **Tyre degradation curves** — SOFT / MEDIUM / HARD compound comparison
- **Driver momentum plots** — pace trend leading into Miami for top 8 drivers
- **Radar chart** — multi-metric comparison for Mercedes, Ferrari, Red Bull, McLaren
- **Pre-Miami Strength Index** — composite score (70% pace, 30% position)

---

## 🔧 Tech Stack

- **Data:** [FastF1](https://theoehrly.github.io/Fast-F1/) — official F1 telemetry API
- **Processing:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`, `plotly`
- **Modeling:** `TensorFlow / Keras`, `scikit-learn`
- **Environment:** Google Colab

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/hemagamer/f1-miami-2026-prediction.git
cd f1-miami-2026-prediction

# Install dependencies
pip install fastf1 pandas numpy matplotlib seaborn plotly scikit-learn tensorflow
```

Open `F1_2026_Miami_GP_Prediction_Project.ipynb` in Google Colab or Jupyter and run cells sequentially.

> **Note:** FastF1 caches session data on first load — subsequent runs are significantly faster.

---

## 🔗 Related Projects

- [F1 Abu Dhabi Analysis](https://github.com/hemagamer/f1-abudhabi-analysis) — Historical race analysis and lap time breakdown for the Abu Dhabi GP

---

## 👤 Author

**Ibrahim Khalil**  
AI/ML & Data Analytics  
📍 Cairo, Egypt
