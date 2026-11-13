# NFL Big Data Bowl 2026 - Prediction (CENG 313 Project)

## 1. Team Members
|      Name    | Student ID  | GitHub Username |
|--------------|-------------|-----------------|
| Eray Akpınar | 24118080067 |    @erayakpnr   |

---

## 2. Chosen Kaggle Competition
* **Competition:** [NFL Big Data Bowl 2026 - Prediction](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/)
* **Goal:**  Predict the future (x, y) field coordinates of targeted offensive and defensive NFL players while the football is in the air during pass plays.
* **Evaluation Metric:** Root Mean Squared Error (RMSE) of Euclidean distance between predicted and actual (x, y) coordinates.

---

## 3. Dataset Description
The dataset is provided by NFL Next Gen Stats (10 Hz tracking data, 1 frame = 0.1 seconds) covering the 2023–2024 NFL regular seasons:
* **`input_2023_w[01-18].csv`:** Player tracking frames up to the moment the quarterback releases the pass (`x`, `y`, speed `s`, acceleration `a`, orientation `o`, motion angle `dir`, and `ball_land_x`/`ball_land_y`).
* **`output_2023_w[01-18].csv`:** Target $(x, y)$ trajectories while the pass is in flight for designated players (`player_to_predict == True`).
* **`test_input.csv` & `test.csv`:** Unseen test sequences and target frame templates.

---

## 4. Approach (Assignment 1: EDA & Baselines)
1. **Exploratory Data Analysis (EDA):**
   * Analyzed kinematic distributions (`s`, `a`, `dir`, `o`) and position-based speed profiles (identifying speed gaps between skill positions like `WR`/`CB` and linemen).
   * Built a custom 2D NFL field visualizer (`src/utils.py`) to inspect pre-pass stems, in-air pursuit trajectories, and ball landing coordinates.
2. **Feature Engineering (`src/data.py`):**
   * Extracted last-observed frame states at pass release, decomposed speed and angle into orthogonal velocity vectors ($v_x = s \sin(\theta)$, $v_y = s \cos(\theta)$), and calculated Euclidean distance and directional deltas to the ball landing spot (`ball_land_x`, `ball_land_y`).
3. **Baseline Models (`src/models.py`):**
   * **Baseline 1 (Stationary):** Assumes players remain at their pass-release coordinates $(x_0, y_0)$.
   * **Baseline 2 (Physics Kinematic):** Extrapolates position linearly using velocity vectors with an aerodynamic/deceleration damping factor.
   * **Baseline 3 (Ridge Regression):** A regularized linear model predicting displacement $(\Delta x, \Delta y)$ using kinematic states, time horizon ($\Delta t$), and ball-landing vectors.

---

## 5. Results (Assignment 1)
Evaluated on out-of-sample Week 2 validation plays (trained on Week 1):

| Model | Validation RMSE (Yards) |
|-------|-------------------------|
| Baseline 1: Stationary Hold | ~5.42 |
| Baseline 2: Physics Kinematic Extrapolation | ~2.89 |
| Baseline 3: Ridge Regression (Kinematics + Ball Landing) | ~2.14 |

*(Note: Exact metrics are dynamically generated in `assignment1_eda_baseline.ipynb`)*

---

## 6. Repository Structure 
```text
├── README.md
├── assignment1_eda_baseline.ipynb
├── requirements.txt
├── .gitignore
└── src/
    ├── __init__.py
    ├── data.py      
    ├── models.py    
    └── utils.py     
└── data/
    └── README.md
└── submissions/
    ├── README.md
    └── submission_baseline.csv
