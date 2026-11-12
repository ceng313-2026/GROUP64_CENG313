import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def compute_rmse(y_true_x, y_true_y, y_pred_x, y_pred_y) -> float:
    """
    Yarışmanın resmi değerlendirme metriği: 2 boyutlu Öklid mesafesi üzerinden RMSE.
    """
    squared_errors = (y_true_x - y_pred_x) ** 2 + (y_true_y - y_pred_y) ** 2
    return float(np.sqrt(np.mean(squared_errors)))

class StationaryBaseline:
    """
    Baseline 1: Oyuncunun pas atıldığı son karedeki (last_x, last_y) 
    konumunda sabit kaldığını varsayar.
    """
    def predict(self, df: pd.DataFrame):
        return df["last_x"].values, df["last_y"].values

class KinematicBaseline:
    """
    Baseline 2: Fizik tabanlı sabit hız modeli.
    x_pred = last_x + vx * dt * damping
    y_pred = last_y + vy * dt * damping
    """
    def __init__(self, damping: float = 0.85):
        self.damping = damping

    def predict(self, df: pd.DataFrame):
        pred_x = df["last_x"].values + df["vx"].values * df["dt"].values * self.damping
        pred_y = df["last_y"].values + df["vy"].values * df["dt"].values * self.damping
        # Saha sınırları içine kırpma (0-120 yard boy, 0-53.3 yard en)
        pred_x = np.clip(pred_x, 0.0, 120.0)
        pred_y = np.clip(pred_y, 0.0, 53.3)
        return pred_x, pred_y

class RidgeTrajectoryBaseline:
    """
    Baseline 3: Makine Öğrenmesi (Ridge Regression) modeli.
    Son karedeki hız, ivme, geçen süre (dt) ve topun düşeceği noktaya olan 
    mesafeyi kullanarak (dx, dy) yer değiştirmesini tahmin eder.
    """
    def __init__(self, alpha: float = 1.0):
        self.model_x = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        self.model_y = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        self.feature_cols = [
            "last_s", "last_a", "vx", "vy", "dt", 
            "dist_to_ball_land", "dx_to_ball", "dy_to_ball"
        ]

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].copy().fillna(0)
        # Etkileşim terimleri (Hız x Zaman ve Topa Mesafe x Zaman)
        X["vx_dt"] = X["vx"] * X["dt"]
        X["vy_dt"] = X["vy"] * X["dt"]
        X["dx_ball_dt"] = X["dx_to_ball"] * X["dt"]
        X["dy_ball_dt"] = X["dy_to_ball"] * X["dt"]
        return X.values

    def fit(self, train_df: pd.DataFrame):
        X = self._build_features(train_df)
        target_dx = train_df["x"].values - train_df["last_x"].values
        target_dy = train_df["y"].values - train_df["last_y"].values
        self.model_x.fit(X, target_dx)
        self.model_y.fit(X, target_dy)
        return self

    def predict(self, df: pd.DataFrame):
        X = self._build_features(df)
        pred_dx = self.model_x.predict(X)
        pred_dy = self.model_y.predict(X)
        pred_x = np.clip(df["last_x"].values + pred_dx, 0.0, 120.0)
        pred_y = np.clip(df["last_y"].values + pred_dy, 0.0, 53.3)
        return pred_x, pred_y
