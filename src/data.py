import os
import glob
import pandas as pd
import numpy as np

def load_week_data(data_dir: str, week: int = 1):
    """
    Belirtilen haftanın input (pas atılana kadar) ve output (top havadayken) 
    tracking verilerini yükler.
    """
    input_path = os.path.join(data_dir, "train", f"input_2023_w{week:02d}.csv")
    output_path = os.path.join(data_dir, "train", f"output_2023_w{week:02d}.csv")
    
    if not os.path.exists(input_path):
        # Alternatif dizin yapısı kontrolü
        input_path = os.path.join(data_dir, f"input_2023_w{week:02d}.csv")
        output_path = os.path.join(data_dir, f"output_2023_w{week:02d}.csv")
        
    df_in = pd.read_csv(input_path)
    df_out = pd.read_csv(output_path)
    return df_in, df_out

def get_dataset_summary(df_in: pd.DataFrame, df_out: pd.DataFrame) -> dict:
    """Veri setinin temel istatistiklerini döndürür."""
    return {
        "input_rows": len(df_in),
        "output_rows": len(df_out),
        "unique_games": df_in["game_id"].nunique(),
        "unique_plays": df_in.groupby(["game_id", "play_id"]).ngroups,
        "unique_players": df_in["nfl_id"].nunique(),
        "players_to_predict": df_in[df_in["player_to_predict"] == True]["nfl_id"].nunique(),
        "missing_values_in": int(df_in.isnull().sum().sum()),
        "missing_values_out": int(df_out.isnull().sum().sum())
    }

def prepare_baseline_features(df_in: pd.DataFrame, df_out: pd.DataFrame = None, is_train: bool = True) -> pd.DataFrame:
    """
    Her oyuncunun pas atıldığı son andaki (last input frame) kinematik verilerini alır 
    ve tahmin edilecek her hedef kare (future frame) ile eşleştirir.
    """
    # Sadece tahmin edilmesi istenen oyuncuları filtrele
    if "player_to_predict" in df_in.columns:
        df_in_pred = df_in[df_in["player_to_predict"] == True].copy()
    else:
        df_in_pred = df_in.copy()

    # Pasın atıldığı son kareyi (last observed frame) al
    last_frames = (
        df_in_pred.sort_values("frame_id")
        .groupby(["game_id", "play_id", "nfl_id"])
        .last()
        .reset_index()
    )

    # Açıları (derece) radyana çevir ve hız vektörlerini (vx, vy) hesapla (NFL standart koordinatları)
    dir_rad = np.deg2rad(last_frames["dir"].fillna(0))
    last_frames["vx"] = last_frames["s"] * np.sin(dir_rad)
    last_frames["vy"] = last_frames["s"] * np.cos(dir_rad)

    # Topun düşeceği noktaya olan mesafe
    if "ball_land_x" in last_frames.columns and "ball_land_y" in last_frames.columns:
        last_frames["dist_to_ball_land"] = np.sqrt(
            (last_frames["ball_land_x"] - last_frames["x"]) ** 2 +
            (last_frames["ball_land_y"] - last_frames["y"]) ** 2
        )
        last_frames["dx_to_ball"] = last_frames["ball_land_x"] - last_frames["x"]
        last_frames["dy_to_ball"] = last_frames["ball_land_y"] - last_frames["y"]
    else:
        last_frames["dist_to_ball_land"] = 0.0
        last_frames["dx_to_ball"] = 0.0
        last_frames["dy_to_ball"] = 0.0

    # Son karedeki sütunları yeniden adlandır
    rename_cols = {
        "x": "last_x",
        "y": "last_y",
        "s": "last_s",
        "a": "last_a",
        "dir": "last_dir",
        "o": "last_o",
        "frame_id": "last_frame_id"
    }
    last_frames = last_frames.rename(columns=rename_cols)

    keep_cols = [
        "game_id", "play_id", "nfl_id", "last_frame_id",
        "last_x", "last_y", "last_s", "last_a", "last_dir", "last_o",
        "vx", "vy", "dist_to_ball_land", "dx_to_ball", "dy_to_ball"
    ]
    for col in ["ball_land_x", "ball_land_y", "player_position", "player_side"]:
        if col in last_frames.columns:
            keep_cols.append(col)

    last_frames = last_frames[keep_cols]

    if df_out is not None:
        merged = pd.merge(
            df_out,
            last_frames,
            on=["game_id", "play_id", "nfl_id"],
            how="inner"
        )
        # Gelecekteki kaçıncı kare olduğunu hesapla (1 kare = 0.1 saniye)
        merged["frames_ahead"] = merged["frame_id"] - merged["last_frame_id"]
        # Eğer output frame_id 1'den başlıyorsa doğrudan frame_id kullan
        if (merged["frames_ahead"] <= 0).any():
            merged["frames_ahead"] = merged["frame_id"]
        merged["dt"] = merged["frames_ahead"] * 0.1
        return merged

    return last_frames
