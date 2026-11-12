import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def set_plot_style():
    """Grafikler için standart görsel tema ayarlar."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12

def plot_kinematic_distributions(df_in: pd.DataFrame):
    """Hız (s), İvme (a), Yönelim (o) ve Hareket Yönü (dir) dağılımlarını çizer."""
    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    sns.histplot(df_in["s"].dropna(), bins=40, kde=True, ax=axes[0, 0], color="royalblue")
    axes[0, 0].set_title("Speed (s) Distribution [yards/sec]")
    
    sns.histplot(df_in["a"].dropna(), bins=40, kde=True, ax=axes[0, 1], color="crimson")
    axes[0, 1].set_title("Acceleration (a) Distribution [yards/sec²]")
    
    sns.histplot(df_in["dir"].dropna(), bins=36, kde=False, ax=axes[1, 0], color="seagreen")
    axes[1, 0].set_title("Motion Direction (dir) Distribution [degrees]")
    
    sns.histplot(df_in["o"].dropna(), bins=36, kde=False, ax=axes[1, 1], color="darkorange")
    axes[1, 1].set_title("Player Orientation (o) Distribution [degrees]")
    
    plt.tight_layout()
    plt.show()

def plot_position_speeds(df_in: pd.DataFrame):
    """Oyuncu pozisyonlarına (WR, CB, QB vb.) göre hız dağılımlarını kutu grafiği ile gösterir."""
    if "player_position" not in df_in.columns:
        return
    set_plot_style()
    plt.figure(figsize=(14, 6))
    order = df_in.groupby("player_position")["s"].median().sort_values(ascending=False).index
    sns.boxplot(data=df_in, x="player_position", y="s", order=order, palette="viridis")
    plt.title("Player Speed Distribution by Position")
    plt.xlabel("Player Position")
    plt.ylabel("Speed (yards/sec)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_play_trajectory(df_in: pd.DataFrame, df_out: pd.DataFrame, game_id: int, play_id: int):
    """
    Seçilen bir oyunda (play) oyuncuların pas öncesi (input) ve pas sonrası (output) 
    gerçek yörüngelerini NFL sahası üzerinde görselleştirir.
    """
    play_in = df_in[(df_in["game_id"] == game_id) & (df_in["play_id"] == play_id)]
    play_out = df_out[(df_out["game_id"] == game_id) & (df_out["play_id"] == play_id)]
    
    fig, ax = plt.subplots(figsize=(12, 6.33))
    ax.set_facecolor("#2e7d32")  # Çim saha rengi
    
    # Saha çizgileri (Her 10 yardda bir)
    for x in range(10, 111, 10):
        ax.axvline(x, color="white", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(10, color="white", linewidth=2)
    ax.axvline(110, color="white", linewidth=2)
    
    # Tahmin edilecek oyuncuların yörüngeleri
    target_ids = play_out["nfl_id"].unique()
    for pid in target_ids:
        p_in = play_in[play_in["nfl_id"] == pid].sort_values("frame_id")
        p_out = play_out[play_out["nfl_id"] == pid].sort_values("frame_id")
        
        ax.plot(p_in["x"], p_in["y"], linestyle=":", color="yellow", alpha=0.8, label="Pre-Pass (Input)" if pid == target_ids[0] else "")
        ax.plot(p_out["x"], p_out["y"], linestyle="-", marker="o", markersize=3, color="cyan", label="In-Air (Target Output)" if pid == target_ids[0] else "")
        if not p_in.empty:
            ax.scatter(p_in["x"].iloc[-1], p_in["y"].iloc[-1], color="orange", s=60, zorder=5)

    # Topun düşeceği nokta
    if "ball_land_x" in play_in.columns and not play_in.empty:
        bx = play_in["ball_land_x"].iloc[0]
        by = play_in["ball_land_y"].iloc[0]
        ax.scatter(bx, by, color="red", marker="X", s=150, zorder=6, label="Ball Landing Spot")

    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    ax.set_title(f"NFL Field Trajectory — Game: {game_id}, Play: {play_id}")
    ax.set_xlabel("Field Length (Yards)")
    ax.set_ylabel("Field Width (Yards)")
    ax.legend(loc="upper right", facecolor="white")
    plt.tight_layout()
    plt.show()
