import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import time
from pathlib import Path
from torch.utils.data import DataLoader
from skimage.transform import resize

# ---------------- 1. 環境設定 ----------------
ROOT = Path(r"C:/BatteryLab")
DATA_PATH = ROOT / "data6/CS2_38_SOH.csv"
MODEL1_DIR = ROOT / "model1"
MODEL2_DIR = ROOT / "model2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 8
EPS = 1e-12

# ---------------- 2. 模型定義 ----------------
class Base_CNN_LSTM(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            # model2 有 BatchNorm，model1 沒有，這裡用判斷式處理
            *( [nn.BatchNorm2d(32)] if in_channels == 5 else [] ),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) 
        )
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        features = self.conv_net(x.view(B*T, C, H, W)).view(B, T, 128)
        lstm_out, _ = self.lstm(features)
        return self.head(lstm_out[:, -1, :])

# ---------------- 3. 虛擬數據生成器 (Nsight 分析專用) ----------------
# 為了穩定分析效能，Nsight 通常使用固定 shape 的 Tensor 進行熱身與測量
def get_dummy_input(batch_size, channels, s):
    return torch.randn(batch_size, SEQ_LEN, channels, s, s).to(DEVICE)

# ---------------- 4. 執行分析邏輯 ----------------
def run_analysis(mode):
    print(f"模式：{mode} | 設備：{DEVICE}")
    
    # 根據模式設定參數
    if mode == "model1":
        target_dir = MODEL1_DIR
        channels = 1
        # 排除 16 的倍數
        s_list = [s for s in range(1, 129) if s % 16 != 0]
        prefix = "model_s"
    else:
        target_dir = MODEL2_DIR
        channels = 5
        # 只跑 16 的倍數
        s_list = [s for s in range(16, 129, 16)]
        prefix = "model_s" # 或是你 model2 的檔名格式

    model = Base_CNN_LSTM(in_channels=channels).to(DEVICE)
    model.eval()

    for s in s_list:
        weight_path = target_dir / f"{prefix}{s}.pth"
        if not weight_path.exists():
            continue
            
        print(f"正在分析 {mode} - 尺寸 s={s}...")
        
        # 載入權重 (使用 CPU 映射避免 GPU 記憶體碎片)
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        
        dummy_x = get_dummy_input(1, channels, s)

        # Nsight 追蹤區間：使用 AMP (如果是 model2)
        with torch.no_grad():
            # 熱身
            for _ in range(5):
                model(dummy_x)
            
            # 正式開始 Nsight 標記區間
            if mode == "model2":
                # 在 model2 啟用 autocast 展現 Tensor Core 效能
                with torch.amp.autocast('cuda'):
                    start = time.perf_counter()
                    output = model(dummy_x)
                    torch.cuda.synchronize()
            else:
                start = time.perf_counter()
                output = model(dummy_x)
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            print(f"   Inference Time: {(end-start)*1000:.4f} ms")

# ---------------- 5. 命令列接口 ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["model1", "model2"], help="選擇分析組別")
    args = parser.parse_args()
    
    run_analysis(args.mode)