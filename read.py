import pandas as pd
import matplotlib.pyplot as plt

def plot_results(csv_path):
    # 讀取你之前儲存的實驗數據
    df = pd.read_csv(csv_path)
    
    # 確保資料按 s 尺寸排序
    df = df.sort_values('s')
    
    # --- 第一張圖：誤差分析 ---
    plt.figure(figsize=(10, 6))
    plt.plot(df['s'], df['RMSE'], marker='o', label='RMSE', color='blue', linewidth=2)
    plt.plot(df['s'], df['MAE'], marker='s', label='MAE', color='red', linewidth=2)
    
    plt.xlabel('Number of states (s)')
    plt.ylabel('Absolute Error')
    plt.title('Error Analysis vs State Size')
    plt.legend() # 這裡會顯示 RMSE 與 MAE 分別是哪條線
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('error_analysis.png')
    plt.show()

    # --- 第二張圖：執行時間分析 ---
    plt.figure(figsize=(10, 6))
    plt.plot(df['s'], df['Inference_Time'], marker='^', label='Running Time', color='green', linewidth=2)
    
    plt.xlabel('Number of states (s)')
    plt.ylabel('Running Time (ms)')
    plt.title('Execution Performance')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('performance_analysis.png')
    plt.show()

if __name__ == "__main__":
    # 指向你儲存結果的 CSV 路徑
    plot_results('experimental_results.csv')