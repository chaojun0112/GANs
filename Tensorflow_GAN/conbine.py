"""import pandas as pd

# 讀取兩個 CSV 檔案
encoded_data_df = pd.read_csv("encoded_data.csv")
fraudulent_transactions_df = pd.read_csv("data10000.csv")

# 合併資料
combined_df = pd.concat([encoded_data_df, fraudulent_transactions_df], ignore_index=True)

# 打散並重新排序
shuffled_combined_df = combined_df.sample(frac=1).reset_index(drop=True)

# 輸出檔案路徑
output_file_path = "helloing.csv"

# 匯出成 CSV
shuffled_combined_df.to_csv(output_file_path, index=False)
print(f"資料已成功儲存於：{output_file_path}")"""


import pandas as pd
import os

def merge_csv_batches(output_file, start_batch=1000, end_batch=2000, step=1000, folder="."):
    all_data = []

    for batch_num in range(start_batch, end_batch + step, step):
        file_name = f"generated_data_batch_{batch_num}.csv"
        file_path = os.path.join(folder, file_name)

        if os.path.exists(file_path):
            print(f"讀取檔案: {file_name}")
            data = pd.read_csv(file_path)
            all_data.append(data)
        else:
            print(f"檔案不存在: {file_name}")

    if all_data:
        merged_data = pd.concat(all_data, ignore_index=True)
        merged_data.to_csv(output_file, index=False)
        print(f"合併完成，已保存至 {output_file}")
    else:
        print("沒有找到任何檔案進行合併！")

# 執行合併
merge_csv_batches("helloing.csv")