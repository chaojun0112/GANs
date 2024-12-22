"""import pandas as pd

# 讀取兩個 CSV 檔案
encoded_data_df = pd.read_csv("encoded_data.csv")
fraudulent_transactions_df = pd.read_csv("data10000.csv")

# 合併資料
combined_df = pd.concat([encoded_data_df, fraudulent_transactions_df], ignore_index=True)

# 打散並重新排序
shuffled_combined_df = combined_df.sample(frac=1).reset_index(drop=True)

# 輸出檔案路徑
output_file_path = "shuffled_data.csv"

# 匯出成 CSV
shuffled_combined_df.to_csv(output_file_path, index=False)
print(f"資料已成功儲存於：{output_file_path}")"""



import pandas as pd

# 讀取兩個 CSV 檔案
encoded_data_df = pd.read_csv("encoded_data.csv")
new_data_df = pd.read_csv("helloing.csv")

# 根據 step 插入資料
combined_df = pd.concat([encoded_data_df, new_data_df], ignore_index=True)

# 根據 step 排序
sorted_combined_df = combined_df.sort_values(by="step").reset_index(drop=True)

# 輸出檔案路徑
output_file_path = "shuffled_data.csv"

# 匯出成 CSV
sorted_combined_df.to_csv(output_file_path, index=False)
print(f"資料已成功儲存於：{output_file_path}")