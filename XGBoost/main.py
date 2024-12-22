from XGBoost_model import FraudDetectionModel

def main():
    # 初始化並讀取資料
    encoded_file_path = 'encoded_data.csv'
    shuffled_file_path = 'shuffled_data.csv'

    print("讀取 encoded_data.csv 並訓練模型")
    encoded_model = FraudDetectionModel(encoded_file_path)
    encoded_model.load_data()
    encoded_model.train_model()

    print("讀取 shuffled_data.csv 並訓練模型")
    shuffled_model = FraudDetectionModel(shuffled_file_path)
    shuffled_model.load_data()
    shuffled_model.train_model()

if __name__ == "__main__":
    main()