import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

class FraudDetectionModel:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.model = None
        
    def load_data(self, frac = 1):
    # 讀取資料並保留指定比例
        try:
            self.data = pd.read_csv(self.file_path)
            self.data = self.data.sample(frac=frac, random_state=42)
            print(f"成功讀取 {len(self.data)} 筆資料")
            
            # 檢查 fraud 欄位的唯一值
            unique_values = self.data['fraud'].unique()
            print(f"fraud 欄位唯一值: {unique_values}")

            # 如果包含不符合預期的值，移除或修正
            if not all(value in [0, 1, 2] for value in unique_values):
                print("發現非預期的 fraud 值，修正中...")
                self.data = self.data[self.data['fraud'].isin([0, 1, 2])]
                print(f"修正後剩餘資料筆數: {len(self.data)}")
        except Exception as e:
            print(f"讀取資料錯誤: {e}")


    def train_model(self, test_size=0.3):
        """訓練 XGBoost 模型"""
        if self.data is None:
            print("請先加載資料！")
            return

        X = self.data.drop('fraud', axis=1)
        y = self.data['fraud']

        # 檢查 y 的值是否包含不符合的數值
        unique_y = y.unique()
        print(f"y 的唯一值: {unique_y}")

        # 如果包含 -1 或其他不預期值，修正
        if not all(value in [0, 1, 2] for value in unique_y):
            print("發現非預期的 y 值，修正中...")
            y = y.replace(-1, 2)  # 將 -1 替換為 2，或者其他合法值
            print(f"修正後的 y 唯一值: {y.unique()}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        self.model = XGBClassifier(
            eval_metric='logloss',  
            n_estimators=300, 
            learning_rate=0.03, 
            max_depth=4, 
            scale_pos_weight=5,  # 若資料不平衡，增加此權重
            random_state=42
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        self.evaluate_model(y_test, y_pred, y_pred_proba)


    def evaluate_model(self, y_test, y_pred, y_pred_proba):
        # 評估模型表現並顯示結果
        report = classification_report(y_test, y_pred, digits=4)
        conf_matrix = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print("Classification Report:\n", report)
        print("Confusion Matrix:\n", conf_matrix)
        print(f"ROC-AUC Score: {roc_auc:.4f}")