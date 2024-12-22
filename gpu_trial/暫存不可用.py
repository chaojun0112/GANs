
"""
# --------------------------------------------------
# 定義數據分佈並生成合成資料
# --------------------------------------------------
age_dist = original_data['age'].value_counts(normalize=True).sort_index()
gender_dist = original_data['gender'].value_counts(normalize=True).sort_index()
merchant_dist = original_data['merchant'].value_counts(normalize=True).sort_index()
category_dist = original_data['category'].value_counts(normalize=True).sort_index()
fraud_dist = original_data['fraud'].value_counts(normalize=True).sort_index()

# 生成數據
customer_start = 4112
num_customers = 1000

synthetic_data = generate_gan_data(
    generator, 
    customer_start, 
    num_customers, 
    scaler=scaler,
    age_dist=age_dist,
    gender_dist=gender_dist,
    merchant_dist=merchant_dist,
    category_dist=category_dist,
    fraud_dist=fraud_dist
)

# --------------------------------------------------
# 8. 儲存合成資料
# --------------------------------------------------
output_file_path = r"C:\Users\USER\Desktop\gpu_trial\r_u_helloing_data.csv"
synthetic_data.to_csv(output_file_path, index=False)
print(f"WGAN-GP 生成的數據已保存至 {output_file_path}")

print(synthetic_data.describe())
print(training_data.describe())
"""









"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------
# 1. 讀取原始資料 & 前處理
# --------------------------------------------------
file_path = r"C:\Users\USER\Desktop\gpu trial\fraud_neighbors2.csv"
original_data = pd.read_csv(file_path)

# 只選擇指定欄位
training_data = original_data[['step', 'age', 'gender', 'merchant', 'category', 'amount', 'fraud']].copy()


# 將 amount 欄位做 0~1 正規化
scaler = MinMaxScaler()
training_data.loc[:, 'amount'] = scaler.fit_transform(
    training_data[['amount']].values.astype(np.float32)
)

# --------------------------------------------------
# 2. 建立 PyTorch Dataset & DataLoader (修改這裡)
# --------------------------------------------------
class GANDataSet(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.data = df.values.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_dataset = GANDataSet(training_data)

def get_dataloader(batch_size):
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# --------------------------------------------------
# 3. 定義 Generator / Discriminator
# --------------------------------------------------
latent_dim = 1024  # 與原程式一致

class Generator(nn.Module):
    def __init__(self, latent_dim = 1024):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2),

            # ★ 新增層：擴大模型容量 (不移除原有行) ★
            nn.Linear(2048, 4096),
            nn.LeakyReLU(0.2),
            nn.Linear(4096, 2048),
            nn.LeakyReLU(0.2),
            # ----------------------------------

            nn.Linear(2048, 7),
            nn.Tanh()  # 最後輸出 7 維，與原程式一致
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),

            # ★ 新增層：擴大模型容量 (不移除原有行) ★
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            # ----------------------------------

            nn.Linear(256, 1),
            nn.Sigmoid()  # 最後輸出 1 維 (真 / 假)
        )

    def forward(self, x):
        return self.net(x)

# --------------------------------------------------
# 4. 初始化模型、損失函式與優化器
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
lr = 0.0002

# 兩個分開的 Adam：一個給判別器，一個給生成器
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

# --------------------------------------------------
# 5. 定義訓練函式
# --------------------------------------------------
def train_gan(epochs=10000, batch_size=1024):
    # PyTorch 通常的做法：每個 epoch 都走遍 train_loader 的所有 batch
    # （原程式是每 epoch 只取一次資料；若要嚴格對應，也可直接 next(iter(train_loader)))）
    train_loader = get_dataloader(batch_size=batch_size)

    # 為了和原程式架構更接近，這裡示範「每 epoch 只取一次 batch」：
    data_iter = iter(train_loader)  # 建立迭代器

    for epoch in range(epochs):
        # (可選) 動態調整學習率
        if epoch % 500 == 0:
            new_lr = 0.0002 * (0.5 ** (epoch // 1000))
            for param_group in d_optimizer.param_groups:
                param_group['lr'] = new_lr
            for param_group in g_optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"新學習率: {new_lr:.6f}")

        # 生成器產生假樣本
        noise = torch.randn(batch_size, latent_dim, device=device)
        fake_samples = generator(noise)

        # 真實資料 (從 DataLoader 取一個 batch)
        try:
            real_samples = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            real_samples = next(data_iter)

        real_samples = real_samples.to(device)

        # Label: 真實標記=0.9 (label smoothing)，假樣本標記=0.1
        real_labels = torch.full((batch_size, 1), 0.9, device=device)
        fake_labels = torch.full((batch_size, 1), 0.1, device=device)

        # -------------------------
        # 1) 訓練 Discriminator
        # -------------------------
        d_optimizer.zero_grad()
        real_out = discriminator(real_samples)
        d_loss_real = criterion(real_out, real_labels)

        fake_out = discriminator(fake_samples.detach())  # detach 避免更新 G
        d_loss_fake = criterion(fake_out, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()

        # -------------------------
        # 2) 訓練 Generator
        # -------------------------
        g_optimizer.zero_grad()
        valid_labels = torch.ones((batch_size, 1), device=device)
        g_out = discriminator(fake_samples)  # 不要 detach，需傳回梯度
        g_loss = criterion(g_out, valid_labels)
        g_loss.backward()
        g_optimizer.step()

        # -------------------------
        # 日誌顯示
        # -------------------------
        if epoch % 500 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}/{epochs} | "
                  f"D Loss Real: {d_loss_real.item():.4f}, "
                  f"D Loss Fake: {d_loss_fake.item():.4f}, "
                  f"G Loss: {g_loss.item():.4f}")

# ★ 修改這裡：把 batch_size 改大，例如 1024 ★
train_gan(epochs=10000, batch_size=1024)

# --------------------------------------------------
# 6. 生成合成資料 (對應原程式的 generate_gan_data)
# --------------------------------------------------
age_dist = original_data['age'].value_counts(normalize=True).sort_index()
gender_dist = original_data['gender'].value_counts(normalize=True).sort_index()
merchant_dist = original_data['merchant'].value_counts(normalize=True).sort_index()
category_dist = original_data['category'].value_counts(normalize=True).sort_index()
fraud_dist = original_data['fraud'].value_counts(normalize=True).sort_index()
amount_mean = original_data['amount'].mean()
amount_std = original_data['amount'].std()

def map_to_discrete(value, distribution):
    cumulative = distribution.cumsum()
    random_noise = np.random.uniform(0, 0.01)
    for idx, prob in enumerate(cumulative):
        if value + random_noise <= prob:
            return distribution.index[idx]
    return distribution.index[-1]

def generate_gan_data(
    generator, 
    customer_start, 
    num_customers, 
    min_transactions=30, 
    max_transactions=180,
    scaler=None  # ★ 新增參數 (預設 None)
):
    synthetic_data = []
    customer_id = customer_start

    generator.eval()  # 測試模式 (關閉 Dropout / BatchNorm 更新等)
    with torch.no_grad():
        for _ in range(num_customers):
            num_transactions = np.random.randint(min_transactions, max_transactions + 1)

            # 固定的 noise 用於決定該「客戶」的某些特徵 (如 age, gender)
            noise_fixed = torch.randn(1, latent_dim, device=device)
            gen_fixed_sample = generator(noise_fixed).detach().cpu().numpy()[0]

            age = map_to_discrete(gen_fixed_sample[1], age_dist)
            gender = map_to_discrete(gen_fixed_sample[2], gender_dist)

            noise_steps = torch.randn(num_transactions, latent_dim, device=device)
            gen_steps = generator(noise_steps).detach().cpu().numpy()

            for step_idx, step_sample in enumerate(gen_steps):
                merchant = map_to_discrete(step_sample[3], merchant_dist)
                category = map_to_discrete(step_sample[4], category_dist)
                fraud = map_to_discrete(step_sample[6], fraud_dist)
                amount = scaler.inverse_transform([[step_sample[5]]])[0, 0]

                synthetic_data.append([
                    step_idx,       # step
                    customer_id,    # customer
                    age,            # age
                    gender,         # gender
                    0,              # zipcodeOri (原程式裡好像設為0)
                    merchant,       # merchant
                    0,              # zipMerchant (原程式裡好像設為0)
                    category,       # category
                    amount,         # amount
                    fraud           # fraud
                ])

            customer_id += 1

    return pd.DataFrame(synthetic_data, columns=original_data.columns)

customer_start = 4112
num_customers = 1000

synthetic_data = generate_gan_data(
    generator, 
    customer_start, 
    num_customers, 
    scaler=scaler  # ★ 將外面定義好的 scaler 傳入
)

# --------------------------------------------------
# 7. 儲存模型 & 輸出合成資料
# --------------------------------------------------
torch.save(generator.state_dict(), "generator_model.pt")
torch.save(discriminator.state_dict(), "discriminator_model.pt")

output_file_path = r"C:\Users\USER\Desktop\gpu trial\r_u_helloing_data.csv"
synthetic_data.to_csv(output_file_path, index=False)
print(f"GAN 生成的數據已保存至 {output_file_path}")
"""