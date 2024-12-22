import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# 禁用 Triton 並使用 Eager 模式
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# 設置高精度計算
torch.set_float32_matmul_precision('high')

# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
setup_seed(42)

# --------------------------------------------------
# 1. 讀取原始資料 & 前處理
# --------------------------------------------------
file_path = r"C:\Users\USER\Desktop\gpu_trial\fraud_neighbors2.csv"
original_data = pd.read_csv(file_path)

# 只選擇指定欄位
training_data = original_data[['step', 'age', 'gender', 'merchant', 'category', 'amount', 'fraud']].copy()

# 初始化 MinMaxScaler 並進行正規化
scaler = MinMaxScaler()
training_data['amount'] = scaler.fit_transform(training_data[['amount']].values.astype(np.float32))

# --------------------------------------------------
# 2. 建立 PyTorch Dataset & DataLoader
# --------------------------------------------------
class GANDataSet(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.data = df.values.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_dataloader(batch_size):
    train_dataset = GANDataSet(training_data)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=False)

# --------------------------------------------------
# 3. 定義 Generator / Critic (原 Discriminator)
# --------------------------------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim=256):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 7),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)

# --------------------------------------------------
# 4. Gradient Penalty 函式
# --------------------------------------------------
def gradient_penalty(critic, real_samples, fake_samples, device, lambda_gp=5.0):
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, device=device).expand_as(real_samples)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)

    with torch.amp.autocast(device_type='cuda'):
        critic_out = critic(interpolates)

    grads = torch.autograd.grad(
        outputs=critic_out,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_out),
        create_graph=True,
        retain_graph=True
    )[0]
    grads = grads.view(batch_size, -1)
    grad_norm = torch.sqrt(torch.sum(grads ** 2, dim=1) + 1e-12)
    gp = lambda_gp * ((grad_norm - 1.0) ** 2).mean()
    return gp

# --------------------------------------------------
# 5. 訓練 WGAN-GP
# --------------------------------------------------
def train_wgan_gp(generator, critic, device, grad_scaler, epochs=10000, batch_size=512, n_critic=5):
    train_loader = get_dataloader(batch_size=batch_size)
    data_iter = iter(train_loader)

    for epoch in range(epochs):
        for _ in range(n_critic):
            try:
                real_samples = next(data_iter).to(device)
            except StopIteration:
                data_iter = iter(train_loader)
                real_samples = next(data_iter).to(device)

            noise = torch.randn(real_samples.size(0), generator.latent_dim, device=device)
            fake_samples = generator(noise).detach()

            c_optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                gp = gradient_penalty(critic, real_samples, fake_samples, device)
                c_loss = -(critic(real_samples).mean() - critic(fake_samples).mean()) + gp

            grad_scaler.scale(c_loss).backward()
            grad_scaler.step(c_optimizer)
            grad_scaler.update()

        g_optimizer.zero_grad()
        noise = torch.randn(batch_size, generator.latent_dim, device=device)
        with torch.amp.autocast(device_type='cuda'):
            g_loss = -critic(generator(noise)).mean()

        grad_scaler.scale(g_loss).backward()
        grad_scaler.step(g_optimizer)
        grad_scaler.update()

        if epoch % 500 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}/{epochs} | Critic Loss: {c_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

# --------------------------------------------------
# 6. 生成合成資料並保存
# --------------------------------------------------
def generate_synthetic_data(generator, scaler, num_samples=1000):
    generator.eval()
    synthetic_data = []
    with torch.no_grad():
        for _ in range(num_samples // 256 + 1):
            noise = torch.randn(256, generator.latent_dim, device=device)
            gen_samples = generator(noise).cpu().numpy()
            synthetic_data.append(gen_samples)

    synthetic_data = np.vstack(synthetic_data)[:num_samples]
    
    # 反正規化 amount 列
    synthetic_data[:, 5] = scaler.inverse_transform(synthetic_data[:, 5:6]).flatten()
    return pd.DataFrame(synthetic_data, columns=['step', 'age', 'gender', 'merchant', 'category', 'amount', 'fraud'])

# 主程式入口
if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    # 初始化 Generator 和 Critic 模型
    generator = Generator(latent_dim=256).to(device)
    critic = Critic().to(device)

    # 初始化模型權重
    def weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    generator.apply(weights_init)
    critic.apply(weights_init)

    # 初始化 MinMaxScaler 和 GradScaler
    scaler = MinMaxScaler()  # 用於數據標準化
    grad_scaler = torch.amp.GradScaler()  # 用於混合精度訓練

    # 初始化優化器
    c_optimizer = optim.Adam(critic.parameters(), lr=0.00005, betas=(0.5, 0.999))
    g_optimizer = optim.Adam(generator.parameters(), lr=0.00005, betas=(0.5, 0.999))

    # 呼叫 WGAN-GP 訓練函式
    train_wgan_gp(generator, critic, device, grad_scaler, epochs=10000, batch_size=512, n_critic=5)

    # 使用 MinMaxScaler 生成合成資料
    synthetic_data = generate_synthetic_data(generator, scaler, num_samples=10000)

    # 儲存合成資料
    output_file_path = r"C:\Users\USER\Desktop\gpu_trial\r_u_helloing_data.csv"
    synthetic_data.to_csv(output_file_path, index=False)
    print(f"WGAN-GP 生成的數據已保存至 {output_file_path}")
    print(synthetic_data.describe())
    print(training_data.describe())
