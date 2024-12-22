import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
import tensorflow as tf
"""from google.colab import files
import os

# 檢查是否在 Colab 環境中
IN_COLAB = 'google.colab' in str(get_ipython())"""

# 在導入套件後立即設定 GPU
print("Tensorflow 版本:", tf.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 啟用記憶體成長
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # 使用 mixed precision 訓練
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("GPU 設定完成，使用 mixed_float16 訓練")
    except RuntimeError as e:
        print(e)

# 檔案上傳功能
if IN_COLAB:
    print("請上傳 CSV 檔案...")
    uploaded = files.upload()
    file_path = next(iter(uploaded))
else:
    file_path = "step10.csv"

# Filter relevant columns for GAN training
original_data = pd.read_csv(file_path)
training_data = original_data[['step', 'age', 'gender', 'merchant', 'category', 'amount', 'fraud']]

# Normalize amount column
scaler = MinMaxScaler()
training_data.loc[:, 'amount'] = scaler.fit_transform(training_data[['amount']])

# Define GAN model
latent_dim = 5

# Generator
generator = Sequential([
    Input(shape=(latent_dim,)),
    Dense(256),
    LeakyReLU(negative_slope=0.2),
    Dense(512),
    LeakyReLU(negative_slope=0.2),
    Dense(1024),
    LeakyReLU(negative_slope=0.2),
    Dense(7, activation='tanh')
])

# 編譯生成器
generator.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0002, weight_decay=0.004)
)

# Discriminator
discriminator = Sequential([
    Input(shape=(7,)),
    Dense(1024),
    LeakyReLU(negative_slope=0.2),
    Dense(512),
    LeakyReLU(negative_slope=0.2),
    Dense(256),
    LeakyReLU(negative_slope=0.2),
    Dense(1, activation='sigmoid')
])

# 編譯鑑別器
discriminator.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0002, weight_decay=0.004),
    metrics=['accuracy']
)

# Combined GAN model
discriminator.trainable = True
gan = Sequential([generator, discriminator])
gan.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0002, weight_decay=0.004)
)

# Training function
def train_gan(epochs, batch_size):
    # 設定適合的批次大小
    batch_size = 256  # 從 512 改回 256
    
    # 創建優化器
    generator_optimizer = tf.keras.optimizers.AdamW(
        learning_rate=0.0004,
        weight_decay=0.004,
        beta_1=0.5
    )
    discriminator_optimizer = tf.keras.optimizers.AdamW(
        learning_rate=0.0004,
        weight_decay=0.004,
        beta_1=0.5
    )
    
    # 優化數據管道
    training_data_normalized = training_data.copy()
    for column in training_data.columns:
        if training_data[column].dtype in ['int64', 'float64']:
            training_data_normalized[column] = (training_data[column] - training_data[column].mean()) / training_data[column].std()
    
    # 使用 float32 而不是 float16
    training_data_tf = tf.cast(training_data_normalized.values, tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices(training_data_tf)
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    @tf.function(jit_compile=True)
    def train_step(real_samples):
        batch_size = tf.shape(real_samples)[0]
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # 生成假樣本
            noise = tf.random.normal([batch_size, latent_dim], dtype=tf.float32)
            generated_samples = generator(noise, training=True)
            
            # 判別器預測
            real_output = discriminator(real_samples, training=True)
            fake_output = discriminator(generated_samples, training=True)
            
            # 計算損失
            d_loss = tf.reduce_mean(
                tf.concat([
                    tf.keras.losses.binary_crossentropy(tf.ones_like(real_output) * 0.9, real_output),
                    tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output) + 0.1, fake_output)
                ], axis=0)
            )
            
            g_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output)
            )
        
        # 計算並應用梯度
        d_gradients = disc_tape.gradient(d_loss, discriminator.trainable_variables)
        g_gradients = gen_tape.gradient(g_loss, generator.trainable_variables)
        
        discriminator_optimizer.apply_gradients(
            zip(d_gradients, discriminator.trainable_variables)
        )
        generator_optimizer.apply_gradients(
            zip(g_gradients, generator.trainable_variables)
        )
        
        return d_loss, g_loss
    
    print(f"\n開始訓練 - 總計 {epochs} epochs")
    print(f"使用批次大小: {batch_size}")
    
    # 訓練循環
    for epoch in range(epochs):
        if epoch % 100 == 0:
            print(f"\nEpoch {epoch}/{epochs}")
            new_lr = 0.0004 * (0.5 ** (epoch // 1000))
            generator_optimizer.learning_rate.assign(new_lr)
            discriminator_optimizer.learning_rate.assign(new_lr)
        
        epoch_d_losses = []
        epoch_g_losses = []
        
        # 訓練一個 epoch
        for batch in dataset:
            d_loss, g_loss = train_step(batch)
            epoch_d_losses.append(float(d_loss))
            epoch_g_losses.append(float(g_loss))
        
        if epoch % 100 == 0:
            avg_d_loss = np.mean(epoch_d_losses)
            avg_g_loss = np.mean(epoch_g_losses)
            print(f"D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")

# Train the GAN model
train_gan(epochs=7500, batch_size=128)

# Calculate statistical properties of the original data
age_dist = original_data['age'].value_counts(normalize=True).sort_index()
gender_dist = original_data['gender'].value_counts(normalize=True).sort_index()
merchant_dist = original_data['merchant'].value_counts(normalize=True).sort_index()
category_dist = original_data['category'].value_counts(normalize=True).sort_index()
fraud_dist = original_data['fraud'].value_counts(normalize=True).sort_index()
amount_mean = original_data['amount'].mean()
amount_std = original_data['amount'].std()


# 映射生成數據到離散值的函數
def map_to_discrete(value, distribution):
    cumulative = distribution.cumsum()
    random_noise = np.random.uniform(0, 0.01)  # 添加隨機噪聲
    for idx, prob in enumerate(cumulative):
        if value + random_noise <= prob:
            return distribution.index[idx]
    return distribution.index[-1]
    # return value

def generate_gan_data(customer_start, num_customers, min_transactions=30, max_transactions=180):
    synthetic_data = []
    customer_id = customer_start

    for _ in range(num_customers):
        # 每位客戶的step範圍
        num_transactions = np.random.randint(min_transactions, max_transactions + 1)
        
        # 定每位客戶的age和gender
        noise_fixed = np.random.normal(0, 1, (1, latent_dim))
        generated_fixed_sample = generator.predict(noise_fixed)[0]
        age = map_to_discrete(generated_fixed_sample[1], age_dist)
        gender = map_to_discrete(generated_fixed_sample[2], gender_dist)

        # 批量生成每個step的數據
        noise_steps = np.random.normal(0, 1, (num_transactions, latent_dim))
        generated_steps = generator.predict(noise_steps)
        generated_steps += np.random.normal(0, 0.2, generated_steps.shape)

        if generated_steps is None or len(generated_steps) == 0:
            raise ValueError("Generator failed to produce output. Check the model configuration and input.")

        # 添加隨機噪聲以提高多樣性
        generated_steps += np.random.normal(0, 0.2, generated_steps.shape)

        for step, step_sample in enumerate(generated_steps):
            merchant = map_to_discrete(step_sample[3], merchant_dist)
            category = map_to_discrete(step_sample[4], category_dist)
            fraud = map_to_discrete(step_sample[6], fraud_dist)
            amount = scaler.inverse_transform([[step_sample[5]]])[0, 0]

            synthetic_data.append([step, customer_id, age, gender, 0, merchant, 0, category, amount, fraud])

        customer_id += 1

    return pd.DataFrame(synthetic_data, columns=original_data.columns)

# 設置生成的客戶數和起始ID
customer_start = 4112
num_customers = 100
synthetic_data = generate_gan_data(customer_start=customer_start, num_customers=num_customers)

# Save the models after training
if IN_COLAB:
    # 在 Colab 中保存模型
    generator.save("generator_model.h5")
    discriminator.save("discriminator_model.h5")
    gan.save("gan_model.h5")
    
    # 下載模型文件
    print("正在下載模型文件...")
    files.download("generator_model.h5")
    files.download("discriminator_model.h5")
    files.download("gan_model.h5")
else:
    generator.save("generator_model.h5")
    discriminator.save("discriminator_model.h5")
    gan.save("gan_model.h5")
print("模型已成功保存！")

# 保存合成數據
output_file_path = "r_u_helloing_data.csv"
synthetic_data.to_csv(output_file_path, index=False)

if IN_COLAB:
    # 在 Colab 中下載生成的數據
    print("正在下載生成的數據...")
    files.download(output_file_path)

print(f"GAN���成的數據已保存至 {output_file_path}")

# 檢查 GPU 是否可用
print("\n=== 系統資訊 ===")
print("Tensorflow 版本:", tf.__version__)
print("可用的設備:")
for device in tf.config.list_physical_devices():
    print(device)

# 檢查是否使用 GPU
if tf.test.gpu_device_name():
    print('使用 GPU:', tf.test.gpu_device_name())
else:
    print("未檢測到 GPU. 使用 CPU 運行.")

# 設定 GPU 記憶體成長
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU 記憶體動態成長已啟用")
    except RuntimeError as e:
        print(e)

# 在訓練前檢查數據
print("訓練數據形狀:", training_data.shape)
print("訓練數據範例:\n", training_data.head())
print("檢查是否有 NaN 值:", training_data.isna().sum().sum())