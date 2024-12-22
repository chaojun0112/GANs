import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
import tensorflow as tf

# 讀取數據
file_path = "fraud_neighbors2.csv"
original_data = pd.read_csv(file_path)
training_data = original_data[['step', 'age', 'gender', 'merchant', 'category', 'amount', 'fraud']]

# 標準化 amount 列
scaler = MinMaxScaler()
training_data.loc[:, 'amount'] = scaler.fit_transform(training_data[['amount']])

# 定義模型參數
latent_dim = 7

# Generator
def build_generator():
    model = Sequential([
        Dense(512, input_dim=latent_dim, kernel_initializer='he_normal'),
        LeakyReLU(alpha=0.2),
        Dense(256, kernel_initializer='he_normal'),
        LeakyReLU(alpha=0.2),
        Dense(128, kernel_initializer='he_normal'),
        LeakyReLU(alpha=0.2),
        Dense(7, activation='tanh')
    ])
    return model

# Discriminator
def build_discriminator():
    model = Sequential([
        Dense(128, input_dim=7, kernel_initializer='he_normal'),
        LeakyReLU(alpha=0.2),
        Dense(256, kernel_initializer='he_normal'),
        LeakyReLU(alpha=0.2),
        Dense(128, kernel_initializer='he_normal'),
        LeakyReLU(alpha=0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 建立和編譯模型
generator = build_generator()
discriminator = build_discriminator()

# 初始化模型
noise = tf.random.normal([1, latent_dim])
_ = generator(noise)
_ = discriminator(generator(noise))

def train_gan(epochs=5000, batch_size=256):
    generator_optimizer = tf.keras.optimizers.AdamW(
        learning_rate=0.0001,
        weight_decay=0.00001,
        beta_1=0.5,
        beta_2=0.999
    )
    discriminator_optimizer = tf.keras.optimizers.AdamW(
        learning_rate=0.00005,
        weight_decay=0.00001,
        beta_1=0.5,
        beta_2=0.999
    )

    training_data_tf = tf.cast(training_data.values, tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices(training_data_tf)
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    @tf.function
    def train_step(real_samples):
        batch_size = tf.shape(real_samples)[0]
        with tf.GradientTape(persistent=True) as tape:
            noise = tf.random.normal([batch_size, latent_dim])
            generated_samples = generator(noise, training=True)
            real_output = discriminator(real_samples, training=True)
            fake_output = discriminator(generated_samples, training=True)

            real_labels = tf.cast(tf.random.uniform([batch_size, 1], 0.8, 1.0), tf.float32)
            fake_labels = tf.cast(tf.random.uniform([batch_size, 1], 0.0, 0.2), tf.float32)

            d_loss_real = tf.reduce_mean(tf.keras.losses.binary_crossentropy(real_labels, real_output))
            d_loss_fake = tf.reduce_mean(tf.keras.losses.binary_crossentropy(fake_labels, fake_output))
            d_loss = d_loss_real + d_loss_fake
            g_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output))

        d_gradients = tape.gradient(d_loss, discriminator.trainable_variables)
        g_gradients = tape.gradient(g_loss, generator.trainable_variables)

        discriminator_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))
        generator_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
        return d_loss, g_loss

    print(f"\n開始訓練 - 總計 {epochs} epochs")
    print(f"使用批次大小: {batch_size}")

    for epoch in range(epochs):
        if epoch % 100 == 0:
            print(f"\nEpoch {epoch}/{epochs}")

        epoch_d_losses = []
        epoch_g_losses = []
        for batch in dataset:
            d_loss, g_loss = train_step(batch)
            epoch_d_losses.append(float(d_loss))
            epoch_g_losses.append(float(g_loss))

        if epoch % 100 == 0:
            avg_d_loss = np.mean(epoch_d_losses)
            avg_g_loss = np.mean(epoch_g_losses)
            print(f"D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")

    print("生成十萬筆資料")
    test_noise = tf.random.normal([200000, latent_dim])
    generated_data = generator(test_noise, training=False)
    generated_df = pd.DataFrame(
        generated_data.numpy(),
        columns=['step', 'age', 'gender', 'merchant', 'category', 'amount', 'fraud']
    )
    generated_df.to_csv('generated_samples.csv', index=False)
    generator.save('generator_model', save_format='tf')
    discriminator.save('discriminator_model', save_format='tf')
    print("模型已保存")

if __name__ == "__main__":
    train_gan(epochs=10000, batch_size=256)
