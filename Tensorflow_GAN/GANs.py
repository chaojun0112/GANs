import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.optimizers.legacy import Adam

# Filter relevant columns for GAN training
file_path = r"C:\Users\USER\Desktop\GAN ni nian\fraud_neighbors2.csv"
original_data = pd.read_csv(file_path)
training_data = original_data[['step', 'age', 'gender', 'merchant', 'category', 'amount', 'fraud']].copy()

# Normalize amount column
scaler = MinMaxScaler()
training_data.loc[:, 'amount'] = scaler.fit_transform(training_data[['amount']].values.astype(np.float32))

def load_data(file_path):
    dataset = pd.read_csv(file_path)
    features = dataset[['step', 'age', 'gender', 'merchant', 'category', 'amount', 'fraud']]
    return tf.data.Dataset.from_tensor_slices(features.values).batch(512).prefetch(tf.data.AUTOTUNE)

# 使用 TensorFlow 管道加載資料
training_dataset = load_data(file_path)

# Define GAN model
latent_dim = 5

# Generator
generator = Sequential([
    Input(shape=(latent_dim,)),
    Dense(256),
    LeakyReLU(alpha=0.2),
    Dense(512),
    LeakyReLU(alpha=0.2),
    Dense(1024),
    LeakyReLU(alpha=0.2),
    Dense(2048),
    LeakyReLU(alpha=0.2),
    Dense(7, activation='tanh')
])

# Discriminator
discriminator = Sequential([
    Input(shape=(7,)),
    Dense(2048),
    LeakyReLU(alpha=0.2),
    Dense(1024),
    LeakyReLU(alpha=0.2),
    Dense(512),
    LeakyReLU(alpha=0.2),
    Dense(256),
    LeakyReLU(alpha=0.2),
    Dense(1, activation='sigmoid')
])

# 定義新的優化器
opt = Adam(learning_rate=0.0002, beta_1=0.5)

discriminator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

discriminator.trainable = False
gan = Sequential([generator, discriminator])
gan.compile(loss='binary_crossentropy', optimizer=opt)

# Training function
def train_gan(epochs, batch_size):
    for epoch in range(epochs):
        if epoch % 500 == 0:
            new_lr = 0.0002 * (0.5 ** (epoch // 1000))
            gan.optimizer.learning_rate.assign(new_lr)
            print(f"新學習率: {new_lr:.6f}")

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_samples = generator.predict(noise, verbose=0)

        real_samples = next(iter(training_dataset.take(1))).numpy()

        real_labels = np.ones((batch_size, 1)) * 0.9
        fake_labels = np.zeros((batch_size, 1)) + 0.1

        d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
        d_loss_fake = discriminator.train_on_batch(generated_samples, fake_labels)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_labels = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_labels)

        if epoch % 500 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}/{epochs} | D Loss Real: {d_loss_real[0]:.4f}, D Loss Fake: {d_loss_fake[0]:.4f}, G Loss: {g_loss:.4f}")

train_gan(epochs=10000, batch_size=512)

# Generate synthetic data
def map_to_discrete(value, distribution):
    cumulative = distribution.cumsum()
    random_noise = np.random.uniform(0, 0.01)
    for idx, prob in enumerate(cumulative):
        if value + random_noise <= prob:
            return distribution.index[idx]
    return distribution.index[-1]

age_dist = original_data['age'].value_counts(normalize=True).sort_index()
gender_dist = original_data['gender'].value_counts(normalize=True).sort_index()
merchant_dist = original_data['merchant'].value_counts(normalize=True).sort_index()
category_dist = original_data['category'].value_counts(normalize=True).sort_index()
fraud_dist = original_data['fraud'].value_counts(normalize=True).sort_index()
amount_mean = original_data['amount'].mean()
amount_std = original_data['amount'].std()

def generate_gan_data(customer_start, num_customers, min_transactions=30, max_transactions=180):
    synthetic_data = []
    customer_id = customer_start

    for _ in range(num_customers):
        num_transactions = np.random.randint(min_transactions, max_transactions + 1)
        noise_fixed = np.random.normal(0, 1, (1, latent_dim))
        generated_fixed_sample = generator.predict(noise_fixed, verbose=0)[0]
        age = map_to_discrete(generated_fixed_sample[1], age_dist)
        gender = map_to_discrete(generated_fixed_sample[2], gender_dist)

        noise_steps = np.random.normal(0, 1, (num_transactions, latent_dim))
        generated_steps = generator.predict(noise_steps, verbose=0)

        for step, step_sample in enumerate(generated_steps):
            merchant = map_to_discrete(step_sample[3], merchant_dist)
            category = map_to_discrete(step_sample[4], category_dist)
            fraud = map_to_discrete(step_sample[6], fraud_dist)
            amount = scaler.inverse_transform([[step_sample[5]]])[0, 0]
            synthetic_data.append([step, customer_id, age, gender, 0, merchant, 0, category, amount, fraud])

        customer_id += 1

    return pd.DataFrame(synthetic_data, columns=original_data.columns)

customer_start = 4112
num_customers = 2000
synthetic_data = generate_gan_data(customer_start, num_customers)

# Save models and data
generator.save("generator_model.h5")
discriminator.save("discriminator_model.h5")
gan.save("gan_model.h5")

output_file_path = "r_u_helloing_data.csv"
synthetic_data.to_csv(output_file_path, index=False)
print(f"GAN生成的數據已保存至 {output_file_path}")
