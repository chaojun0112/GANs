import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import models
from tensorflow.keras.models import load_model

# Load pre-trained models
generator = load_model("generator_model.h5")
discriminator = load_model("discriminator_model.h5")
gan = load_model("gan_model.h5")

# Define GAN data generation function
latent_dim = 5

def generate_gan_data(customer_start, num_customers, min_transactions=30, max_transactions=180, batch_size=1000):
    synthetic_data = []
    customer_id = customer_start

    for current_customer in range(num_customers):
        num_transactions = np.random.randint(min_transactions, max_transactions + 1)
        noise_fixed = np.random.normal(0, 1, (1, latent_dim))
        generated_fixed_sample = generator.predict(noise_fixed)[0]
        age = int(generated_fixed_sample[1] * 7)
        gender = int(generated_fixed_sample[2] * 3)

        noise_steps = np.random.normal(0, 1, (num_transactions, latent_dim))
        generated_steps = generator.predict(noise_steps)

        for step, step_sample in enumerate(generated_steps):
            merchant = int(step_sample[3] * 49)
            category = int(step_sample[4] * 18)
            fraud = int(step_sample[6] * 1)
            amount = max(0, step_sample[5] * 10000)  # Ensure amount is positive

            synthetic_data.append([step, customer_id, age, gender, 0, merchant, 0, category, amount, fraud])
        
        print(f"Generated {num_transactions} transactions for customer {customer_id} ({current_customer + 1}/{num_customers})")
        customer_id += 1

        if (current_customer + 1) % batch_size == 0 or current_customer == num_customers - 1:
            batch_file_path = f"generated_data_batch_{current_customer + 1}.csv"
            pd.DataFrame(synthetic_data, columns=[
                'step', 'customer_id', 'age', 'gender', 'region', 'merchant', 'transaction_type', 'category', 'amount', 'fraud'
            ]).to_csv(batch_file_path, index=False)
            print(f"Saved batch data to {batch_file_path}")
            synthetic_data = []

# Generate data using pre-trained models
customer_start = 4112
num_customers = 2000
batch_size = 500
generate_gan_data(customer_start=customer_start, num_customers=num_customers, batch_size=batch_size)
