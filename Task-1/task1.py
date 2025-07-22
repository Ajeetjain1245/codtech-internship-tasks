# etl_pipeline.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 1️⃣ Extract
df = pd.read_csv('data.csv')  # Replace with your dataset
print("Original Data:")
print(df.head())

# 2️⃣ Transform - Cleaning
df.dropna(inplace=True)  # Remove missing rows

# Example: Scale numeric columns
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df[['amount']])

# Example: One-hot encode a categorical column
encoder = OneHotEncoder(sparse=False)
encoded = encoder.fit_transform(df[['category']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['category']))
df = pd.concat([df.drop('category', axis=1), encoded_df], axis=1)

print("Transformed Data:")
print(df.head())

# 3️⃣ Load - Save cleaned data
df.to_csv('cleaned_data.csv', index=False)
print("Data saved as cleaned_data.csv")
