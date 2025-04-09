import pandas as pd

# Read the original dataset
original_data = pd.read_csv('data/processed/BTC-USD.csv')

# Take the first 5000 rows
sample_data = original_data.head(5000)

# Save to the same location
sample_data.to_csv('data/processed/BTC-USD-sample.csv', index=False)

print(f"Created small dataset with {len(sample_data)} rows")
print("Saved to: data/processed/BTC-USD-sample.csv") 