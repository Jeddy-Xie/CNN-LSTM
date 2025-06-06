import pandas as pd
import os

# Read the original dataset
original_data = pd.read_csv('data/processed/BTC-USD.csv')

# Take the first 5000 rows
sample_data = original_data.head(14400)

# Save to the same location
# Overwrite the existing file
sample_data.to_csv('data/processed/BTC-USD-sample.csv', index=False)

print(f"Created small dataset with {len(sample_data)} rows")
print("Saved to: data/processed/BTC-USD-sample.csv") 