import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the predictions
previous_best = pd.read_csv('best_prediction.csv')
new_predictions = pd.read_csv('submission_0315.csv')

# Ensure the coordinates match (they might be in different orders)
# Sort by coordinates to ensure a fair comparison
previous_best = previous_best.sort_values(['Latitude', 'Longitude']).reset_index(drop=True)
new_predictions = new_predictions.sort_values(['Latitude', 'Longitude']).reset_index(drop=True)

# Extract UHI Index values
prev_uhi = previous_best['UHI Index'].values
new_uhi = new_predictions['UHI Index'].values

# Calculate metrics
mse = mean_squared_error(prev_uhi, new_uhi)
rmse = np.sqrt(mse)
r2 = r2_score(prev_uhi, new_uhi)

# Calculate percentage difference
pct_diff = np.abs(prev_uhi - new_uhi) / prev_uhi * 100
mean_pct_diff = np.mean(pct_diff)
median_pct_diff = np.median(pct_diff)

# Calculate similarity score (higher is better)
similarity = 100 - mean_pct_diff

print(f"Comparison of Previous Best (86% accuracy) vs New Predictions:")
print(f"MSE: {mse:.8f}")
print(f"RMSE: {rmse:.8f}")
print(f"R² score: {r2:.8f}")
print(f"Mean percentage difference: {mean_pct_diff:.2f}%")
print(f"Median percentage difference: {median_pct_diff:.2f}%")
print(f"Similarity score: {similarity:.2f}%")

# Calculate improvement
if similarity > 86:
    improvement = similarity - 86
    print(f"\nOur new model shows an improvement of {improvement:.2f}% over the previous model")
else:
    print(f"\nOur new model does not show improvement over the previous 86% accuracy")

# Create a dataframe with detailed comparison
comparison = pd.DataFrame({
    'Latitude': previous_best['Latitude'],
    'Longitude': previous_best['Longitude'],
    'Previous_UHI': prev_uhi,
    'New_UHI': new_uhi,
    'Absolute_Diff': np.abs(prev_uhi - new_uhi),
    'Percentage_Diff': pct_diff
})

# Save the comparison to a file
comparison.to_csv('prediction_comparison_0315.csv', index=False)
print(f"Detailed comparison saved to prediction_comparison_0315.csv")

# Create visualization
plt.figure(figsize=(12, 8))

# Scatter plot
plt.subplot(2, 2, 1)
plt.scatter(prev_uhi, new_uhi, alpha=0.5)
plt.plot([min(prev_uhi), max(prev_uhi)], [min(prev_uhi), max(prev_uhi)], 'r--')
plt.xlabel('Previous UHI Index Predictions')
plt.ylabel('New UHI Index Predictions')
plt.title('Prediction Comparison')

# Distribution of differences
plt.subplot(2, 2, 2)
plt.hist(np.abs(prev_uhi - new_uhi), bins=20)
plt.xlabel('Absolute Difference')
plt.ylabel('Frequency')
plt.title('Distribution of Differences')

# Percentage differences
plt.subplot(2, 2, 3)
plt.hist(pct_diff, bins=20)
plt.xlabel('Percentage Difference (%)')
plt.ylabel('Frequency')
plt.title('Distribution of Percentage Differences')

# Cumulative distribution
plt.subplot(2, 2, 4)
plt.hist(pct_diff, bins=20, cumulative=True, density=True)
plt.xlabel('Percentage Difference (%)')
plt.ylabel('Cumulative Probability')
plt.title('Cumulative Distribution of Differences')

plt.tight_layout()
plt.savefig('prediction_comparison_0315.png')
print(f"Comparison visualization saved to prediction_comparison_0315.png")

# Print the top 10 most different predictions
print("\nTop 10 locations with biggest differences:")
top_diff = comparison.sort_values('Percentage_Diff', ascending=False).head(10)
print(top_diff[['Latitude', 'Longitude', 'Previous_UHI', 'New_UHI', 'Percentage_Diff']])
