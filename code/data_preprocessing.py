
# ===========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ===========================================
# 1. SET UP RELATIVE PATHS
# ===========================================

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define relative paths
data_path = os.path.join(script_dir, "..", "data", "financial_literacy_dataset.csv")
plots_dir = os.path.join(script_dir, "..", "results", "plots", "data_preprocessing")
tables_dir = os.path.join(script_dir, "..", "results", "tables", "data_preprocessing")

# Create directories if they don't exist
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)

print("=" * 60)
print("Starting Data Preprocessing for Financial Literacy System")
print("=" * 60)
print("Data path:", data_path)
print("Plots directory:", plots_dir)
print("Tables directory:", tables_dir)
print()

# ===========================================
# 2. LOAD THE DATASET
# ===========================================

print("Loading data...")
try:
    df = pd.read_csv(data_path)
    print("Data loaded successfully!")
    print("Number of rows:", df.shape[0])
    print("Number of columns:", df.shape[1])
except FileNotFoundError:
    print("Error: Data file not found!")
    print("Please check the path:", data_path)
    exit()

print("\nFirst few rows of data:")
print(df.head())
print()

# ===========================================
# 3. DATA CLEANING
# ===========================================

print("Starting data cleaning...")

# 3.1 Check for missing values
print("\n1. Checking for missing values:")
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing_Values': missing_values,
    'Percentage': missing_percentage
})

# Only show columns with missing values
missing_with_data = missing_df[missing_df['Missing_Values'] > 0]

if len(missing_with_data) > 0:
    print("Found missing values in these columns:")
    print(missing_with_data)
else:
    print("No missing values found in the data")

# 3.2 Check for duplicates
print("\n2. Checking for duplicates:")
duplicates_count = df.duplicated().sum()

if duplicates_count > 0:
    print("Found", duplicates_count, "duplicate rows")
    print("Removing duplicates...")
    df = df.drop_duplicates()
    print("Duplicates removed. New row count:", len(df))
else:
    print("No duplicate rows found in the data")

# 3.3 Check rating range
print("\n3. Checking rating range:")
rating_min = df['rating'].min()
rating_max = df['rating'].max()

print("Minimum rating:", rating_min)
print("Maximum rating:", rating_max)

if rating_min < 1 or rating_max > 5:
    print("Ratings are outside 1-5 range. Normalizing...")
    # Normalize ratings to 1-5 range
    df['rating'] = np.clip(df['rating'], 1, 5)
    print("Ratings normalized to 1-5 range")
else:
    print("All ratings are in the correct 1-5 range")

# ===========================================
# 4. BASIC STATISTICS
# ===========================================

print("\nCalculating basic statistics...")

# 4.1 Basic counts
num_users = df['user_id'].nunique()
num_items = df['item_id'].nunique()
num_ratings = len(df)

print("Number of unique users:", num_users)
print("Number of unique items:", num_items)
print("Number of ratings/interactions:", num_ratings)

# 4.2 Calculate sparsity
possible_ratings = num_users * num_items
sparsity = 1 - (num_ratings / possible_ratings)

print("\nData sparsity level:")
print("Total possible interactions:", possible_ratings)
print("Total actual interactions:", num_ratings)
print("Sparsity percentage:", round(sparsity * 100, 2), "%")

# 4.3 Rating distribution
print("\nRating distribution:")
rating_dist = df['rating'].value_counts().sort_index()
print(rating_dist)

# Save statistics to CSV file
stats_data = {
    'Metric': ['Number_of_Users', 'Number_of_Items', 'Number_of_Ratings', 'Sparsity'],
    'Value': [num_users, num_items, num_ratings, sparsity]
}
stats_df = pd.DataFrame(stats_data)
stats_path = os.path.join(tables_dir, "basic_statistics.csv")
stats_df.to_csv(stats_path, index=False)
print("\nStatistics saved to:", stats_path)

# ===========================================
# 5. EXPLORATORY DATA ANALYSIS
# ===========================================

print("\nCreating visualizations...")

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')

# 5.1 Rating distribution plot
plt.figure(figsize=(10, 6))
rating_counts = df['rating'].value_counts().sort_index()
bars = plt.bar(rating_counts.index.astype(str), rating_counts.values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6BAA75'][:len(rating_counts)])

# Add numbers on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 50,
             str(int(height)), ha='center', va='bottom', fontsize=10)

plt.title('Rating Distribution in the System', fontsize=14, fontweight='bold')
plt.xlabel('Rating (1 to 5)', fontsize=12)
plt.ylabel('Number of Ratings', fontsize=12)
plt.grid(True, alpha=0.3)

# Save the plot
rating_plot_path = os.path.join(plots_dir, "rating_distribution.png")
plt.tight_layout()
plt.savefig(rating_plot_path, dpi=300, bbox_inches='tight')
print("Rating distribution plot saved:", rating_plot_path)
plt.close()

# 5.2 User activity distribution
plt.figure(figsize=(12, 6))

# Calculate number of ratings per user
user_activity = df.groupby('user_id').size().sort_values(ascending=False)

# Categorize users by activity level
active_users = (user_activity >= 20).sum()
moderate_users = ((user_activity >= 5) & (user_activity < 20)).sum()
inactive_users = (user_activity < 5).sum()

# Pie chart for activity distribution
plt.subplot(1, 2, 1)
activity_labels = ['Very Active (20+ ratings)', 'Moderate (5-19 ratings)', 'Inactive (<5 ratings)']
activity_sizes = [active_users, moderate_users, inactive_users]
activity_colors = ['#2E86AB', '#F18F01', '#A23B72']

plt.pie(activity_sizes, labels=activity_labels, colors=activity_colors, autopct='%1.1f%%', startangle=90)
plt.title('User Activity Distribution', fontsize=12, fontweight='bold')

# Histogram of activity
plt.subplot(1, 2, 2)
user_activity_hist = user_activity[user_activity <= 50]  # Focus on reasonable range
plt.hist(user_activity_hist, bins=30, color='#2E86AB', edgecolor='black', alpha=0.7)
plt.axvline(user_activity_hist.mean(), color='red', linestyle='--', linewidth=2, label='Average: {:.1f}'.format(user_activity_hist.mean()))
plt.title('Ratings per User Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Number of Ratings per User', fontsize=10)
plt.ylabel('Number of Users', fontsize=10)
plt.legend()
plt.grid(True, alpha=0.3)

plt.suptitle('User Activity Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save the plot
user_activity_path = os.path.join(plots_dir, "user_activity_distribution.png")
plt.savefig(user_activity_path, dpi=300, bbox_inches='tight')
print("User activity plot saved:", user_activity_path)
plt.close()

# 5.3 Item popularity distribution
plt.figure(figsize=(12, 6))

# Calculate number of ratings per item
item_popularity = df.groupby('item_id').size().sort_values(ascending=False)

# Bar chart for top 20 items
plt.subplot(1, 2, 1)
top_items = item_popularity.head(20)
plt.barh(range(len(top_items)), top_items.values, color='#F18F01', edgecolor='black')
plt.yticks(range(len(top_items)), top_items.index)
plt.gca().invert_yaxis()  # Invert to show highest at top
plt.title('Top 20 Most Popular Items', fontsize=12, fontweight='bold')
plt.xlabel('Number of Ratings', fontsize=10)
plt.grid(True, alpha=0.3, axis='x')

# Histogram of item popularity
plt.subplot(1, 2, 2)
plt.hist(item_popularity, bins=30, color='#A23B72', edgecolor='black', alpha=0.7)
plt.axvline(item_popularity.mean(), color='red', linestyle='--', linewidth=2, label='Average: {:.1f}'.format(item_popularity.mean()))
plt.title('Item Popularity Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Number of Ratings per Item', fontsize=10)
plt.ylabel('Number of Items', fontsize=10)
plt.legend()
plt.grid(True, alpha=0.3)

plt.suptitle('Item Popularity Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save the plot
item_popularity_path = os.path.join(plots_dir, "item_popularity_distribution.png")
plt.savefig(item_popularity_path, dpi=300, bbox_inches='tight')
print("Item popularity plot saved:", item_popularity_path)
plt.close()

# 5.4 Long-tail problem analysis
plt.figure(figsize=(10, 6))

# Sort items from most to least popular
cumulative_items = np.arange(1, len(item_popularity) + 1)
cumulative_ratings = item_popularity.cumsum()
cumulative_percentage = (cumulative_ratings / cumulative_ratings.iloc[-1]) * 100

# Plot long-tail curve
plt.plot(cumulative_items, cumulative_percentage, color='#2E86AB', linewidth=3)

# Add reference lines
plt.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% of Interactions')
plt.axvline(x=len(item_popularity)*0.2, color='green', linestyle='--', alpha=0.5, label='20% of Items')

# Fill areas
plt.fill_between(cumulative_items, 0, cumulative_percentage, alpha=0.2, color='#2E86AB')
plt.fill_between(cumulative_items[:int(len(item_popularity)*0.2)], 0, 
                 cumulative_percentage[:int(len(item_popularity)*0.2)], 
                 alpha=0.4, color='#F18F01', label='Short Tail (20% of Items)')

plt.title('Long-Tail Problem Analysis', fontsize=14, fontweight='bold')
plt.xlabel('Items (Ranked from Most to Least Popular)', fontsize=12)
plt.ylabel('Cumulative % of Interactions', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# Calculate long-tail metrics
top_20_percent_items = int(len(item_popularity) * 0.2)
interactions_top_20 = item_popularity.head(top_20_percent_items).sum()
total_interactions = item_popularity.sum()
percentage_top_20 = (interactions_top_20 / total_interactions) * 100

# Add text annotation
plt.text(0.6, 0.3, '{:.1f}% of interactions\ncome from 20% of items'.format(percentage_top_20), 
         transform=plt.gca().transAxes, fontsize=11, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

long_tail_path = os.path.join(plots_dir, "long_tail_analysis.png")
plt.tight_layout()
plt.savefig(long_tail_path, dpi=300, bbox_inches='tight')
print("Long-tail analysis plot saved:", long_tail_path)
plt.close()

# ===========================================
# 6. SAVE RESULTS AND REPORTS
# ===========================================

print("\nSaving results...")

# 6.1 Save distribution reports
distributions = {
    'Rating': df['rating'].value_counts().sort_index(),
    'Financial_Knowledge': df['financial_knowledge'].value_counts(),
    'Content_Difficulty': df['difficulty'].value_counts(),
    'Content_Type': df['content_type'].value_counts(),
    'Primary_Topic': df['primary_topic'].value_counts()
}

for name, dist in distributions.items():
    dist_df = pd.DataFrame({'Category': dist.index, 'Count': dist.values})
    dist_path = os.path.join(tables_dir, "{}_distribution.csv".format(name))
    dist_df.to_csv(dist_path, index=False)
    print("{} distribution saved: {}".format(name, dist_path))

# 6.2 Save detailed statistics
detailed_stats = df.describe(include='all').T
detailed_stats_path = os.path.join(tables_dir, "detailed_statistics.csv")
detailed_stats.to_csv(detailed_stats_path)
print("Detailed statistics saved:", detailed_stats_path)

# 6.3 Long-tail report
long_tail_report = pd.DataFrame({
    'Metric': ['Percentage_of_Items_in_Short_Tail', 'Percentage_of_Interactions_in_Short_Tail', 'Imbalance_Index'],
    'Value': [20.0, percentage_top_20, percentage_top_20/20.0]
})
long_tail_report_path = os.path.join(tables_dir, "long_tail_report.csv")
long_tail_report.to_csv(long_tail_report_path, index=False)
print("Long-tail report saved:", long_tail_report_path)

# 6.4 Save cleaned data
clean_data_path = os.path.join(os.path.dirname(data_path), "cleaned_financial_data.csv")
df.to_csv(clean_data_path, index=False)
print("Cleaned data saved:", clean_data_path)

# ===========================================
# 7. RESULTS SUMMARY
# ===========================================

print("\n" + "=" * 60)
print("DATA PREPROCESSING SUMMARY")
print("=" * 60)

print("\nMAIN STATISTICS:")
print("  Number of users: {:,}".format(num_users))
print("  Number of items: {:,}".format(num_items))
print("  Number of interactions: {:,}".format(num_ratings))
print("  Sparsity level: {:.2%}".format(sparsity))

print("\nRATING DISTRIBUTION:")
for rating, count in rating_dist.items():
    percentage = (count / num_ratings) * 100
    print("  Rating {}: {:,} ({:.1f}%)".format(rating, count, percentage))

print("\nUSER ACTIVITY ANALYSIS:")
print("  Average ratings per user: {:.1f}".format(user_activity.mean()))
print("  Most active user: {} ratings".format(user_activity.max()))
print("  Least active user: {} ratings".format(user_activity.min()))

print("\nITEM POPULARITY ANALYSIS:")
print("  Average ratings per item: {:.1f}".format(item_popularity.mean()))
print("  Most popular item: {} ratings".format(item_popularity.max()))
print("  Least popular item: {} ratings".format(item_popularity.min()))

print("\nLONG-TAIL PROBLEM ANALYSIS:")
print("  20% of items receive {:.1f}% of interactions".format(percentage_top_20))
print("  Imbalance index: {:.2f} (higher than 1 indicates imbalance)".format(percentage_top_20/20.0))

if percentage_top_20 > 80:
    print("  WARNING: Strong long-tail problem detected (>80%)")
elif percentage_top_20 > 60:
    print("  WARNING: Moderate long-tail problem detected (60-80%)")
else:
    print("  GOOD: No significant long-tail problem detected")

print("\nFILES SAVED:")
print("  Plots: {} files in {}".format(len([f for f in os.listdir(plots_dir) if f.endswith('.png')]), plots_dir))
print("  Tables: {} files in {}".format(len([f for f in os.listdir(tables_dir) if f.endswith('.csv')]), tables_dir))
print("  Cleaned data: {}".format(clean_data_path))

print("\n" + "=" * 60)
print("DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 60)