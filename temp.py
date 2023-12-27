import pandas as pd
import matplotlib.pyplot as plt

# List of dataset file paths
dataset_paths = ["tech_iela_instagram_posts.csv", "technical_sapien_instagram_posts.csv", "tech_unboxing007_instagram_posts.csv","techburner_instagram_posts.csv","sanidhya.ai_instagram_posts.csv","trakintech_instagram_posts.csv","manojsaru_instagram_posts.csv"]  # Add your file paths here

# Initialize an empty DataFrame to store the combined data
combined_df = pd.DataFrame()

# Loop through the list of dataset file paths and concatenate them
for dataset_path in dataset_paths:
    df = pd.read_csv(dataset_path)
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# Convert the 'Date' and 'Time' columns to a datetime object
combined_df['Datetime'] = pd.to_datetime(combined_df['Date'] + ' ' + combined_df['Time'])

# Group the data by the hour of the day and calculate the average views
combined_df['Hour'] = combined_df['Datetime'].dt.hour
hourly_views = combined_df.groupby('Hour')['Views'].mean()

# Plot the average views by hour
plt.figure(figsize=(12, 6))
plt.plot(hourly_views.index, hourly_views.values, marker='o', linestyle='-')
plt.title('Average Views by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Views')
plt.xticks(range(24))
plt.grid()

# Find the hour with the highest average views
best_hour = hourly_views.idxmax()
best_views = hourly_views.max()

print(f'The best time to post for maximum views is around {best_hour}:00 with an average of {best_views:.2f} views.')

plt.show()
