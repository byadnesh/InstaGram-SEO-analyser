import pandas as pd
import re
import matplotlib.pyplot as plt

# List of dataset file paths
dataset_paths = [
    "techburner_instagram_posts.csv",  # Replace with your dataset file paths
    "technical_sapien_instagram_posts.csv",
    "trakintech_instagram_posts.csv",
    "technicalguruji_instagram_posts.csv",
    "tech_unboxing007_instagram_posts.csv",
    "tech_iela_instagram_posts.csv",
    "manojsaru_instagram_posts.csv",
    "nomadatoast_instagram_posts.csv"
]

# Initialize an empty dictionary to store hashtag counts
hashtag_counts = {}

# Initialize an empty dictionary to store hourly views
hourly_views = {hour: 0 for hour in range(24)}

# Loop through the list of dataset file paths
for dataset_path in dataset_paths:
    df = pd.read_csv(dataset_path)

    # Extract and count hashtags
    for index, row in df.iterrows():
        captions = row['Caption']
        views = row['Views']
        datetime = pd.to_datetime(row['Date'] + ' ' + row['Time'])
        hour = datetime.hour

        # Check if 'captions' is a valid string
        if isinstance(captions, str):
            # Use regular expression to extract hashtags
            hashtags = re.findall(r'#\w+', captions)

            for hashtag in hashtags:
                # Add hashtag to the dictionary if not present, and update the views
                if hashtag in hashtag_counts:
                    hashtag_counts[hashtag] += views
                else:
                    hashtag_counts[hashtag] = views

        # Update the hourly views
        hourly_views[hour] += views

# Sort the hashtags by views in descending order
sorted_hashtags = sorted(hashtag_counts.items(), key=lambda x: x[1], reverse=True)

# Get the top 30 hashtags
top_30_hashtags = sorted_hashtags[:30]

# Find the hour with the highest average views
best_hour = max(hourly_views, key=hourly_views.get)
best_views = hourly_views[best_hour]

# Plot a line graph for hourly views
plt.figure(figsize=(10, 6))
plt.plot(hourly_views.keys(), hourly_views.values(), marker='o', linestyle='-')
plt.title('Average Views by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Views')
plt.xticks(range(24))
plt.grid()

# Display the best hour to post
print(f'The best time to post for maximum views is around {best_hour}:00 with an average of {best_views:.2f} views.')

# Display the top 30 hashtags
for i, (hashtag, views) in enumerate(top_30_hashtags, start=1):
    print(f"{i}. {hashtag}: {views} views")

# Create a bar graph for the top hashtags
plt.figure(figsize=(12, 6))
top_hashtags, top_views = zip(*top_30_hashtags)
plt.barh(top_hashtags, top_views)
plt.title('Top 30 Hashtags by Views')
plt.xlabel('Views')
plt.gca().invert_yaxis()

# Show both plots
plt.show()
