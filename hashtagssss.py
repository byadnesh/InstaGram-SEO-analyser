import pandas as pd
import re

# List of dataset file paths
dataset_paths = [
    "nomadatoast_instagram_posts.csv",  # Replace with your dataset file paths
    "sanidhya.ai_instagram_posts.csv",
    "technical_sapien_instagram_posts.csv"
]

# Initialize an empty dictionary to store hashtag counts
hashtag_counts = {}

# Loop through the list of dataset file paths
for dataset_path in dataset_paths:
    df = pd.read_csv(dataset_path)

    # Extract and count hashtags
    for index, row in df.iterrows():
        captions = row['Caption']
        views = row['Views']

        # Use regular expression to extract hashtags
        hashtags = re.findall(r'#\w+', captions)

        for hashtag in hashtags:
            # Add hashtag to the dictionary if not present, and update the views
            if hashtag in hashtag_counts:
                hashtag_counts[hashtag] += views
            else:
                hashtag_counts[hashtag] = views

# Sort the hashtags by views in descending order
sorted_hashtags = sorted(hashtag_counts.items(), key=lambda x: x[1], reverse=True)

# Get the top 30 hashtags
top_30_hashtags = sorted_hashtags[:30]

# Display the top 30 hashtags
for i, (hashtag, views) in enumerate(top_30_hashtags, start=1):
    print(f"{i}. {hashtag}: {views} views")

# You can also create a new CSV file to save the top 30 hashtags and their views
top_hashtags_df = pd.DataFrame(top_30_hashtags, columns=['Hashtag', 'Views'])
top_hashtags_df.to_csv('top_hashtags.csv', index=False)

# If you want to plot the top 30 hashtags, you can use matplotlib as shown in the previous code example
