import instaloader
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

# Initialize an Instaloader instance.
L = instaloader.Instaloader()
L.login("king_pink69","tahamadarchod")

# Define the username of the Instagram account you want to analyze.
username = "technical_sapien"

# Create lists to store post timestamps and view counts.
timestamps = []
view_counts = []

# Fetch Instagram data for the specified account.
try:
    profile = instaloader.Profile.from_username(L.context, username)

    # Sort the posts by the number of views (descending order) and limit to the top 100 posts.
    top_posts = sorted(profile.get_posts(), key=lambda post: post.video_view_count if post.is_video else 0, reverse=True)[:100]

    # Extract timestamps and view counts for the top 100 posts.
    for post in top_posts:
        timestamps.append(post.date_utc)
        view_counts.append(post.video_view_count if post.is_video else 0)

except instaloader.exceptions.ProfileNotExistsException:
    print(f"The account '{username}' does not exist.")
    exit()

# Convert timestamps to hours for better visualization.
hours = [(timestamp - timestamps[0]).total_seconds() / 3600 for timestamp in timestamps]

# Create various types of plots to compare post timing vs. view count.
plt.figure(figsize=(12, 6))

# Line Plot
plt.subplot(2, 2, 1)
plt.plot(hours, view_counts, marker='o', linestyle='-', color='blue', alpha=0.7)
plt.xlabel("Time Since First Post (hours)")
plt.ylabel("View Count")
plt.title("Line Plot")

# Scatter Plot
plt.subplot(2, 2, 2)
plt.scatter(hours, view_counts, marker='o', c='red', alpha=0.7)
plt.xlabel("Time Since First Post (hours)")
plt.ylabel("View Count")
plt.title("Scatter Plot")

# Bar Plot
plt.subplot(2, 2, 3)
plt.bar(hours, view_counts, width=0.2, color='green', align='center', alpha=0.7)
plt.xlabel("Time Since First Post (hours)")
plt.ylabel("View Count")
plt.title("Bar Plot")

# Histogram
plt.subplot(2, 2, 4)
sns.histplot(hours, kde=True, color='purple')
plt.xlabel("Time Since First Post (hours)")
plt.ylabel("Frequency")
plt.title("Histogram")

plt.tight_layout()
plt.show()

# Interactive Plotly Scatter Plot
fig = px.scatter(x=hours, y=view_counts, title="Interactive Scatter Plot")
fig.update_traces(marker=dict(size=8, opacity=0.7), selector=dict(mode='markers+text'))
fig.update_xaxes(title_text="Time Since First Post (hours)")
fig.update_yaxes(title_text="View Count")
fig.show()
