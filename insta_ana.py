import instaloader

# Initialize an Instaloader instance.
L = instaloader.Instaloader()

# Define the username of the Instagram account you want to analyze.
username = "192.168.0.1p"

# Create a list to store post timestamps.
timestamps = []

# Fetch Instagram data for the specified account.
try:
    profile = instaloader.Profile.from_username(L.context, username)

    # Sort the posts by date (ascending order) and limit to the top 6 posts.
    top_posts = sorted(profile.get_posts(), key=lambda post: post.date_utc)[:6]

    # Extract timestamps for the top 6 posts.
    for post in top_posts:
        timestamps.append(post.date_utc)

except instaloader.exceptions.ProfileNotExistsException:
    print(f"The account '{username}' does not exist.")
    exit()

# Print the timing of the top 6 posts uploaded.
print("Timing of the top 6 posts uploaded:")
for i, timestamp in enumerate(timestamps, start=1):
    print(f"Post {i}: {timestamp}")

# You can further process or analyze the timestamps as needed.
