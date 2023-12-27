import instaloader
from bs4 import BeautifulSoup
import pandas as pd
import re
import time

# Initialize Instaloader
L = instaloader.Instaloader()


# Replace 'target_username' with the Instagram username you want to scrape
username = '192.168.0.1p'

# Initialize a list to store post data
post_data = []

# Login (optional)
L.login('king_pink69', 'tahamadarchod')

# Get the profile of the target Instagram account
profile = instaloader.Profile.from_username(L.context, username)

# Get the top 500 posts (you can change this number if needed)
count = 0
for post in profile.get_posts():
    caption = post.caption if post.caption is not None else ''  # Check if caption is None
    # Insert the delay here
    time.sleep(1)  # Add a 1-second delay before each request
    post_data.append({
        'Date': post.date.strftime('%Y-%m-%d'),
        'Time': post.date.strftime('%H:%M:%S'),
        'Views': post.video_view_count,
        'Likes': post.likes,
        'Comments': post.comments,
        'Caption': post.caption,
        'Hashtags': ', '.join(re.findall(r'#\w+', caption if isinstance(caption, str) else ''))
    })
    count += 1

    if count >= 500:
        break

# Create a DataFrame
df = pd.DataFrame(post_data)

# Save the data to a CSV file
df.to_csv(f'{username}_instagram_posts_LIKES.csv', index=False)

# Optional: Logout (if you logged in)
# L.logout()
