import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load CSV
csv_path = "phrases_ngram1.csv"   # change this
df = pd.read_csv(csv_path)
df = df.iloc[:10, :]

# Convert two columns into a dictionary: {"data": 347, "information": 74, ...}
freq_dict = dict(zip(df["phrase"], df["count"]))

# Create word cloud
wc = WordCloud(
    width=1200,
    height=600,
    background_color="white"
)

wordcloud_img = wc.generate_from_frequencies(freq_dict)

# Display word cloud
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud_img, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.show()

# Save
wordcloud_img.to_file("wordcloud.png")
print("Saved to wordcloud.png")
