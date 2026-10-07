import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/sleep.csv")

cols = ["Age", "Sleep Duration", "Quality of Sleep",
        "Physical Activity Level", "Stress Level",
        "Heart Rate", "Daily Steps"]

corr = df[cols].corr()
print(corr["Quality of Sleep"])

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("What affects sleep quality?")
plt.tight_layout()
plt.show()