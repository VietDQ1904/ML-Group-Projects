import pandas as pd
import random
from collections import defaultdict
from K_Nearest_Neighbors import K_Nearest_Neighbors as KNN
import os

# Lấy đúng thư mục chứa file .py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "data", "letter+recognition", "letter-recognition.data")

# Đọc file
df = pd.read_csv(file_path, header=None)

columns = [
    "lettr", "x-box", "y-box", "width", "high", "onpix",
    "x-bar", "y-bar", "x2bar", "y2bar", "xybar",
    "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"
]
df.columns = columns

# Chuyển sang int
for col in columns[1:]:
    df[col] = df[col].astype(int)

# Chia tập train và test
train_data = defaultdict(list)
test_data  = defaultdict(list)

for label in df["lettr"].unique():
    subset = df[df["lettr"] == label].values.tolist()
    random.shuffle(subset)

    # Chia tỷ lệ 80% train và 20% test.
    split = int(len(subset) * 0.8)

    for row in subset[:split]:
        train_data[label].append(row[1:])

    for row in subset[split:]:
        test_data[label].append(row[1:])

# Train KNN
print("Training...")
knn = KNN(train_data, k=5)
knn.test(test_data)
