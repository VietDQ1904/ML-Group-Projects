# main.py
import pandas as pd
import random
from Naive_Bayes import Naive_Bayes

# Đọc dữ liệu
df = pd.read_csv("Iris.csv")
df.drop('Id', axis=1, inplace=True)

# Chuyển thành list rồi shuffle
data = df.values.tolist()
random.seed(42)
random.shuffle(data)

# Chia train/test: 120 train, 30 test
train_data = pd.DataFrame(data[:120])
test_data  = pd.DataFrame(data[120:])

# Huấn luyện và kiểm tra
model = Naive_Bayes(train_data)
model.test(test_data)