import pandas as pd
import numpy as np
from Naive_Bayes_Letter import CategoricalNaiveBayes

# Đọc dữ liệu (đã có sẵn file letter-recognition.data)
print("Đang đọc dữ liệu Letter Recognition...")
df = pd.read_csv("letter-recognition.data", header=None)
print(f"Đã load {len(df):,} mẫu thành công!")

# Cột 0 = ký tự (A-Z), cột 1–16 = 16 đặc trưng
y = df[0].values
X = df.iloc[:, 1:].values.astype(int)

# Chia train/test 80-20
np.random.seed(42)
indices = np.random.permutation(len(X))
split = int(0.8 * len(X))

X_train, X_test = X[indices[:split]], X[indices[split:]]
y_train, y_test = y[indices[:split]], y[indices[split:]]

print(f"Train: {len(X_train):,} mẫu | Test: {len(X_test):,} mẫu")

# Huấn luyện mô hình
model = CategoricalNaiveBayes(alpha=1.0)
model.fit(X_train, y_train)

# Đánh giá
accuracy = model.score(X_test, y_test)
print(f"\nĐỘ CHÍNH XÁC TRÊN TẬP TEST: {accuracy*100:.2f}%")

# Dự đoán vài mẫu
print("\n10 ví dụ dự đoán ngẫu nhiên:")
test_indices = np.random.randint(0, len(X_test), 10)
for i in test_indices:
    true = y_test[i]
    pred = model.predict([X_test[i]])[0]
    print(f"  {'✓' if pred == true else '✗'} Thực tế: {true:>2} | Dự đoán: {pred:>2}")