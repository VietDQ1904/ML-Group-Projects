# Naive_Bayes.py
import numpy as np

class Naive_Bayes:
    def __init__(self, data_set):
        self.ds = data_set
        # Tính mean và variance của từng đặc trưng theo từng loài hoa
        self.ds_means = self.ds.groupby(4).mean()
        self.ds_variances = self.ds.groupby(4).var()
        # Xác suất tiên nghiệm của từng class
        self.class_probabilities = self.get_class_probabilities(self.ds)

    def get_class_probabilities(self, data_set):
        class_sizes = data_set.groupby(4).size()
        ds_total = data_set.shape[0]
        probs = {}
        for class_name, count in class_sizes.items():
            probs[class_name] = count / ds_total
        return probs

    def get_probability_density(self, x, mean, variance):
        # Tránh chia cho 0
        variance = max(variance, 1e-9)
        exponent = np.exp(- (x - mean) ** 2 / (2 * variance))
        return (1 / (np.sqrt(2 * np.pi * variance))) * exponent

    def predict(self, x):
        posteriors = {}
        for group, prior in self.class_probabilities.items():
            post = prior
            for i in range(4):  # 4 đặc trưng
                mean = self.ds_means.loc[group][i]
                var = self.ds_variances.loc[group][i]
                post *= self.get_probability_density(x[i], mean, var)
            posteriors[group] = post
        return max(posteriors, key=posteriors.get)

    def test(self, test_data):
        correct = 0
        total = test_data.shape[0]
        print("=== Kết quả kiểm tra trên tập test ===")
        print("Các mẫu dự đoán sai:")
        
        for idx, row in enumerate(test_data.itertuples(), start=121):
            features = row[1:5]
            true_label = row[5]
            pred = self.predict(features)
            if pred != true_label:
                print(f"  Mẫu {idx}: {features} → Dự đoán: {pred:15} | Thực tế: {true_label}")
            else:
                correct += 1

        accuracy = correct / total * 100
        print(f"\n=> Độ chính xác: {accuracy:.2f}% ({correct}/{total} mẫu đúng)\n")