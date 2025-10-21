# Báo cáo EDA — Phân tích dữ liệu Titanic

Phiên bản: 1.0

Lược đồ: dựa trên notebook `eda.ipynb` trong cùng thư mục. Báo cáo này tóm tắt và diễn giải các khám phá đồ họa và thống kê đã thực hiện trong notebook, đưa ra nhận xét, kết luận tạm thời và khuyến nghị cho bước tiền xử lý và mô hình hoá tiếp theo.

---

## 1. Mục tiêu

Mục tiêu của EDA là hiểu cấu trúc dữ liệu, các phân bố chính, mối quan hệ giữa biến (đặc biệt là với nhãn `Survived`), phát hiện missing values và outliers, và đề xuất các bước tiền xử lý / feature engineering để chuẩn bị cho mô hình phân loại (dự đoán khả năng sống sót).

Ghi chú: notebook sử dụng tập `train.csv` làm nguồn chính để khám phá (tệp được đọc vào `df_train` và gán vào `df_dataset`). Nếu bạn đang dùng một tập dữ liệu khác (ví dụ hợp nhất train+test), một số kết luận về tỷ lệ phần trăm và kích thước mẫu có thể khác. Ở một vài nơi tôi suy đoán cấu trúc phổ biến của bộ dữ liệu Titanic (ví dụ các cột tiêu chuẩn), nhưng không tạo số liệu tóm tắt giả định mà không có dữ liệu thô rõ ràng.

---

## 2. Tổng quan dữ liệu & biến

Các cột chính (đã được ghi trong notebook):

- `Survived`: nhãn mục tiêu (0 = không sống sót, 1 = sống sót)
- `Pclass`: hạng vé (1, 2, 3)
- `Sex`: giới tính (`male`/`female`)
- `Age`: tuổi
- `SibSp`: số anh/chị/em và vợ/chồng đi cùng
- `Parch`: số cha/mẹ/con đi cùng
- `Fare`: giá vé
- `Embarked`: cảng khởi hành (S, C, Q)
- Các cột khác thường có trong dataset gốc: `Name`, `Ticket`, `Cabin` (notebook đề xuất drop hoặc xử lý riêng)

Kiểm tra nhanh (notebook thực thi `shape`, `info()`, `describe()`, `isnull().sum()`):

- Notebook đã in cấu trúc và kiểm tra missing values; các nhận xét chi tiết về missing values (ví dụ `Age` có missing) được dùng làm cơ sở cho khuyến nghị imputation bên dưới.

---

## 3. Phân tích đơn biến (Univariate)

3.1. Fare

- Quan sát: phân bố `Fare` lệch phải mạnh; tồn tại một số giá vé rất lớn (outliers). Histogram từ `df_dataset.hist()` thể hiện đuôi phải dài.
- Ảnh hưởng: outliers và độ lệch lớn khiến các mô hình tuyến tính/độ đo khoảng cách bị ảnh hưởng; giá vé cao tương quan dương với khả năng sống sót.
- Khuyến nghị: thử log-transform (ví dụ `log1p(Fare)`), hoặc binning (fare bins), hoặc winsorize tại 99th percentile.

3.2. Age

- Quan sát: tuổi tập trung chủ yếu ở nhóm người trưởng thành (khoảng 20-40). Notebook cho thấy nhiều giá trị missing trên `Age`.
- Ảnh hưởng: thiếu `Age` có thể làm giảm hiệu suất nếu không impute hợp lý; phân bố tuổi cho thấy trẻ em có tỷ lệ sống cao hơn.
- Khuyến nghị: impute theo nhóm (median theo `Pclass` và `Sex`) hoặc dùng mô hình dự đoán tuổi; tạo thêm feature `is_child` (ví dụ Age < 16) và `AgeGroup` (bins).

3.3. SibSp, Parch

- Quan sát: nhiều giá trị 0 (nhiều hành khách đi một mình). Phân bố đếm nhỏ (0,1,2...).
- Ảnh hưởng: có thể kết hợp thành `family_size = SibSp + Parch + 1` (bao gồm chính người đó) và phân loại thành `solo`, `small`, `large` families.

3.4. Pclass, Sex, Embarked

- Quan sát: `Pclass` là một proxy cho tầng lớp xã hội; `Sex` (male/female) thường có phân bố khác biệt theo `Survived` (phụ nữ thường có khả năng sống cao hơn). Notebook không hiển thị trực tiếp các countplot, nhưng các pairplot/violin/boxplot đã phản ánh mối quan hệ.

---

## 4. Phân tích hai biến & đa biến (Bivariate / Multivariate)

4.1. Pairplot

- Mô tả: pairplot màu theo `Survived` cung cấp cái nhìn trực quan về mối quan hệ giữa các biến số.
- Quan sát: `Fare` và `Pclass` phân biệt rõ giữa các nhãn; trẻ em (nhóm tuổi nhỏ) xuất hiện nhiều hơn trong nhóm sống sót.

4.2. Heatmap / Matshow (Ma trận tương quan)

- Kết luận chính từ ma trận tương quan:
	- `Pclass` và `Fare` có tương quan âm mạnh (~ -0.549).
	- `Fare` và `Survived` có tương quan dương nhẹ (~ 0.257).
	- `SibSp` và `Parch` có tương quan dương vừa phải (~ 0.415).

- Diễn giải:
	- Mối tương quan âm giữa `Pclass` và `Fare` có ý nghĩa thực tế: hạng cao (Pclass=1) đi kèm giá vé cao; hạng 3 đi cùng giá vé thấp.
	- Tương quan dương giữa `Fare` và `Survived` gợi ý hành khách trả vé cao có lợi thế (tiếp cận cứu hộ, vị trí cabin tốt hơn) — nhưng đây chưa phải bằng chứng nhân quả.

4.3. Boxplot và Violinplot

- `sns.boxplot(x='Survived', y='Fare', ...)`: median `Fare` cao hơn trong nhóm sống; nhiều outliers.
- `df_dataset.boxplot(by='Survived')`: tổng quan các biến phân theo nhãn, xác nhận outliers và phạm vi biến khác nhau.
- `sns.violinplot(x='Survived', y='Age', ...)`: trẻ em có mật độ cao hơn trong nhóm sống, hỗ trợ giả thiết "trẻ em được cứu trước".

---

## 5. Outliers & Missing Values

- Outliers: Giá vé (`Fare`) chứa outliers rõ rệt; một số cá nhân trả vé rất cao (cần kiểm tra xem là vé hợp lệ hay lỗi nhập dữ liệu). Outliers làm méo kết quả mô hình và tương quan.
- Missing: `Age` có missing values (notebook đã kiểm tra `isnull().sum()`); `Cabin` thường có nhiều missing trong dataset Titanic gốc — notebook khuyến nghị có thể drop hoặc xử lý riêng.

Khuyến nghị xử lý:

- `Age`: impute theo chiến lược có ý nghĩa (median theo `Pclass` & `Sex`) hoặc mô hình hồi quy để dự đoán tuổi; sau imputation, kiểm tra phân bố lại.
- `Fare`: áp dụng log1p trước khi cho vào mô hình hoặc cắt tại percentile (ví dụ 99th) nếu outliers không phải thông tin thực tế.
- `Cabin`: do nhiều missing, có thể tách thành `has_cabin` (0/1) hoặc lấy ký tự đầu của cabin (chỉ nếu có signal) — nếu missing quá nhiều, drop hoặc xử lý đặc biệt.

---

## 6. Gợi ý feature engineering (từ EDA)

1. family_size = SibSp + Parch + 1 (số thành viên đi cùng)
2. is_alone = (family_size == 1)
3. fare_log = log1p(Fare) hoặc fare_bins (quartiles hoặc custom bins)
4. age_group / is_child (Age < 16 hoặc bins: child, young, adult, senior)
5. title extraction từ `Name` (Mr/Mrs/Miss/Dr...) — có thể bắt thông tin địa vị xã hội/giới tính
6. encode `Pclass` (one-hot hoặc ordinal) và `Sex` (binary)
7. has_cabin (0/1) từ `Cabin` để biểu thị thông tin bị thiếu nhưng có ý nghĩa
8. interaction features: `Sex * Pclass`, `AgeGroup * Pclass`, hoặc `Fare * Pclass` (hoặc `fare per person` khi biết family_size)

---

## 7. Ảnh hưởng đến mô hình & khuyến nghị cho pipeline

- Chuẩn hoá / Scale: các mô hình nhạy với scale (kNN, SVM, Logistic nếu dùng regularization khác nhau) cần chuẩn hoá `Age`, `Fare` (sau log) và các biến numeric khác (StandardScaler hoặc MinMax).
- Xử lý categorical: `Sex`, `Embarked`, `Pclass` nên được one-hot hoặc target-encoding (với cross validation) tuỳ mô hình.
- Missing values: apply imputation (Age), hoặc mô hình hồi quy để impute; giữ cờ missing nếu có thể tạo thông tin.
- Multicollinearity: `Pclass` và `Fare` tương quan khá cao; nếu dùng mô hình tuyến tính mạnh (Linear regression/Logistic) cần kiểm tra VIF hoặc tránh đưa cả hai thô vào mà không xử lý tương tác.

---

## 8. Kiểm định thống kê đề xuất

- Để xác nhận những khác biệt quan sát được:
	- So sánh `Fare` giữa hai nhóm `Survived` bằng t-test (hoặc Mann-Whitney nếu không phân phối chuẩn).
	- Kiểm tra liên hệ `Sex` vs `Survived` bằng test Chi-square (2x2 contingency table).
	- So sánh phân phối `Age` giữa hai nhóm bằng Kolmogorov-Smirnov hoặc Mann-Whitney.

---

## 9. Các bước tiếp theo (kế hoạch hành động ngắn hạn)

1. Impute `Age` theo `Pclass` & `Sex`, tạo `is_child`, `age_group`.
2. Transform `Fare` bằng `log1p` và kiểm tra biểu đồ lại.
3. Tạo `family_size` và `is_alone` từ `SibSp`/`Parch`.
4. Rút trích `Title` từ `Name`, thử gộp các title ít gặp.
5. Encode categorical và thử baseline models (LogisticRegression, RandomForest, XGBoost / CatBoost) với cross-validation.
6. Lựa chọn feature bằng permutation importance / SHAP để kiểm tra ảnh hưởng thực tế.

---

## 10. Phần phụ lục — Mã/command tái tạo trực quan (tham khảo từ notebook)

Một số lệnh được dùng trong notebook để tạo các hình chính (đặt trong cell code của notebook):

```python
# Pairplot (hight-level pairwise plots)
sns.pairplot(df_dataset, hue='Survived')

# Heatmap tương quan
numeric_df = df_dataset.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), cmap='Blues', annot=True)

# Matshow tương quan (matplotlib)
correlations = numeric_df.corr()
plt.matshow(correlations, vmin=-1, vmax=1); plt.colorbar()

# Histograms
df_dataset.hist()

# Boxplot Fare theo Survived
sns.boxplot(x='Survived', y='Fare', data=df_dataset)

# Violinplot Age theo Survived
sns.violinplot(x='Survived', y='Age', data=df_dataset)
```

Bạn có thể chạy các cell này trong `eda.ipynb` để tái tạo từng biểu đồ.

---

## 11. Tóm tắt ngắn gọn

- Các phát hiện chính: `Pclass` và `Fare` liên quan chặt; `Fare` có tương quan dương với `Survived`; `Age` (nhất là trẻ em) có ảnh hưởng; `SibSp`/`Parch` gợi ý cấu trúc gia đình.
- Hành động đề xuất: impute `Age`, transform `Fare`, tạo `family_size`/`is_alone`, rút trích `Title`, scale numeric và thử nhiều mô hình với CV; kiểm tra multicollinearity và xử lý outliers.

---

Nếu bạn muốn, tôi có thể tiếp tục và thực hiện trực tiếp các bước tiền xử lý đề xuất trên `df_dataset` trong notebook (ví dụ: thêm một cell code để impute Age, tạo feature `family_size`, apply log-transform cho Fare, và vẽ lại các biểu đồ để so sánh trước/sau). Hãy cho biết bạn muốn tôi áp dụng changes tự động vào notebook hay chỉ tạo các cell code mẫu để bạn chèn và chạy.

