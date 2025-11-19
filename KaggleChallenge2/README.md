# **Kaggle Challenge 2: Dự báo giá nhà (House Prices - Advanced Regression Techniques)**

## **Danh sách thành viên:**
- Đỗ Quốc Việt - 3123410426
- Phạm Minh Dương - 3123410065
- Phạm Tấn Đạt - 3123410072
- Nguyễn Đăng Khoa - 3123410168

## **Mô tả**
Cuộc thi "House Prices - Advanced Regression Techniques" trên nền tảng Kaggle
là một bài toán hồi quy nâng cao, sử dụng dữ liệu thực tế về nhà ở tại Ames, Iowa, Mỹ. Bộ
dữ liệu bao gồm 79 đặc trưng mô tả các khía cạnh khác nhau của ngôi nhà, từ vị trí địa lý
(Neighborhood, LotArea), chất lượng xây dựng (OverallQual, Foundation), đến tiện nghi
(GarageCars, Fireplaces). Bài toán này kinh điển vì nó phản ánh các vấn đề thực tế trong
lĩnh vực bất động sản: dữ liệu thiếu, phân bố lệch (skewed distribution), outliers, và sự
tương tác phức tạp giữa các biến.
  
## **Mục tiêu**
- Xây dựng một pipeline hồi quy toàn diện để dự đoán giá bán nhà (SalePrice) từ các đặc
trưng nhà ở, đạt độ chính xác cao trên dữ liệu test ẩn của Kaggle.

## **Dataset**
- Nguồn dataset: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
- Thành phần của dataset:
   + train.csv: tập huấn luyện
   + test.csv: tập kiểm thử
   + data_description.txt: mô tả dữ liệu
   + sample_submission.csv: file nộp bài mẫu.

## **Yêu cầu** 
- python (3.13.5)
- catboost (1.2.8)
- conda (25.7.0)
- jupyter (1.1.1)
- lightgbm (4.6.0)
- numpy (2.1.3)
- pandas (2.3.2)
- scikit-learn (1.7.2)
- scipy (1.15.3)
- xgboost (3.0.1)

## **Baseline**

### Cấu trúc thư mục

```
KaggleChallenge2
│
├───clean
│   │   cleaning.ipynb
│   │   dist_box_sale_price.png
│   │   dist_hist_sale_price.png
│   │
│   ├───backup
│   │       cleaning_backup.ipynb
│   │       cleaning_backup.py
│   │
│   └───data
│           test_clean.csv
│           test_clean.pkl
│           train_clean.csv
│           train_clean.pkl
│
├───data
│       data_description.txt
│       sample_submission.csv
│       test.csv
│       train.csv
│
├───eda
│       eda.ipynb
│
├───exps
│   │   note_challenge2.xlsx
│   │
│   ├───features
│   │       features_13-11_19-59.html
│   │       ...
│   │
│   └───model
│           model_01-11_18-53.ipynb
│           ...
│
├───models
│       model.ipynb
│       model_stack.ipynb
│       saved_model.pkl
│
├───preprocessing
│   │   preprocessing.ipynb
│   │
│   └───data
│           test_processed.pkl
│           train_processed.pkl
│           train_y_processed.pkl
│           train_y_unprocessed.pkl
│
└───run
        run.ipynb
        submission.csv
```

### Cách chạy baseline

+ Chạy file ```cleaning.ipynb``` để dọn dẹp dữ liệu.
+ Chạy file ```preprocessing.ipynb``` để thực hiện việc tiền xử lý dữ liệu.
+ Chạy file ```model.ipynb``` và ```model_stack.ipynb``` để thực hiện việc huấn luyện mô hình.
+ Chạy file ```run.ipynb``` để xuất ra kết quả dự đoán.

## **Các mô hình và tham số được sử dụng**
- Mô hình Stacking Model, tổng hợp kết quả của 6 mô hình bao gồm RandomForest, GradientBoosting,
XGBRegressor, SVM, CatBoost và LightGBM. Dùng Linear Regression làm meta model.
- Các tham số mô hình:
   + Random Forest: n_jobs=-1, 
                    max_depth=8, 
                    min_samples_leaf=2, 
                    n_estimators=500
   + Gradient Boosting: n_estimators=2000,
                        learning_rate=0.03,
                        max_depth=5,
                        min_samples_leaf=2,
                        subsample=0.85,
                        max_features='sqrt'
   + XGBoost: max_depth=8,
            learning_rate=0.05,
            n_estimators=500,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            eval_metric='rmse'  

   + SVM: kernel='rbf',
                  C=50000,
                  epsilon=0.1 

   + CatBoost: depth=8,
               l2_leaf_reg=3,
               subsample=0.8,
               learning_rate=0.05,
               iterations=1000,
               loss_function='RMSE', 
               eval_metric='RMSE',
               verbose=0,
               allow_writing_files=False
   
   + LightGBM: n_estimators=1000,
               learning_rate=0.05,
               max_depth=-1,
               num_leaves=64,
               subsample=0.8,
               colsample_bytree=0.8,
               reg_lambda=3,
               n_jobs=-1

- Tham số meta model Linear Regression: n_jobs=-1
- Tham số K-Fold: 10
- Tham số random_state: 42

## Kết quả chính thức trên Kaggle
- Đạt mức điểm 0.11903 tính theo thang điểm RMSE của Kaggle. Kết quả đạt được nằm trong top 5% của bảng xếp hạng các lượt submission trên Kaggle.
