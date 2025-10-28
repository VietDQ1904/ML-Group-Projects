#!/usr/bin/env python
# coding: utf-8

# # **Kaggle Challenge 2: Đề tài dự báo giá nhà.**
# # **PHẦN 3: TIỀN XỬ LÝ DỮ LIỆU**

# ## 1. Định nghĩa vấn đề
# + **Mô tả**:
#    - Dự báo giá nhà dựa trên 79 tính chất của dataset nhà ở tại Ames, Iowa.
#    - Dữ liệu đầu vào gồm 2 file:
#       * train_clean.pkl: Dữ liệu từ file train đã được xử lý ở phần 1.
#       * test_clean.pkl: Dữ liệu từ file test đã được xử lý ở phần 1.
# 
# + **Mục tiêu**:
#    - Xử lý dữ liệu để các mô hình máy học có thể học hiệu quả, thêm một số đặc trưng giúp cải tiến hiệu suất mô hình.
# 

# ## 2. Chuẩn bị                

# ### 2.1. Import các thư viện cần thiết

# In[151]:


# Load libraries
from IPython import display
import numpy as np
import pickle

import matplotlib.pyplot as plt
import random

import pandas as pd
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler

# Hiển thị tối đa 200 cột
pd.set_option('display.max_columns', 200)


# ### 2.2. Đọc dataset train và test từ dataset đã xử lý trong phần dọn dẹp dữ liệu
# Đọc các file .pkl trong thư mục /clean/data

# In[152]:


train_data = pd.DataFrame(pd.read_pickle("../clean/data/train_clean.pkl"))
test_data = pd.DataFrame(pd.read_pickle("../clean/data/test_clean.pkl"))


# ### 2.3. Kiểm tra dữ liệu dataset train và test

# In[153]:


train_data.head()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  


# In[154]:


test_data.head()


# In[155]:


print("Số cột null trong train: ", train_data.isnull().sum().gt(0).sum())
print("Số cột null trong test: ", test_data.isnull().sum().gt(0).sum())


# In[156]:


train_data.dtypes.value_counts()


# In[157]:


test_data.dtypes.value_counts()


# ## 3. Tiền xử lý dữ liệu

# In[158]:


train_y = train_data["SalePrice"]  # tách biến mục tiêu ra khỏi tập train
train_y.to_pickle("./data/train_y_unprocessed.pkl") # Lưu file chưa xử lý để phục hồi lại giá trị ban đầu sau khi train.
train_data = train_data.drop(columns="SalePrice") 


# ### 3.1. Xử lý biến mục tiêu
# Dùng log transform của Numpy để xử lý biến mục tiêu do dữ liệu bị lệch phải.

# In[159]:


train_y = np.log1p(train_y)


# ### 3.2. Xử lý các cột số liên tục

# Dùng log transform để xử lý các giá trị lớn hơn 0 (do 0 được hiểu là không có).

# In[160]:


processed_cols = [
    "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1",
    "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF",
    "2ndFlrSF", "LowQualFinSF", "GrLivArea", "GarageArea",
    "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch",
    "ScreenPorch", "PoolArea", "MiscVal"
]

# Chỉ log transform các cột có giá trịlớn hơn 0
for col in processed_cols:
    train_data[col] = train_data[col].apply(lambda x: np.log1p(x) if x > 0 else 0)
    test_data[col] = test_data[col].apply(lambda x: np.log1p(x) if x > 0 else 0)


# ### 3.3. Xử lý các cột phi số

# In[161]:


train_data.select_dtypes(include=["object"]).columns


# #### **1. Các cột có tính chất phân loại được xử lý bằng one hot encoding.**

# In[162]:


categorical_data = ["MSZoning", "Street", "PavedDrive", "Alley", "LotConfig", "Utilities", "Neighborhood",
                    "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd",
                    "MasVnrType", "Foundation", "Heating", "CentralAir", "Electrical", "GarageType",
                    "Fence", "MiscFeature", "SaleType", "SaleCondition"]

numerical_data = train_data.drop(columns=categorical_data).columns

# One-hot encode các cột categorical
encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
encoder.fit(train_data[categorical_data])

train_cat = pd.DataFrame(
    encoder.transform(train_data[categorical_data]),
    columns=encoder.get_feature_names_out(categorical_data),
    index=train_data.index
)
test_cat = pd.DataFrame(
    encoder.transform(test_data[categorical_data]),
    columns=encoder.get_feature_names_out(categorical_data),
    index=test_data.index
)

# Nối lại với phần numeric
train_data = pd.concat([train_data[numerical_data], train_cat], axis=1)
test_data = pd.concat([test_data[numerical_data], test_cat], axis=1)


# #### **2. Các cột có tính chất thứ tự được map lại theo thứ tự.**

# In[163]:


for df in [train_data, test_data]:
    df["LotShape"] = df["LotShape"].map({"Reg": 3, "IR1": 2, "IR2": 1, "IR3": 0})
    df["LandContour"] = df["LandContour"].map({"Lvl": 3, "Bnk": 2, "HLS": 1, "Low": 0})
    df["LandSlope"] = df["LandSlope"].map({"Gtl": 2, "Mod": 1, "Sev": 0})
    df["BsmtQual"] = df["BsmtQual"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0})
    df["BsmtCond"] = df["BsmtCond"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0})
    df["BsmtExposure"] = df["BsmtExposure"].map({"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0})
    df["BsmtFinType1"] = df["BsmtFinType1"].map({"GLQ": 6,"ALQ": 5,"BLQ": 4,
												"Rec": 3,"LwQ": 2,"Unf": 1,
												"None": 0})
    df["BsmtFinType2"] = df["BsmtFinType2"].map({"GLQ": 6,"ALQ": 5,"BLQ": 4,
												"Rec": 3,"LwQ": 2,"Unf": 1,
												"None": 0})
    df["ExterQual"] = df["ExterQual"].map({"Ex": 3, "Gd": 2, "TA": 1, "Fa": 0})
    df["ExterCond"] = df["ExterCond"].map({"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "Po": 0})
    df["HeatingQC"] = df["HeatingQC"].map({"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "Po": 0})
    df["KitchenQual"] = df["KitchenQual"].map({"Ex": 3, "Gd": 2, "TA": 1, "Fa": 0})    
    df["FireplaceQu"] = df["FireplaceQu"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0})
    df["Functional"] = df["Functional"].map({"Typ": 7, "Min1": 6, "Min2": 5,
											 "Mod": 4, "Maj1": 3, "Maj2": 2,
											 "Sev": 1, "Sal": 0})
    df["GarageFinish"] = df["GarageFinish"].map({"Fin": 3, "RFn": 2, "Unf": 1, "None": 0}) 
    df["GarageQual"] = df["GarageQual"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0})
    df["GarageCond"] = df["GarageCond"].map({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0})
    df["PoolQC"] = df["PoolQC"].map({"Ex": 3, "Gd": 2, "Fa": 1, "None": 0})


# ### 3.4. Xử lý các cột rời rạc có tính chất phân loại (YrSold, MoSold, MSSubClass)

# In[164]:


processed_cols = ["MSSubClass", "YrSold"]

# One-hot encode cho MSSubClass
encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
encoder.fit(train_data[processed_cols])

train_cat = pd.DataFrame(
    encoder.transform(train_data[processed_cols]),
    columns=encoder.get_feature_names_out(processed_cols),
    index=train_data.index
)
test_cat = pd.DataFrame(
    encoder.transform(test_data[processed_cols]),
    columns=encoder.get_feature_names_out(processed_cols),
    index=test_data.index
)

# Nối lại với phần numeric
train_data = pd.concat([train_data, train_cat], axis=1)
test_data = pd.concat([test_data, test_cat], axis=1)

# Dùng Cyclic Encoding để giữ tính chất tuần hoàn (tháng 12 và 1 gần nhau).
for df in [train_data, test_data]:
	df['MoSold_sin'] = np.sin(2 * np.pi * df['MoSold'] / 12)
	df['MoSold_cos'] = np.cos(2 * np.pi * df['MoSold'] / 12)


# ### 3.5. Kiểm tra dữ liệu tập train và test sau khi xử lý

# In[165]:


train_data.info()


# In[166]:


test_data.info()


# In[167]:


train_data.head()


# In[168]:


test_data.head()


# ### 3.6. Chuẩn hóa dữ liệu

# #### **1. Chuẩn hóa biến mục tiêu bằng RobustScaler**
# Dùng RobustScaler để xử lý khi dữ liệu có các giá trị ngoại lai rất lớn. Dữ liệu được scale theo IQR, là khoảng cách giữa tứ phân vị thứ nhất ($Q_1$ hay phân vị 25%) và tứ phân vị thứ ba ($Q_3$ hay phân vị 75%). Công thức scale là:
# $$ 
# \\ x_{scaled} = \frac{x - \text{Median}(x)}{\text{IQR}(x)}
# $$
# Trong đó: $\text{IQR}(x) = Q_3 - Q_1$.

# In[169]:


scaler = RobustScaler()
train_y = scaler.fit_transform(train_y.values.reshape(-1, 1))


# #### **2. Chuẩn hóa các cột liên tục bằng RobustScaler**
# Dùng RobustScaler để xử lý khi dữ liệu có các giá trị ngoại lai rất lớn, không scale với giá trị 0.

# In[ ]:


processed_cols = [
    "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1",
    "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF",
    "2ndFlrSF", "LowQualFinSF", "GrLivArea", "GarageArea",
    "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch",
    "ScreenPorch", "PoolArea", "MiscVal"
]

for col in processed_cols:
    scaler = RobustScaler()

    # Chỉ lấy giá trị > 0 để fit scaler
    non_zero_train = train_data[col][train_data[col] > 0].values.reshape(-1,1)

    if len(non_zero_train) > 0:
        scaler.fit(non_zero_train)

        # Chuẩn hóa giá trị > 0, giữ 0 nguyên
        train_data[col] = train_data[col].apply(lambda x: scaler.transform([[x]])[0][0] if x > 0 else 0)
        test_data[col] = test_data[col].apply(lambda x: scaler.transform([[x]])[0][0] if x > 0 else 0)


# #### **3. Chuẩn hóa các cột YearBuilt, YearRemodAdd, GarageYrBlt bằng MinMax Scaler**
# Scale dữ liệu các cột này về khoảng [0, 1] và giữ lại quan hệ thứ tự. Quá trình scale này được thực hiện theo công thức sau:
# $$
# X_{\text{scaled}} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}
# $$
# Trong đó:
# - $X$ là giá trị gốc của một điểm dữ liệu.
# - $X_{\text{min}}$ là giá trị nhỏ nhất (minimum) của đặc trưng đó trong tập huấn luyện.
# - $X_{\text{max}}$ là giá trị lớn nhất (maximum) của đặc trưng đó trong tập huấn luyện.

# In[ ]:


time_cols = ["YearBuilt", "YearRemodAdd", "GarageYrBlt"]
scaler = MinMaxScaler()

train_data[time_cols] = scaler.fit_transform(train_data[time_cols])
# Transform test dùng scaler của train
test_data[time_cols] = scaler.transform(test_data[time_cols])


# ### 3.7. Kỹ thuật đặc trưng

# ##### **1. Thêm các cột tổng diện tích sử dụng (TotalSF), tổng số phòng (TotalRooms), tổng số phòng tắm (TotalBath)**

# In[ ]:


for df in [train_data, test_data]:
	df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
	df['TotalRooms'] = df['TotRmsAbvGrd'] + df['FullBath'] + df['HalfBath']
	df["TotalBath"] = df["FullBath"] + (0.5 * df["HalfBath"]) + df["BsmtFullBath"] + (0.5 * df["BsmtHalfBath"])


# #### **2. Tạo cột nhị phân (0, 1) cho các cột phân bố liên tục có nhiều dữ liệu 0.**

# In[ ]:


processed_cols = ["MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "2ndFlrSF", "TotalBsmtSF", "LowQualFinSF", "WoodDeckSF", 
                  "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal", "GarageArea"]

for col in processed_cols:
    # Tạo tên cột nhị phân mới
    new_col = col + "Binary"

    # Áp dụng: 0 -> 0, >0 -> 1
    train_data[new_col] = train_data[col].apply(lambda x: 0 if x == 0 else 1)
    test_data[new_col]  = test_data[col].apply(lambda x: 0 if x == 0 else 1)


# #### **3. Kiểm tra dữ liệu tập train và test.**

# In[ ]:


train_data.info()


# In[ ]:


test_data.info()


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# ## 4. Xuất dữ liệu dọn dẹp của train và test thành file .pkl

# In[ ]:


train_data.to_pickle("./data/train_processed.pkl")  # lưu pkl
test_data.to_pickle("./data/test_processed.pkl")  # lưu pkl

train_y = pd.DataFrame(train_y, columns=["SalePrice"])
train_y.to_pickle("./data/train_y_processed.pkl") # lưu pkl


# ## 5. Xuất các file backup
# Lưu thành file preprocessing_backup.py và file preprocessing_backup.ipynb trong thư mục backup.

# In[ ]:


get_ipython().system('jupyter nbconvert --to script preprocessing.ipynb --output ./backup/preprocessing_backup')


# In[ ]:


get_ipython().system('copy preprocessing.ipynb .\\backup\\preprocessing_backup.ipynb')


# # **Kết thúc**
