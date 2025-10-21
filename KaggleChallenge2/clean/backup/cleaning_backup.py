#!/usr/bin/env python
# coding: utf-8

# # **Kaggle Challenge 2: Đề tài dự báo giá nhà.**
# # **PHẦN 1: TÌM HIỂU CƠ SỞ VÀ DỌN DẸP DỮ LIỆU**

# ## 1. Định nghĩa vấn đề
# + **Mô tả**:
#    - Dự báo giá nhà dựa trên 79 tính chất của dataset nhà ở tại Ames, Iowa.
#    - Dữ liệu bao gồm 4 file:
#       * train.csv: Dataset cho việc train.
#       * test.csv: Dataset cho việc test.
#       * sample_submission.csv: Mẫu submission.
#       * data_description.txt: Mô tả dữ liệu của từng cột.
# 
# + **Mục tiêu**:
#    - Xem thông tin chung, kiểm tra tính toàn vẹn và xử lý các giá trị thiếu cho dữ liệu.
# 
# + **Dữ liệu đầu vào**:
#    - MSSubClass: Loại nhà.
#    - MSZoning: Phân loại quy hoạch.
#    - LotFrontage: Chiều dài lô đất (đơn vị feet).
#    - LotArea: Diện tích lô đất (đơn vị $ft^{2}$)
#    - Street: Loại đường mặt tiền.
#    - Alley: Loại lối đi phụ sau nhà.
#    - LotShape: Hình dạng tổng thể của khu đất.
#    - LandContour: Địa hình của khu đất.
#    - Utilities: Các loại tiện nghi có sẵn.
#    - LotConfig: Vị trí và cấu hình của khu đất so với các lô khác.
#    - LandSlope: Đọ nghiêng tổng thể khu đất.
#    - Neighborhood: Loại khu dân cư.
#    - Condition1: Điều kiện 1 - Vị trí xung quanh căn nhà (ví dụ đường lớn, đường sắt, công viên).
#    - Condition2: Điều kiện 2 - Vị trí xung quanh căn nhà.
#    - BldgType: Loại hình nhà ở.
#    - HouseStyle: Kiểu thiết kế nhà.
#    - OverallQual: Đánh giá chất lượng vật liệu và hoàn thiện của nhà (1 - 10).
#    - OverallCond: Đánh giá tình trạng của ngôi nhà (0 - 10).
#    - YearBuilt: Năm xây nhà.
#    - YearRemodAdd: Năm tu bổ (trùng với năm xây nhà nếu không tu bổ).
#    - RoofStyle: Loại mái nhà.
#    - RoofMatl: Vật liệu mái nhà.
#    - Exterior1st: Vật liệu bao phủ bên ngoài tường nhà.
#    - Exterior2nd: Vật liệu phụ bao phủ bên ngoài tường nhà (nếu có hơn 1 loại vật liệu).
#    - MasVnrType: Vật liệu trang trí ốp bên ngoài tường nhà.
#    - MasVnrArea: Tổng diện tích bề mặt vật liệu trang trí ốp bên ngoài tường nhà (đơn vị $ft^{2}$).
#    - ExterQual: Đánh giá chất lượng vật liệu bao phủ bên ngoài tường nhà.
#    - ExterCond: Đánh giá tình trạng vật liệu bao phủ bên ngoài tường nhà.
#    - Foundation: Loại móng nhà.
#    - BsmtQual: Đánh giá độ cao tầng hầm.
#    - BsmtCond: Đánh giá tình trạng chung của tầng hầm.
#    - BsmtExposure: Mức độ tiếp xúc ánh sáng / tường hở của tầng hầm.
#    - BsmtFinType1: Chất lượng khu vực hoàn thiện tầng hầm (loại 1).
#    - BsmtFinSF1: Diện tích hoàn thiện tầng hầm (loại 1).
#    - BsmtFinType2: Chất lượng khu vực hoàn thiện tầng hầm (loại 2).
#    - BsmtFinSF2: Diện tích hoàn thiện tầng hầm (loại 2).
#    - Heating: Loại sưởi ấm.
#    - HeatingQC: Chất lượng sưởi ấm.
#    - CentralAir: Có điều hóa không khí trung tâm.
#    - Electrical: Loại hệ thống điện.
#    - 1stFlrSF: Diện tích sàn tầng 1 (đơn vị $ft^{2}$)
#    - 2ndFlrSF: Diện tích sàn tầng 2 (đơn vị $ft^{2}$)
#    - LowQualFinSF: Tổng diện tích hoàn thiện chất lượng thấp (tất cả các tầng) (đơn vị $ft^{2}$)
#    - GrLivArea: Tổng diện tích đạt chỉ tiêu sinh hoạt (đơn vị $ft^{2}$)
#    - BsmtFullBath: Số nhà tắm đầy đủ(tầng hầm).
#    - BsmtHalfBath: Số nhà tắm nhỏ(tầng hầm).
#    - Bedroom: Số phòng ngủ (không kể phòng ngủ tầng hầm).
#    - Kitchen: Số nhà bếp.
#    - KitchenQual: Chất lượng nhà bếp.
#    - TotRmsAbvGrd: Tổng số phòng (không kể nhà tắm).
#    - Functional: Đánh giá mức độ sử dụng ngôi nhà.
#    - Fireplaces: Số lò sưởi.
#    - FireplaceQu: Chất lượng lò sưởi.
#    - GarageType: Loại gara.
#    - GarageYrBlt: Năm xây dựng gara.
#    - GarageFinish: Đánh giá độ hoàn thiện gara.
#    - GarageCars: Số lượng ô tô gara có thể chứa.
#    - GarageArea: Diện tích gara (đơn vị $ft^{2}$).
#    - GarageQual: Chất lượng gara.
# 	- GarageCond: Tình trạng gara.
#    - PavedDrive: Đường đi ra vào trong nhà cho ô tô.
#    - WoodDeckSF: Diện tích sàn gỗ ngoài trời (đơn vị $ft^{2}$).
#    - OpenPorchSF: Diện tích hiên ngoài trời (đơn vị $ft^{2}$).
#    - EnclosedPorch: Diện tích hiên có tường (đơn vị $ft^{2}$).
#    - 3SsnPorch: Diện tích hiên 3 mùa (đơn vị $ft^{2}$).
#    - ScreenPorch: Diện tích hiên có lưới (đơn vị $ft^{2}$).
#    - PoolArea: Diện tích hồ bơi (đơn vị $ft^{2}$).
#    - PoolQC: Chất lượng hồ bơi.
#    - Fence: Chất lượng hàng rào quanh nhà.
#    - MiscFeature: Các đặc trưng khác không được đề cập.
#    - MiscVal: Giá trị của đặc trưng.
#    - MoSold: Tháng bán.
#    - YrSold: Năm bán.
#    - SaleType: Thể loại bán.
#    - SaleCondition: Điều kiện bán.
#    
# + **Kết quả**: 
#    - SalePrice: Giá bán của căn nhà.
# 
# + **Một số thông tin khác về dữ liệu (theo file mô tả)**:
#    - Một số cột như Alley, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, FireplaceQu, GarageType, GarageQual, GarageFinish, GarageCond, PoolQC, Fence, MiscFeature chứa giá trị NA và được hiểu là "không có", không phải thiếu.
# 
#    - Cột MasVnrType chứa cả None và NA (None được hiểu là không có, NA là bị thiếu).

# ## 2. Chuẩn bị vấn đề.

# ### 2.1. Import các thư viện cần thiết.

# In[856]:


# Load libraries
from IPython import display
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', 50) # Số cột tối đa có thể hiển thị


# ### 2.2. Lấy dữ liệu từ dataset.

# In[857]:


train_data = pd.read_csv("../data/train.csv", keep_default_na=False,# chỉ mã hóa NA thành null, còn None giữ nguyên
    na_values=["NA"] )
test_data = pd.read_csv("../data/test.csv", keep_default_na=False,
    na_values=["NA"])


# ### 2.3. Kiểm tra dữ liệu dataset.

# In[858]:


train_data.head()


# In[859]:


test_data.head()


# ## 3. Xem thông tin cơ bản của dữ liệu

# ### 3.1. Thống kê, mô tả

# ##### **1) Hiển thị một số thông tin về dữ liệu**
# + Số dòng, số cột của dữ liệu
# + Kiểu dữ liệu của từng cột
# + Thông tin chung về dữ liệu

# In[860]:


train_data.info()


# In[861]:


test_data.info()


# In[862]:


train_data.dtypes.value_counts()


# In[863]:


test_data.dtypes.value_counts()


# In[864]:


train_data.describe()


# In[865]:


test_data.describe()


# **Nhận xét**:
# + Dữ liệu có 80 tính chất để phân lớp.
# + Có 37 tính chất ở dạng số (float64, int64), 43 tính chất ở dạng phi số (object).
# + Tổng số dòng dữ liệu ở tập train là 1460 dòng, tập test là 1459 dòng.
# + Dữ liệu để phân lớp ở cột SalePrice.
# + Giá trị max của GarageYrBlt ở tập test bất hợp lý (2207), do năm xây gara không thể lớn hơn năm hiện tại.
# 

# ##### **2) Kiểm tra tính toàn vẹn của dữ liệu**
# + Dữ liệu có bị trùng lặp không?
# + Dữ liệu có tồn tại giá trị Null không?

# In[866]:


train_data.isnull().sum().sort_values(ascending=False).head(40)


# In[867]:


test_data.isnull().sum().sort_values(ascending=False).head(40)


# In[868]:


# In ra những cột bị null ở test nhưng không bị null ở train.
print("Các cột bị null ở test nhưng không bị null ở train: ")
cols_null_in_test_not_in_train = [
    col for col in test_data.columns
    if train_data.isnull().sum()[col] == 0 and test_data.isnull().sum()[col] > 0
]

for col in cols_null_in_test_not_in_train:
    print(f"{col}: {test_data[col].isnull().sum()} giá trị null trong test nhưng 0 ở train.")

print(f"Danh sách có {len(cols_null_in_test_not_in_train)} dòng.")


# In[869]:


# In ra những cột bị null ở train nhưng không bị null ở test.
print("Các cột bị null ở train nhưng không bị null ở test: ")
train_data_without_sale_price = train_data.drop("SalePrice", axis=1)
cols_null_in_train_not_in_test = [
    col for col in train_data_without_sale_price.columns
    if train_data.isnull().sum()[col] > 0 and test_data.isnull().sum()[col] == 0
]

for col in cols_null_in_train_not_in_test:
    print(f"{col}: {train_data[col].isnull().sum()} giá trị null trong train nhưng 0 ở test")


# In[870]:


print("Số cột null ở train: ", (train_data.isnull().sum() > 0).sum()) # Đếm số cột chứa giá trị null


# In[871]:


print("Số cột null ở test: ", (test_data.isnull().sum() > 0).sum())


# In[872]:


print("Số dòng trùng lặp trong train: ", train_data.duplicated().sum())


# In[873]:


print("Số dòng trùng lặp trong test: " , test_data.duplicated().sum())


# **Nhận xét**:
# + Dữ liệu tập train và test lần lượt có 19 và 33 cột chứa giá trị null. 
# + Các cột PoolQC, MiscFeature, Alley, Fence có số lượng giá trị null rất lớn (trên 1000 dòng).
# + Các cột FireplaceQu, LotFrontage có số lượng giá trị null khá lớn (từ 227 đến 730 dòng).
# + Các cột GarageQual, GarageFinish, GarageType, GarageYrBlt, GarageCond, BsmtFinType2, BsmtExposure, BsmtCond, BsmtQual, BsmtFinType1 có số lượng giá trị null tương đối ít (37 đến 81 dòng).
# + Các cột MSZoning, Utilities, Exterior1st, Exterior2nd, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath, BsmtHalfBath, KitchenQual, Functional, GarageCars, GarageArea, Electrical, MasVnrArea, MasVnrType, SaleType có ít giá trị null (dưới 16 dòng).
# + Một số cột không chứa giá trị null ở train nhưng chứa giá trị null ở test và ngược lại.
# + Một số cột bị mã hóa sai (do mã hóa NA thành null).
# + Dữ liệu tập train và test không có dòng trùng lặp.
# 
# **Cần xử lý các giá trị bị null trong cả 2 tập train và test.**

# #### **3) Tần số xuất hiện (Distribution) trên dữ liệu phân lớp (SalePrice)**

# In[874]:


plt.figure(figsize=(10,6))
plt.hist(train_data["SalePrice"], bins=40, color="skyblue", edgecolor="black")
plt.title("Phân phối biến SalePrice", fontsize=16)
plt.xlabel("SalePrice")
plt.ylabel("Tần suất")
plt.savefig("dist_hist_sale_price.png")
plt.show()


# In[875]:


plt.figure(figsize=(10,6))
sns.boxplot(y=train_data["SalePrice"], color="skyblue", fliersize=3, linewidth=1)
plt.title("Phân phối biến SalePrice", fontsize=16)
plt.ylabel("SalePrice")
plt.xlabel("Tần suất")
plt.savefig("dist_box_sale_price.png")
plt.show()


# **Nhận xét**:
# + Dữ liệu tập trung nhiều ở mức 100000 đến 200000.
# + Dữ liệu có xu hướng bị lệch phải.
# + Dữ liệu có nhiều giá trị ngoại lai rất lớn.

# ## 4. Dọn dẹp dữ liệu
# 

# ### 4.1. Xử lý các cột thiếu giá trị.

# ##### **1) Xử lý các cột có số lượng giá trị null rất lớn (trên 1000 dòng)**
# Theo dữ liệu mô tả, giá trị NA trong các cột PoolQC, MiscFeature, Alley, Fence được hiểu là "không có" nhưng bị mã hóa thành null, do đó cần chuyển về  thành dạng chuỗi.

# In[876]:


cols_to_modify = ["PoolQC", "MiscFeature", "Alley", "Fence"]
train_data[cols_to_modify] = train_data[cols_to_modify].fillna("None")
test_data[cols_to_modify] = test_data[cols_to_modify].fillna("None")


# ##### **2) Xử lý các cột có số lượng giá trị null ít (dưới 16 dòng)**
# Điền vào giá trị thiếu bằng median của cột tương ứng đối với dữ liệu số, bằng giá trị phổ biến nhất đối với dữ liệu phi số. Thay đổi 

# In[877]:


replaced_cols = [
    'MSZoning', 
    'Utilities', 
    'Exterior1st', 
    'Exterior2nd', 
    'BsmtFinSF1', 
    'BsmtFinSF2', 
    'BsmtUnfSF', 
    'TotalBsmtSF', 
    'BsmtFullBath', 
    'BsmtHalfBath', 
    'KitchenQual', 
    'Functional', 
    'GarageCars', 
    'GarageArea', 
    'Electrical', 
    'MasVnrArea',
    'MasVnrType',
    'SaleType'
]

print("Train data: ")
print(train_data[replaced_cols].dtypes)


# In[878]:


replaced_numerical_cols = train_data[replaced_cols].select_dtypes(include=["int64", "float64"]).columns
replaced_object_cols = train_data[replaced_cols].select_dtypes(include=["object"]).columns

print("Cột số: ", replaced_numerical_cols)
print("Cột phi số: ", replaced_object_cols)


# **Nhận xét**:
# + Các cột phi số bị thiếu không nằm trong các cột dùng NA cho "không có" (Alley, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, FireplaceQu, GarageType, GarageQual, GarageFinish, GarageCond, PoolQC, Fence, MiscFeature), như vậy các cột này có khả năng bị thiếu.

# In[879]:


# Điền vào giá trị null là median của cột tương ứng.
train_data[replaced_numerical_cols] = train_data[replaced_numerical_cols].fillna(train_data[replaced_numerical_cols].median())
test_data[replaced_numerical_cols] = test_data[replaced_numerical_cols].fillna(test_data[replaced_numerical_cols].median())

# Điền vào giá trị null là giá trị có tần số lớn nhất.
for col in replaced_object_cols:
    train_data[col] = train_data[col].fillna(train_data[col].mode()[0])
    test_data[col] = test_data[col].fillna(test_data[col].mode()[0])


# In[880]:


# Kiểm tra số dòng null theo cột
train_data.isnull().sum().sort_values(ascending=False).head(20)


# In[881]:


# Kiểm tra số dòng null theo cột
test_data.isnull().sum().sort_values(ascending=False).head(20)


# ##### **3) Xử lý các cột có số lượng giá trị null tương đối ít (37 - 81 dòng)** 
# Điền vào giá trị giá trị phổ biến nhất (cho giá trị phi số) của cột đó theo phân loại cột Neighborhood. Dùng cột Neighborhood để phân loại do cột này có nhiều loại giá trị nhất (25 loại giá trị) và không chứa giá trị null trên cả train và test, giúp cho giá trị được điền vào ít bị trùng lặp. Cột GarageYrBlt là cột thứ tự thời gian nên phải xử lý bằng cách điền vào giá trị phổ biến nhất theo phân loại cột Neighborhood, xử lý cột GarageYrBlt trên test có giá trị max bất hợp lý (2207).

# In[882]:


print("Phân phối giá trị của cột Neighborhood trong train: ", set(train_data["Neighborhood"].unique()))
print("Phân phối giá trị của cột Neighborhood trong test: ", set(test_data["Neighborhood"].unique()))


# In[883]:


print(f"Phạm vi của cột GarageYrBlt: {test_data["GarageYrBlt"].min()} {test_data["GarageYrBlt"].max()}")


# In[884]:


test_data.loc[test_data["GarageYrBlt"] > 2015, "GarageYrBlt"] = np.nan # Chuyển những giá trị năm > 2015 thành null.


# In[885]:


print(f"Phạm vi của cột GarageYrBlt: {test_data["GarageYrBlt"].min()} {test_data["GarageYrBlt"].max()}") # Kiểm tra


# In[886]:


replaced_cols = [
    "GarageQual",
    "GarageFinish",    
    "GarageType",      
    "GarageCond",      
    "BsmtFinType2",      
    "BsmtExposure",     
    "BsmtCond",       
    "BsmtQual",         
    "BsmtFinType1"
]
print("Train data: ")
print(train_data[replaced_cols].dtypes)


# **Nhận xét**:
# + Các cột phi số bị thiếu nằm trong các cột dùng NA cho "không có" (Alley, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, FireplaceQu, GarageType, GarageQual, GarageFinish, GarageCond, PoolQC, Fence, MiscFeature), như vậy cần mã hóa các cột phi số này thành dạng chuỗi.

# In[887]:


# Điền giá trị năm phổ biến nhất vào cột GarageYrBlt phân phối theo Neighborhood
train_data["GarageYrBlt"] = (
    train_data.groupby('Neighborhood')["GarageYrBlt"]
    .transform(lambda x: x.fillna(x.mode().iloc[0]))
)

test_data["GarageYrBlt"] = (
    test_data.groupby('Neighborhood')["GarageYrBlt"]
    .transform(lambda x: x.fillna(x.mode().iloc[0]))
)

# Điền vào None biểu diễn cho không có.
train_data[replaced_cols] = train_data[replaced_cols].fillna("None")
test_data[replaced_cols] = test_data[replaced_cols].fillna("None")


# In[888]:


# Kiểm tra số dòng null theo cột
train_data.isnull().sum().sort_values(ascending=False).head(20)


# In[889]:


# Kiểm tra số dòng null theo cột
test_data.isnull().sum().sort_values(ascending=False).head(20)


# ##### **4) Xử lý các cột có số lượng giá trị null khá lớn (227 - 730 dòng)**

# In[890]:


replaced_cols = [
    "FireplaceQu",     
    "LotFrontage"    
]

print("Train data: ")
print(train_data[replaced_cols].dtypes)


# In[891]:


replaced_numerical_cols = train_data[replaced_cols].select_dtypes(include=["int64", "float64"]).columns
replaced_object_cols = train_data[replaced_cols].select_dtypes(include=["object"]).columns

print("Cột số: ", replaced_numerical_cols)
print("Cột phi số: ", replaced_object_cols)


# **Nhận xét**:
# + Cột phi số (FireplaceQu) bị thiếu nằm trong các cột dùng NA cho "không có" (Alley, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, FireplaceQu, GarageType, GarageQual, GarageFinish, GarageCond, PoolQC, Fence, MiscFeature), như vậy cần mã hóa các cột phi số này thành dạng chuỗi.
# + Cột số (LotFrontage) bị thiếu khoảng hơn 200 dòng, có thể điền vào giá trị trung bình của LotFrontage theo cột Neighborhood.

# In[892]:


# Điền vào giá trị null là median của cột tương ứng phân phối theo Neighborhood.
train_data[replaced_numerical_cols] = (
    train_data.groupby('Neighborhood')[replaced_numerical_cols]
    .transform(lambda x: x.fillna(x.median()))
)

test_data[replaced_numerical_cols] = (
    test_data.groupby('Neighborhood')[replaced_numerical_cols]
    .transform(lambda x: x.fillna(x.median()))
)

# Điền None vào các giá trị bị thiếu.
train_data[replaced_object_cols] = train_data[replaced_object_cols].fillna("None")
test_data[replaced_object_cols] = test_data[replaced_object_cols].fillna("None")


# In[893]:


# Kiểm tra số dòng null theo cột
train_data.isnull().sum().sort_values(ascending=False).head(20)


# In[894]:


# Kiểm tra số dòng null theo cột
test_data.isnull().sum().sort_values(ascending=False).head(20)


# ##### **5) Kiểm tra số lượng cột chứa giá trị null sau khi xử lý**

# In[895]:


print("Số cột null trong train: ", train_data.isnull().sum().gt(0).sum())
print("Số cột null trong test: ", test_data.isnull().sum().gt(0).sum())


# **Nhận xét:**
# Tất cả các cột trên train và test chứa giá trị null đã được xử lý.

# ### 4.2. Chuyển các cột có kiểu dữ liệu float64 về int64.
# Dữ liệu số ban đầu chỉ bao gồm các số nguyên, nhưng do một số cột chứa giá trị NA nên pandas mặc định chuyển kiểu dữ liệu cột về dạng float64.

# In[896]:


train_data.dtypes.value_counts()


# In[897]:


test_data.dtypes.value_counts()


# In[898]:


# Chuyển từ float64 => int64
train_data[train_data.select_dtypes('float64').columns] = train_data.select_dtypes('float64').round().astype('int64')
test_data[test_data.select_dtypes('float64').columns] = test_data.select_dtypes('float64').round().astype('int64')


# In[899]:


train_data.dtypes.value_counts()


# In[900]:


test_data.dtypes.value_counts()


# **Nhận xét:**
# Không còn cột float64, tất cả được chuyển về kiểu int64.

# ### 4.3. Xóa cột Id
# Cột này chỉ đánh số thứ tự cho việc định danh, không có ý nghĩa cho việc phân tích.

# In[901]:


print("Số cột trong train trước khi xóa: ", len(train_data.columns))
print("Số cột trong test trước khi xóa: ", len(test_data.columns))


# In[902]:


train_data = train_data.drop(columns=["Id"])
test_data = test_data.drop(columns=["Id"])


# In[903]:


print("Số cột trong train sau khi xóa: ", len(train_data.columns))
print("Số cột trong test sau khi xóa: ", len(test_data.columns))


# ## 5. Xuất dữ liệu dọn dẹp của train và test thành file .csv và .pkl

# In[904]:


train_data.to_csv("./data/train_clean.csv", index=False)       # lưu csv
train_data.to_pickle("./data/train_clean.pkl")  # lưu pkl

test_data.to_csv("./data/test_clean.csv", index=False)       # lưu csv
test_data.to_pickle("./data/test_clean.pkl")  # lưu pkl


# ## 6. Xuất các file backup
# Lưu thành file preprocessing_backup.py và file preprocessing_backup.ipynb trong thư mục backup.

# In[ ]:


get_ipython().system('jupyter nbconvert --to script cleaning.ipynb --output ./backup/cleaning_backup')


# In[ ]:


get_ipython().system('copy cleaning.ipynb .\\backup\\cleaning_backup.ipynb')

