import opendatasets as od
import os

# Download FER2013 từ Kaggle
dataset_url = 'https://www.kaggle.com/datasets/msambare/fer2013'

print("Đang tải dataset FER2013...")
print("Lần đầu tiên sẽ yêu cầu Kaggle username và API key")
print("Lấy tại: https://www.kaggle.com/settings/account → Create New API Token\n")

od.download(dataset_url)

print("\n✅ Đã tải xong dataset!")
print("📁 Vị trí: ./fer2013/")