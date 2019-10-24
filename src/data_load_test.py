from fastai.vision import *

path_data = untar_data(URLs.MNIST_TINY)
il_data = ItemList.from_folder(path_data)
il_data= il_data.split_by_folder(train='train', valid='valid')
il_data = il_data.label_from_folder()
print(il_data.train.to_df().head(5))