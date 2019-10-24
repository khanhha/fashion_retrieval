import pandas as pd
from fastai.vision import *
from fastai.callbacks import *
from fastai.metrics import accuracy, top_k_accuracy

def load_file_list(dir):
    with open(f'{dir}/Anno/list_category_cloth.txt') as file:
        categories = []
        for i, line in enumerate(file.readlines()):
            if i> 1:
                categories.append(line.split(' ')[0])

    with open(f'{dir}/Anno/list_category_img.txt') as file:
        images = []
        for i, line in enumerate(file.readlines()):
            if i > 1:
                images.append([word.strip() for word in line.split(' ') if len(word) > 0])

    with open(f'{dir}/Eval/list_eval_partition.txt', 'r') as file:
        images_partition = []
        for i, line in enumerate(file.readlines()):
            if i> 1:
                images_partition.append([word.strip() for word in line.split(' ') if len(word) > 0])

    data_df  = pd.DataFrame(images, columns=['images', 'category_label'])
    data_df['category_label'] = data_df['category_label'].astype(int)
    partition_df  = pd.DataFrame(images_partition, columns=['images', 'dataset'])
    data_df = data_df.merge(partition_df, on='images')

    #print(data_df.head(10))
    #print(data_df['dataset'].value_counts())

    data_df['category'] = data_df['category_label'].apply(lambda x: categories[int(x) - 1])
    #print(data_df['category_label'].nunique())
    return data_df