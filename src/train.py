import pandas as pd
import argparse
from pathlib import Path
from fastai.vision import *
from fastai.callbacks import *
from fastai.metrics import accuracy, top_k_accuracy
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
from util import load_file_list

def train(args):
    img_dir = Path(args.dir)
    data_df = load_file_list(img_dir)
    #print(data_df['dataset'].head(10))
    #data_df = data_df[:1000]

    data_source = (ImageList.from_df(df=data_df, path=img_dir, cols='images')
                   .split_by_idxs((data_df[data_df['dataset'] == 'train'].index), (data_df[data_df['dataset'] == 'val'].index))
                   .label_from_df(cols='category'))

    tmfs = get_transforms()

    data = data_source.transform(tmfs, size=224).databunch(bs=32, num_workers=4).normalize(imagenet_stats)

    test_data = ImageList.from_df(df=data_df[data_df['dataset'] == 'test'], path=img_dir, cols='images')
    data.add_test(test_data)

    top_3_accuracy = partial(top_k_accuracy, k=3)
    top_5_accuracy = partial(top_k_accuracy, k=5)
    learner = cnn_learner(data, models.resnet18, metrics=[accuracy])
    learner.model = learner.model.cuda()
    learner.lr_find()

    learner.fit_one_cycle(args.epoch, max_lr=1e-03, callbacks=[SaveModelCallback(learner, every='improvement', monitor='accuracy', name='best_model')])
    #learner.fit_one_cycle(args.epoch, max_lr=1e-03)
    learner.export()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-d', '--dir', type=str)
    parser.add_argument('-e', '--epoch', type=int, default=10)
    args = parser.parse_args()
    print(args)
    train(args)