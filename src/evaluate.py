import argparse
from util import load_file_list
from fastai.vision import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-d', '--dir', type=str)
    args = parser.parse_args()

    img_dir = Path(args.dir)
    data_df = load_file_list(img_dir)

    data_source = (ImageList.from_df(df=data_df, path=img_dir, cols='images')
                   .split_by_idxs(train_idx = (data_df[data_df['dataset'] == 'train'].index), valid_idx = (data_df[data_df['dataset'] == 'test'].index))
                   .label_from_df(cols='category'))
    tmfs = get_transforms()
    data = data_source.transform(None, size=224).databunch(bs=32, num_workers=4).normalize(imagenet_stats)

    learn = load_learner(args.dir)
    learn.data = data

    interp = ClassificationInterpretation.from_learner(learn, ds_type=DatasetType.Valid)
    fig = interp.plot_confusion_matrix(return_fig=True)
    fig.savefig('./tmp/confusion.png', dpi=500)
    plt.show()