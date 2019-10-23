import torchvision
import torch
from fastai.vision import *
from fastai.callbacks import *
from util import load_file_list
import argparse
import time
from scipy.spatial.distance import cosine
import annoy
from annoy import AnnoyIndex
from sklearn.externals import joblib

class Hook():
    "Create a hook on `m` with `hook_func`."
    def __init__(self, m: nn.Module, hook_func: HookFunc, is_forward: bool = True, detach: bool = True):
        self.hook_func, self.detach, self.stored = hook_func, detach, None
        f = m.register_forward_hook if is_forward else m.register_backward_hook
        self.hook = f(self.hook_fn)
        self.removed = False

    def hook_fn(self, module: nn.Module, input: Tensors, output: Tensors):
        "Applies `hook_func` to `module`, `input`, `output`."
        if self.detach:
            input = (o.detach() for o in input) if is_listy(input) else input.detach()
            output = (o.detach() for o in output) if is_listy(output) else output.detach()
        self.stored = self.hook_func(module, input, output)

    def remove(self):
        "Remove the hook from the model."
        if not self.removed:
            self.hook.remove()
            self.removed = True

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()

def get_output(module, input_value, output):
    return output.flatten(1)

def get_input(module, input_value, output):
    return list(input_value)[0]

def get_named_module_from_model(model, name):
    for n, m in model.named_modules():
        if n == name:
            return m
    return None

def get_similar_images(img_repr_df, img_index, n=10):
    start = time.time()
    base_img_id, base_vector, base_label  = img_repr_df.iloc[img_index, [0, 1, 2]]
    cosine_similarity = 1 - img_repr_df['img_repr'].apply(lambda x: cosine(x, base_vector))
    similar_img_ids = np.argsort(cosine_similarity)[-11:-1][::-1]
    end = time.time()
    #print(f'{end - start} secs')
    return base_img_id, base_label, img_repr_df.iloc[similar_img_ids]

def show_similar_images(learner, similar_images_df):
    images = [open_image(img_id) for img_id in similar_images_df['img_id']]
    categories = [learner.data.train_ds.y.reconstruct(y) for y in similar_images_df['label_id']]
    return learner.data.show_xys(images, categories)

def get_similar_images_annoy(t, img_repr_df, img_index):
    start = time.time()
    base_img_id, base_vector, base_label  = img_repr_df.iloc[img_index, [0, 1, 2]]
    similar_img_ids = t.get_nns_by_item(img_index, 13)
    end = time.time()
    #print(f'{(end - start) * 1000} ms')
    return base_img_id, base_label, img_repr_df.iloc[similar_img_ids[1:]]

from tqdm import tqdm
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-d', '--dir', type=str)
    args = parser.parse_args()

    img_dir = Path(args.dir)
    data_df = load_file_list(img_dir)
    #data_df = data_df[:1000]

    learn = load_learner(args.dir)
    model = learn.model
    linear_output_layer = get_named_module_from_model(model, '1.4')

    train_valid_images_df = data_df[data_df['dataset'] != 'test']
    inference_data_source = (ImageList.from_df(df=train_valid_images_df, path=img_dir, cols ='images')
                             .split_none()
                             .label_from_df(cols='category'))

    tmfs = get_transforms()
    bs = 32
    inference_data = inference_data_source.transform(tmfs, size=224).databunch(bs=bs).normalize(imagenet_stats)
    inference_dataloader = inference_data.train_dl.new(shuffle=False)

    img_repr_map = {}
    os.makedirs('./tmp', exist_ok=True)
    tmppath = Path('./tmp/img_reprs.pkl')
    if not tmppath.exists():
        with Hook(linear_output_layer, get_output, True, True) as hook:
            start = time.time()
            for i, (xb, yb) in tqdm(enumerate(inference_dataloader), total=len(data_df)//bs, desc='feature extraction'):
                bs = xb.shape[0]
                img_ids = inference_dataloader.items[i * bs: (i + 1) * bs]
                result = model.eval()(xb)
                img_reprs = hook.stored.cpu().numpy()
                img_reprs = img_reprs.reshape(bs, -1)
                for img_id, img_repr in zip(img_ids, img_reprs):
                    img_repr_map[img_id] = img_repr
                if (len(img_repr_map) % 12800 == 0):
                    end = time.time()
                    print(f'{end - start} secs for 12800 images')
                    start = end
        joblib.dump(img_repr_map, filename=tmppath)
    else:
        img_repr_map = joblib.load(tmppath)

    img_repr_df = pd.DataFrame(img_repr_map.items(), columns=['img_id', 'img_repr'])
    print(img_repr_df.head(10))

    img_repr_df['label'] = [inference_data.classes[x] for x in inference_data.train_ds.y.items[0:img_repr_df.shape[0]]]
    img_repr_df['label_id'] = inference_data.train_ds.y.items[0:img_repr_df.shape[0]]

    f = len(img_repr_df['img_repr'][0])
    t = AnnoyIndex(f, metric='euclidean')

    for i, vector in tqdm(enumerate(img_repr_df['img_repr']), total = len(data_df), desc="build knn"):
        t.add_item(i, vector)
    _ = t.build(inference_data.c)

    # 230000, 130000, 190000
    N = int(len(data_df)/2)
    samples = 100
    for idx, i in tqdm(enumerate(np.random.randint(0, N, samples)), total=samples, desc='compute similarity'):
        print(i)
        base_image, base_label, similar_images_df = get_similar_images_annoy(t, img_repr_df, i)
        n_imgs = 1 + len(similar_images_df)
        n_col = int(np.sqrt(n_imgs+1))
        plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')
        fig, axes = plt.subplots(nrows=n_col, ncols=n_col)
        imgs = [base_image] + [path for path in similar_images_df['img_id']]
        for i, ax in enumerate(axes.flatten()):
            img = plt.imread(imgs[i])
            ax.imshow(img)
        dir = Path('/media/F/projects/datasets/DeepFashion/debug/sim_imgs/')
        plt.savefig(dir/Path(base_image).name, dpi=200)
        plt.clf()