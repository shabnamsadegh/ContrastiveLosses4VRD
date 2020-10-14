import pandas as pd
import json
import numpy as np
from tqdm import tqdm
import os

# Preparing data annotations
# In[]:
root_dir = "./data/imaterialist2/"
train_or_test = "train"
add_attributes = False
detection_file = os.path.join(root_dir, train_or_test + "_detections.json")

with open(detection_file, "r+") as f:
    temp_data = json.load(f)
    pd_imaterialist_annotation = pd.DataFrame(temp_data["annotations"])
    pd_imaterialist_annotation.index = pd_imaterialist_annotation["id"].values
    pd_imaterialist_images = pd.DataFrame(temp_data["images"])
    pd_imaterialist_images.index = pd_imaterialist_images["id"].values
    cat_id_to_category = {dic["id"]: dic for dic in temp_data["categories"]}
    del temp_data
    del f


with open(root_dir + "image_name_to_id_dict.json", "r+") as f:
    image_name_to_id_dict = json.load(f)
    id_to_image_name_dict = {image_name_to_id_dict[name]: name for name in image_name_to_id_dict}
    del image_name_to_id_dict
    del f

with open(root_dir + "mask_id_to_encoded_pix.json", "r+") as f:
    mask_id_to_encoded_pix = json.load(f)
    del f

with open(root_dir + "mask_id_to_attributes.json", "r+") as f:
    mask_id_to_attributes = json.load(f)
    del f

# In[]:
print("Done:reading")
pd_imaterialist_annotation["cat_name"] = pd_imaterialist_annotation["category_id"].map(cat_id_to_category).map(
    lambda x: x["name"])
pd_imaterialist_annotation["cat_supercat"] = pd_imaterialist_annotation["category_id"].map(cat_id_to_category).map(
    lambda x: x["supercategory"])

merged = pd.merge(pd_imaterialist_annotation, pd_imaterialist_images, left_on='image_id', right_on='id')
"""pd_imaterialist_annotation["image_shape"] = list(
    merged[["height", "width"]].to_records(index=False))
pd_imaterialist_annotation["image_file_name"] = merged["file_name"]
pd_imaterialist_annotation["encodedpix"] = pd_imaterialist_annotation["id"].map(str).map(mask_id_to_encoded_pix)
"""
merged["image_shape"] = list(
    merged[["height", "width"]].to_records(index=False))
del merged["height"]
del merged["width"]
del merged["id_y"]
merged.rename(columns={"id_x":"id"},inplace=True)
merged["encodedpix"] = merged["id"].map(str).map(mask_id_to_encoded_pix)
merged.index = merged["id"].values
del pd_imaterialist_annotation

# In[ ]:
print("Done: merging data")
from pycocotools import mask as COCOmask


def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')

from PIL import Image
from colorthief import ColorThief
def get_color_of_mask(img_path, mask_rle, shape):
    #note: shape is (height,width)
    pil_img = Image.open(img_path)
    assert pil_img.size == (shape[1],shape[0])

    mask_np = rle_decode(mask_rle,shape)
    mask_img = mask_np.astype('uint8')*255
    mask_img = Image.fromarray(mask_img)
    pil_img.putalpha(mask_img)

    return ColorThief(pil_img).get_color(quality=1)

all_relations = {}

from tqdm import tqdm_pandas #progress bar
tqdm.pandas(tqdm)
def parallelize_dataframe(df, func, n_cores=20):
    df_split = np.array_split(df, n_cores)
    with Pool(n_cores) as p:
        df = pd.concat(p.map(func, df_split))
    return df
from multiprocessing import Pool

def apply_color_detection(part):
    print(' ', end='', flush=True)
    image_path = "./data/imaterialist/train/"
    return pd.Series(part.progress_apply(lambda x: get_color_of_mask(image_path+x["file_name"], x["encodedpix"], x["image_shape"]), axis=1))

colors = parallelize_dataframe(merged, apply_color_detection,n_cores= 20)
mask_id_to_color={int(id):color for id,color in zip(merged["id"].values,colors)}
with open(root_dir + "mask_id_to_color_"+ train_or_test +".json","w+") as f:
    json.dump(mask_id_to_color,f)