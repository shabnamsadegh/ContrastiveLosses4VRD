import pandas as pd
import json
import numpy as np
from tqdm import tqdm
import os

# Preparing data annotations
# In[]:
root_dir = "./data/imaterialist2/"
train_or_test = "train"
add_attributes = True
add_with = True
add_color = False
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

with open(root_dir + "mask_id_to_color.json","r+") as f:
    mask_id_to_color = json.load(f)
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

all_relations = {}


def encode_color(rgb):
    """rgb to 8-color pallete"""
    r = "1" if rgb[0] > 127 else "0"
    g = "1" if rgb[1] > 127 else "0"
    b = "1" if rgb[2] > 127 else "0"

    for i in range(8):
        if r + g + b == format(i, '03b'):
            return i
for image_id, group in tqdm(merged.groupby("image_id")):
    image_file_name = group.iloc[0]["file_name"]
    masks_length = len(group)
    obj_subj_list = []

    if add_with:
        masks_list = group[["encodedpix", "image_shape"]].apply(lambda x: rle_decode(x["encodedpix"], x["image_shape"]),
                                                                axis=1).map(COCOmask.encode)
        #step1. adding relation ships with predicate "with"
        #finding the overlaps in order to detect hierachies
        masks_iou = COCOmask.iou(list(masks_list), list(masks_list), [True] * masks_length)
        contain_indices = np.argwhere(0.9 < masks_iou)

        mask_ids = group["id"].values
        obj_subj_list = [(x, y) for x, y in map(lambda a: group.iloc[a].to_dict('records'), contain_indices) if
                         x["id"] != y["id"]]
        obj2parent_dict = { x["id"]:y for x, y in map(lambda a: group.iloc[a].to_dict('records'), contain_indices) if
                         x["id"] != y["id"]}
        obj_subj_list = [
            {'predicate': 0, 'subject': {"obj_id":y["id"],'category': y["category_id"], 'bbox': y["bbox"], 'mask': y["encodedpix"]},
             'object': {"obj_id":x["id"],'category': x["category_id"], 'bbox': x["bbox"], 'mask': x["encodedpix"]}} for x, y in
            obj_subj_list] # predicate 0 = "with"

    #step2. adding the attributes. object and sobject are the same but the predicate differs
    if add_attributes:
        for id, row in group.iterrows ():
            if len(mask_id_to_attributes[str(id)])!=0:##important we add 1 to att_id because position 0 is for with
                for attribute in [int(attid)+1 for attid in mask_id_to_attributes[str(id)].split(",")]:# id +1 because number 0 belongs to "with"
                    subject_dict = {"obj_id":row["id"],'category': row["category_id"], 'bbox': row["bbox"], 'mask': row["encodedpix"]}
                    if row["id"] in obj2parent_dict:#this object belongs to a parent. object is set as parent
                        parent = obj2parent_dict[row["id"]]
                        object_dict = {"obj_id": parent["id"], 'category': parent["category_id"], 'bbox': parent["bbox"],
                                       'mask': parent["encodedpix"]}
                    else: #relate to itself
                        object_dict = subject_dict
                    obj_subj_list.append({'predicate': attribute, 'subject': subject_dict,'object': object_dict})

    if add_color:
        for id, row in group.iterrows():
            if str(id) in mask_id_to_color:  ##important we add 1 to att_id because position 0 is for with
                obj_subj_list.append({'predicate': encode_color(mask_id_to_color[str(id)]) + 342,
                                          'subject': {"obj_id": row["id"], 'category': row["category_id"],
                                                      'bbox': row["bbox"]},
                                          'object': {"obj_id": row["id"], 'category': row["category_id"],
                                                     'bbox': row["bbox"]}})


    if len(obj_subj_list) != 0:#why? because RelDN code does not work if there is an empty list
        all_relations[image_file_name] = obj_subj_list

with open(root_dir + "rel_annotations_attr_parent_"+train_or_test+".json","w+") as f:
    json.dump(all_relations,f)


print("Done: all data stored")