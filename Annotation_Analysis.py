import os
import json
print(os.listdir("./data/vg"))
root_dir = "./data/vg/"
with open(root_dir + "detections_train.json","r+") as f:
    detection_train = json.load(f)

print(detection_train.keys())#important
print("images:", detection_train["images"][0].keys())
print("annotations:", detection_train["annotations"][0].keys())
print("categories:", detection_train["categories"][0].keys())
print("number of images: ", len(detection_train["images"]))

print("image sample:", detection_train["images"][0])
#file names seems must be numbers
#all are jpg
#bbox: [x,y,width,height]
print("annotation sample:", detection_train["annotations"][1])
print("category sample:", detection_train["categories"][1])

for x in detection_train["categories"]: #VG does not have hierachical category
    if x["name"]!= x["supercategory"]:
        print(x)


with open(root_dir + "rel_annotations_train.json","r+") as f:
    rel_annotations = json.load(f)

print(rel_annotations.keys())#important
print(len(rel_annotations.keys()))
print(rel_annotations['7.jpg'])

with open("./label_descriptions.json","r+") as f:
    imaterialist_data = json.load(f)

with open(root_dir+ "predicates.json","r+") as f:
    predicates = json.load(f)

with open(root_dir + "objects.json","r+") as f:
    objects = json.load(f)



import pandas as pd
from pandas import Series
import os
import json

root_dir = "./data/imaterialist/"
with open(root_dir + "detections_train.json","r+") as f:
    imaterialist_data = json.load(f)

pd_imaterialist_data = pd.DataFrame(imaterialist_data["annotations"])
print(pd_imaterialist_data.columns)
Series.idxmax(pd_imaterialist_data.groupby("image_id").count()["id"])
pd_imaterialist_data.loc[pd_imaterialist_data["image_id"] == 9672]