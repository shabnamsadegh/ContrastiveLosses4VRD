import sys
import os
import csv
import numpy as np
import json
from tqdm import tqdm
csv.field_size_limit(sys.maxsize)
def rle2bbox(rle, shape):
    '''
    from: https://www.kaggle.com/eigrad/convert-rle-to-bounding-box-x0-y0-x1-y1
    rle: run-length encoded image mask, as string
    shape: (height, width) of image on which RLE was produced
    Returns (x0, y0, x1, y1) tuple describing the bounding box of the rle mask

    Note on image vs np.array dimensions:

        np.array implies the `[y, x]` indexing order in terms of image dimensions,
        so the variable on `shape[0]` is `y`, and the variable on the `shape[1]` is `x`,
        hence the result would be correct (x0,y0,x1,y1) in terms of image dimensions
        for RLE-encoded indices of np.array (which are produced by widely used kernels
        and are used in most kaggle competitions datasets)
    '''

    a = np.fromiter(rle.split(), dtype=np.uint)
    a = a.reshape((-1, 2))  # an array of (start, length) pairs
    a[:, 0] -= 1  # `start` is 1-indexed

    y0 = a[:, 0] % shape[0]
    y1 = y0 + a[:, 1]
    if np.any(y1 > shape[0]):
        # got `y` overrun, meaning that there are a pixels in mask on 0 and shape[0] position
        y0 = 0
        y1 = shape[0]
    else:
        y0 = np.min(y0)
        y1 = np.max(y1)

    x0 = a[:, 0] // shape[0]
    x1 = (a[:, 0] + a[:, 1]) // shape[0]
    x0 = np.min(x0)
    x1 = np.max(x1)

    if x1 > shape[1]:
        # just went out of the image dimensions
        raise ValueError("invalid RLE or image dimensions: x1=%d > shape[1]=%d" % (
            x1, shape[1]
        ))

    return [int(x0), int(y0), int(x1-x0+1), int(y1-y0+1)] #json does not recognize NumPy data types --> converted to int


def convert_annotations_detection_train(path_to_annotaion_file, path_to_categories, path_to_output_dir = "./data/imaterialist2/"):
    assert os.path.exists(path_to_output_dir)
    assert os.path.exists(path_to_categories)
    with open(path_to_categories, "r+") as labeljson:
        label_desc = json.load(labeljson)

    label_desc = label_desc["categories"]

    assert os.path.exists(path_to_annotaion_file)
    train_images = []
    test_images = []
    image_name_to_id_dict = {}
    mask_id_to_encoded_pix = {}

    mask_id_to_attributes = {}
    train_annotations = []
    test_annotations = []
    one_nth_portion = 5
    with open(path_to_annotaion_file, "r+") as traincsv:
        datareader = csv.reader(traincsv)#original csv from kaggle
        # row includes [0:'ImageId', 1:'EncodedPixels', 2:'Height', 3:'Width', 4:'ClassId', 5:'AttributesIds']
        next(datareader)#ignoring the header
        mask_id = 0
        img_id =0
        for row in tqdm(datareader):
            mask_id+=1

            image_dict = {}
            image_dict["file_name"] = row[0] + ".jpg"
            image_dict["height"] = int(row[2])
            image_dict["width"] = int(row[3])

            if row[0] not in image_name_to_id_dict: #if file name not in dict then add to dict and increase the counter
                img_id += 1
                image_dict["id"] = img_id
                if image_dict["id"] % one_nth_portion == 0:
                    train_images.append(image_dict)
                else:
                    test_images.append(image_dict)
                image_name_to_id_dict[row[0]] = img_id

            else:
                image_dict["id"] = image_name_to_id_dict[row[0]]

            #storing masks
            mask_id_to_encoded_pix[str(mask_id)] = row[1]

            #storing attributes
            mask_id_to_attributes[str(mask_id)] = row[-1]

            annot_dict = {}
            bbox= rle2bbox(row[1],(image_dict["height"],image_dict["width"]))
            area = bbox[-1]*bbox[-2] #w*h
            annot_dict["area"] = area
            annot_dict["bbox"] = bbox
            annot_dict["category_id"] = int(row[4])
            annot_dict["id"] = mask_id
            annot_dict["image_id"] = image_dict["id"]
            annot_dict["iscrowd"] = 0

            if image_dict["id"] % one_nth_portion == 0:
                test_annotations.append(annot_dict)
            else:
                train_annotations.append(annot_dict)

    #with open(path_to_output_dir + "image_name_to_id_dict.json","w+") as imgtoid:
    #    json.dump(image_name_to_id_dict,imgtoid)
    #with open(path_to_output_dir + "mask_id_to_encoded_pix.json","w+") as f:
    #    json.dump(mask_id_to_encoded_pix,f)
    #with open(path_to_output_dir + "mask_id_to_attributes.json","w+") as f:
    #    json.dump(mask_id_to_attributes,f)
    #with open(path_to_output_dir + "train_detections.json","w+") as dettrain:
    #    json.dump({"images": train_images, "annotations": train_annotations, "categories": label_desc},dettrain)
    #with open(path_to_output_dir + "test_detections.json","w+") as dettrain:
    #    json.dump({"images": test_images, "annotations": test_annotations, "categories": label_desc},dettrain)
    pass

if __name__ == "__main__":

    convert_annotations_detection_train(sys.argv[1],sys.argv[2] )