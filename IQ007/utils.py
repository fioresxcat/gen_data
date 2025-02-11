import os
import pdb
import shutil
import json
import cv2
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from copy import deepcopy
from typing_extensions import Literal, Union, Tuple, List, Dict, Optional
from torchvision import transforms as T


def is_image(fp):
    fp = str(fp)
    return fp.endswith('.jpg') or fp.endswith('.png') or fp.endswith('.jpeg') or fp.endswith('.JPG') or fp.endswith('.JPEG') or fp.endswith('.PNG')


def write_to_xml(boxes, labels, size, xml_path):
    w, h = size
    root = ET.Element('annotations')
    filename = ET.SubElement(root, 'filename')
    filename.text = Path(xml_path).stem + '.jpg'
    size = ET.SubElement(root, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(w)
    height = ET.SubElement(size, 'height')
    height.text = str(h)
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'
    for box, label in zip(boxes, labels):
        obj = ET.SubElement(root, 'object')
        name = ET.SubElement(obj, 'name')
        name.text = label
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin, ymin = ET.SubElement(bndbox, 'xmin'), ET.SubElement(bndbox, 'ymin')
        xmax, ymax = ET.SubElement(bndbox, 'xmax'), ET.SubElement(bndbox, 'ymax')
        xmin.text, ymin.text, xmax.text, ymax.text = map(str, box)
    ET.ElementTree(root).write(xml_path)


def iou_bbox(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # Compute the area of the intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the Intersection over Union (IoU)
    r1 = interArea / boxAArea
    r2 = interArea / boxBArea
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return r1, r2, iou


def resize_to_h(im: Image, new_h: int):
    w, h = im.size
    new_w = int(w * new_h / h)
    im = im.resize((new_w, new_h))
    return im


def resize_to_w(im: Image, new_w: int):
    w, h = im.size
    new_h = int(h * new_w / w)
    im = im.resize((new_w, new_h))
    return im


def random_crop(im: Image, size):
    new_w, new_h = size
    xmin = np.random.randint(0, im.width - new_w)
    ymin = np.random.randint(0, im.height - new_h)
    xmax = xmin + new_w
    ymax = ymin + new_h
    crop = im.crop((xmin, ymin, xmax, ymax))
    return crop


def find_intersection(rect1, rect2):
    # Unpack the coordinates of the rectangles
    xmin1, ymin1, xmax1, ymax1 = rect1
    xmin2, ymin2, xmax2, ymax2 = rect2
    
    # Find the intersection coordinates
    xmin_inter = max(xmin1, xmin2)
    ymin_inter = max(ymin1, ymin2)
    xmax_inter = min(xmax1, xmax2)
    ymax_inter = min(ymax1, ymax2)
    
    # Check if there is an actual intersection
    if xmin_inter >= xmax_inter or ymin_inter >= ymax_inter:
        return None  # No intersection
    
    # Return the intersection rectangle
    return (xmin_inter, ymin_inter, xmax_inter, ymax_inter)


def get_hand_bounding_box(image, alpha_threshold=70):
    alpha = image.getchannel("A")
    # Convert alpha to binary mask (1 = opaque, 0 = transparent)
    binary_alpha = alpha.point(lambda p: 255 if p > alpha_threshold else 0)
    # Get bounding box of the hand
    bbox = binary_alpha.getbbox()
    return bbox


def get_largest_foreground_region(img):
    if img.shape[2] != 4:
        raise ValueError("Image does not have an alpha channel!")
    # Extract the alpha channel (foreground mask)
    alpha_channel = img[:, :, 3]
    # Convert to binary mask (foreground = 255, background = 0)
    _, binary = cv2.threshold(alpha_channel, 128, 255, cv2.THRESH_BINARY)
    # Find all contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No foreground regions found!")
    # Get the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    # Get bounding box of the largest foreground region
    x, y, w, h = cv2.boundingRect(largest_contour)
    return (x, y, x+w, y+h)