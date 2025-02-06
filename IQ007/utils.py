import os
import pdb
import shutil
import json
import cv2
from PIL import Image
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

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

