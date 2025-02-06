import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import os
import shutil
import pdb
import cv2



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



def parse_txt(txt_fp, img_size, class2idx):
    idx2class = {v: k for k, v in class2idx.items()}
    img_w, img_h = img_size

    with open(txt_fp, 'r') as f:
        lines = f.readlines()
    boxes, names = [], []
    for line in lines:
        class_id, x, y, w, h = line.split()
        class_id = int(class_id)
        class_name = idx2class[class_id]
        x, y, w, h = float(x), float(y), float(w), float(h)
        xmin = int((x - w/2) * img_w)
        ymin = int((y - h/2) * img_h)
        xmax = int((x + w/2) * img_w)
        ymax = int((y + h/2) * img_h)
        boxes.append([xmin, ymin, xmax, ymax])
        names.append(class_name)
    return np.array(boxes, dtype=np.float32), names


def txt2xml(txt_fp, out_xp, img_size, class2idx):
    boxes, labels = parse_txt(txt_fp, img_size, class2idx)
    write_to_xml(boxes, labels, img_size, out_xp)
    print(f'Done converting {txt_fp} to {out_xp}')



def split_train_val_random(dir, out_dir, val_ratio=0.15, seed=42):
    fpaths = list(Path(dir).glob('*.jpg'))
    np.random.seed(seed)
    for _ in range(10):
        np.random.shuffle(fpaths)
    num_train = int(len(fpaths) * (1-val_ratio))
    for index, fp in enumerate(fpaths):
        if index < num_train:
            split = 'train'
        else:
            split = 'val'
        save_dir = os.path.join(out_dir, split)
        os.makedirs(save_dir, exist_ok=True)
        shutil.move(str(fp), save_dir)

        xp = fp.with_suffix('.xml')
        if xp.exists(): shutil.move(str(xp), save_dir)

        tp = fp.with_suffix('.txt')
        if tp.exists(): shutil.move(str(tp), save_dir)

        jp = fp.with_suffix('.json')
        if jp.exists(): shutil.move(str(jp), save_dir)

        print(f'done {fp} to {save_dir}')



if __name__ == '__main__':
    pass

    # class2idx = {'avatar': 0, 'checked': 1, 'uncheck': 2}
    # dir = 'temp_MS04'
    # for tp in Path(dir).glob('*.txt'):
    #     ip = tp.with_suffix('.jpg')
    #     im = cv2.imread(str(ip))
    #     h, w = im.shape[:2]
    #     txt2xml(tp, tp.with_suffix('.xml'), (w, h), class2idx)
    #     print(f'done {tp}')

    split_train_val_random(dir='results/part2', out_dir='results/part2', val_ratio=0.15, seed=42)
