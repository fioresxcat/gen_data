import os
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
import random
from tqdm import tqdm

photo_folder = 'anh_the'
photos = [os.path.join(photo_folder, path) for path in os.listdir(photo_folder) if path.endswith('.jpg')]

template_folder = 'template'
templates = os.listdir(template_folder)
templates_dict = dict()
for template in templates:
    template_path = os.path.join(template_folder, template)
    points = list()
    for file in os.listdir(template_path):
        file_path = os.path.join(template_path, file)
        with open(file_path, 'r') as f:
            data = json.load(f)
        points.append(data['shapes'][0]['points'])
    w_h = list(map(lambda x: (x[1][0] - x[0][0], x[1][1] - x[0][1]), points))
    w_h = np.average(w_h, axis=0)
    templates_dict[template] = w_h

raw_image_folder = 'raw_images'
point_label_folder = 'labels/avatar_points'
image_types = os.listdir(point_label_folder)

image_folder = 'results/part2/gen_images/'
os.makedirs(image_folder, exist_ok=True)
label_folder = 'results/part2/gen_labels/'
os.makedirs(label_folder, exist_ok=True)
MAX_IMAGE_PER_TYPE = 150

for folder in tqdm(image_types, desc="Processing image types"):
    cnt = 0
    image_type_folder = os.path.join(point_label_folder, folder)
    for file in tqdm(os.listdir(image_type_folder), desc=f"Processing files in {folder}"):
        avatar_point_path = os.path.join(image_type_folder, file)
        img_path = os.path.join(raw_image_folder, folder, file.replace('.json', '.jpg'))
        template_name = os.path.basename(os.path.dirname(img_path))

        with open(avatar_point_path, 'r') as f:
            avatar_points = json.load(f)

        for _ in range(1):  # Generate each image 1 times
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            labels = []
            for shape in avatar_points['shapes']:
                point = shape['points'][0]
                point = (int(point[0]), int(point[1]))

                photo_path = random.choice(photos)
                photo = cv2.imread(photo_path)
                photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)

                # Get the template dimensions
                template_w, template_h = templates_dict[template_name]

                # Calculate the scaling factor to maintain the aspect ratio
                scale_w = template_w / photo.shape[1]
                scale_h = template_h / photo.shape[0]
                scale = min(scale_w, scale_h)

                # Resize the photo to fit within the template dimensions while maintaining the aspect ratio
                new_w = int(photo.shape[1] * scale)
                new_h = int(photo.shape[0] * scale)
                photo_resized = cv2.resize(photo, (new_w, new_h))

                # Calculate the center of the resized photo
                center_photo = (int(photo_resized.shape[1] / 2), int(photo_resized.shape[0] / 2))

                # Calculate the top-left corner of the resized photo to place it in the img array
                top_left_x = point[0] - center_photo[0]
                top_left_y = point[1] - center_photo[1]

                # Ensure the coordinates are within the bounds of the img array
                top_left_x = max(0, top_left_x)
                top_left_y = max(0, top_left_y)

                # Place the resized photo in the img array
                img[top_left_y:top_left_y + new_h, top_left_x:top_left_x + new_w] = photo_resized

                # Calculate YOLO format values for the photo
                x_center = (top_left_x + new_w / 2) / img.shape[1]
                y_center = (top_left_y + new_h / 2) / img.shape[0]
                width = new_w / img.shape[1]
                height = new_h / img.shape[0]

                # Append the label to the list
                labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            # Save the modified image
            output_image_path = os.path.join(image_folder, file.replace('.json', f'_{_}.jpg'))
            cv2.imwrite(output_image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # Save all labels in YOLO format
            output_label_path = os.path.join(label_folder, file.replace('.json', f'_{_}.txt'))
            with open(output_label_path, 'w') as label_file:
                label_file.writelines(labels)
            
            cnt += 1
        if cnt >= MAX_IMAGE_PER_TYPE:
            break
