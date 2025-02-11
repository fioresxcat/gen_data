from utils import *
from torchvision import transforms


class ElevenkHands:
    def __init__(self):
        pass

    @staticmethod
    def get_hand_roi_images():
        dir = 'resources/11k_hands'
        out_dir = 'resources/11k_hands-transparent_cropped'
        os.makedirs(out_dir, exist_ok=True)

        with open('resources/11k_hands-info.txt') as f:
            infos = f.readlines()[1:]
        image_info = {}
        for line in infos:
            line = line.strip()
            id, age, gender, skin_color, accessories, nailPolish, aspectOfHand, im_name, irregularities = line.split(',')
            if im_name in image_info: continue
            hand_side, hand_type = aspectOfHand.split()
            image_info[im_name] = {
                'skin_color': skin_color,
                'has_nail': nailPolish,
                'hand_side': 'up' if hand_side == 'palmar' else 'down',
                'hand_type': hand_type
            }
        
        ipaths = sorted(list(Path(dir).glob('*.jpg')))
        for _ in range(100): np.random.shuffle(ipaths)
        for ip in ipaths:
        
            # image = cv2.imread(ip)
            # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # # Apply a binary threshold to separate the hand from the white background
            # _, threshold = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            # # Find contours in the thresholded image
            # contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # hand_contour = max(contours, key=cv2.contourArea)
            # x, y, w, h = cv2.boundingRect(hand_contour)
            # # Create a mask for the hand
            # mask = np.zeros_like(gray)
            # cv2.drawContours(mask, [hand_contour], -1, (255), thickness=cv2.FILLED)
            # # Smooth the mask edges to reduce jaggedness (optional)
            # mask = cv2.GaussianBlur(mask, (5, 5), 0)
            # # Create a 4-channel image (BGRA) with transparency
            # bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            # # Set the alpha channel based on the mask
            # bgra[:, :, 3] = mask
            # cropped = bgra[y:y+h, x:x+w]

            # # Save the result as a PNG file
            # cv2.imwrite('test.png', cropped)

            if image_info[ip.name]['skin_color'] in ['very fair']:
                continue
            
            img = cv2.imread(str(ip), cv2.IMREAD_COLOR)
            # if image_info[ip.name]['skin_color'] == 'very fair':
            #     print(ip)
            #     cv2.imwrite('test.png', img)
            #     # pdb.set_trace()
            # else:
            #     continue

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Define mask for white background (tune if necessary)
            lower_white = np.array([0, 0, 200], dtype=np.uint8)
            upper_white = np.array([180, 50, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_white, upper_white)
            # Invert mask to keep the hand
            mask_inv = cv2.bitwise_not(mask)
            # Find contours
            contours, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                print("No hand detected in the image.")
                continue
            hand_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(hand_contour)
            # Create an RGBA image with transparency
            rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            # Set alpha channel using the mask
            rgba[:, :, 3] = mask_inv
            hand_cropped = rgba[y:y+h, x:x+w]

            hand_side, hand_type = image_info[ip.name]['hand_side'], image_info[ip.name]['hand_type']
            out_path = os.path.join(out_dir, f'{ip.stem}-{hand_side}-{hand_type}.png')
            cv2.imwrite(out_path, hand_cropped)
            print(f'done {ip.name}')



class EgoHand:
    def __init__(self):
        pass

    @staticmethod
    def get_labels():
        import scipy

        metadata = scipy.io.loadmat('resources/egohands_data/metadata.mat')
        annotations = metadata['video'][0]
        new_annos = {}
        for x in annotations:
            x = list(x)
            video_id, _, _, _, _, _, frame_annos = x # more info the readme
            video_id = video_id[0]
            frame_annos = frame_annos[0]
            for frame_ann in frame_annos:
                frame_id = frame_ann[0].reshape(-1)[0]
                polygons = []
                for idx, polygon in enumerate(frame_ann):
                    if idx > 0 and len(polygon) > 0:
                        polygons.append(polygon.astype(np.int32).tolist())
                frame_name = 'frame_{:04d}.jpg'.format(frame_id)
                ip = os.path.join('resources/egohands_data/_LABELLED_SAMPLES', video_id, frame_name)
                new_annos[f'{video_id}-{frame_id}'] = polygons
                print(f'done {video_id}: {frame_name}')

                # im = cv2.imread(ip)
                # # Convert polygons to NumPy arrays with correct shape
                # polygons_np = [np.array(polygon, dtype=np.int32) for polygon in polygons]
                # # Draw polygons (unfilled)
                # cv2.polylines(im, polygons_np, isClosed=True, color=(0, 255, 0), thickness=2)
                # cv2.imwrite('test.png', im)
                # pdb.set_trace()


        with open('resources/egohands_data/new_annos.json', 'w') as f:
            json.dump(new_annos, f)

    
    @staticmethod
    def get_hand_images():
        output_dir = 'resources/egohands_data/hands_segmented_filtered'
        os.makedirs(output_dir, exist_ok=True)

        with open('resources/egohands_data/new_annos.json') as f:
            annos = json.load(f)
        
        cnt = 0
        for fn, polygons in annos.items():
            video_id, frame_id = fn.split('-')
            frame_name = 'frame_{:04d}.jpg'.format(int(frame_id))
            ip = os.path.join('resources/egohands_data/_LABELLED_SAMPLES', video_id, frame_name)
            im = cv2.imread(ip)

            # Convert image to RGBA (to add transparency)
            if im.shape[2] == 3:  # If image is RGB, add alpha channel
                im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)

            for i, polygon in enumerate(polygons):
                # Convert polygon to NumPy array
                poly_np = np.array(polygon, dtype=np.int32)

                # Create an empty mask
                mask = np.zeros(im.shape[:2], dtype=np.uint8)

                # Draw filled polygon on the mask
                cv2.fillPoly(mask, [poly_np], 255)

                # Extract object using mask
                extracted = cv2.bitwise_and(im, im, mask=mask)

                # Convert mask to 4-channel (RGBA) for transparency
                transparent = np.zeros_like(im, dtype=np.uint8)
                transparent[:, :, :3] = extracted[:, :, :3]  # Copy RGB channels
                transparent[:, :, 3] = mask  # Use mask as alpha channel

                # Find bounding box of the polygon
                x, y, w, h = cv2.boundingRect(poly_np)
                pos = 'upper' if y < transparent.shape[0] // 2 else 'lower'

                # Crop the region of interest
                cropped = transparent[y:y+h, x:x+w]
                if max(w/h, h/w) > 2:
                    continue
                if (h*w) / (im.shape[0]*im.shape[1]) < 0.05:
                    continue 

                # Save as PNG with transparency
                output_path = os.path.join(output_dir, f"{video_id}-{frame_name}-{i}.png")
                cv2.imwrite(output_path, cropped)
                cnt += 1
                print(f"Saved: {output_path}, {cnt} hands so far")




class COCO:
    def __init__(self):
        pass

    @staticmethod
    def get_object_images():
        from pycocotools.coco import COCO

        # Paths (Update these paths based on your setup)
        COCO_ROOT = "resources/coco2017"
        IMAGE_DIR = os.path.join(COCO_ROOT, "val2017")  # Path to images
        ANNOTATION_FILE = os.path.join(COCO_ROOT, "annotations", "instances_val2017.json")  # COCO annotations
        OUTPUT_DIR = os.path.join(COCO_ROOT, "coco_objects")  # Where to save extracted images

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        coco = COCO(ANNOTATION_FILE)
        image_ids = coco.getImgIds()
        cat_id_to_name = {cat["id"]: cat["name"] for cat in coco.loadCats(coco.getCatIds())}
        
        def extract_coco_objects(image_id):
            """
            Extracts segmented objects from a COCO image and saves them with transparency.
            """
            # Load image info
            image_info = coco.loadImgs(image_id)[0]
            file_name = image_info["file_name"]
            img_path = os.path.join(IMAGE_DIR, file_name)
            
            # Load image
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f"Failed to load image: {file_name}")
                return

            # Convert image to RGBA (for transparency)
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            
            # Get annotations for this image
            annotation_ids = coco.getAnnIds(imgIds=image_id)
            annotations = coco.loadAnns(annotation_ids)

            for i, ann in enumerate(annotations):
                try:
                    if "segmentation" not in ann or not ann["segmentation"]:
                        continue
                    cl_name = cat_id_to_name[ann['category_id']]
                    if cl_name in ['person', 'bicycle', 'car', "airplane", "bus", "train", "truck", "boat"]:
                        continue

                    # Create an empty mask
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)

                    # Draw all segmentation polygons
                    for seg in ann["segmentation"]:
                        poly_np = np.array(seg, dtype=np.int32).reshape(-1, 2)
                        cv2.fillPoly(mask, [poly_np], 255)

                    # Extract object using mask
                    extracted = cv2.bitwise_and(image, image, mask=mask)

                    # Add transparency using mask as alpha channel
                    transparent = np.zeros_like(image, dtype=np.uint8)
                    transparent[:, :, :3] = extracted[:, :, :3]  # Copy RGB channels
                    transparent[:, :, 3] = mask  # Use mask as alpha channel

                    # Find bounding box and crop
                    x, y, w, h = cv2.boundingRect(mask)
                    cropped = transparent[y:y+h, x:x+w]
                    if w < 50 or h < 50:
                        continue

                    # Save as PNG
                    output_path = os.path.join(OUTPUT_DIR, f"{file_name.split('.')[0]}-{cl_name}-{i}.png")
                    cv2.imwrite(output_path, cropped)
                    print(f"Saved: {output_path}")

                except Exception as e:
                    print(f'ERROR: {e}')
                    continue
        # Process first 10 images as an example
        for image_id in image_ids[:]:
            try:
                extract_coco_objects(image_id)
            except:
                continue


def segment_fingers():
    dir = 'resources/11k_hands-transparent_cropped'
    # dir = 'resources/11k_hands-temp'
    out_dir = 'resources/11k_hands-fingers'
    os.makedirs(out_dir, exist_ok=True)

    for ip in Path(dir).glob('*.png'):
        try:
            # if 'Hand_0011555-down-left' not in ip.name:
            #     continue
            # Load image with alpha channel (RGBA)
            img = cv2.imread(str(ip), cv2.IMREAD_UNCHANGED)
            im_h, im_w = img.shape[:2]
            
            # Extract alpha channel (transparency mask)
            if img.shape[2] == 4:
                hand_mask = img[:, :, 3]  # Alpha channel as mask
            else:
                raise ValueError("Image does not have an alpha channel!")

            # Convert mask to binary
            _, binary = cv2.threshold(hand_mask, 128, 255, cv2.THRESH_BINARY)

            # Find contours of the hand
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                raise ValueError("No hand contour found!")
            
            hand_contour = max(contours, key=cv2.contourArea)  # Get the largest contour

            # Find convex hull and defects
            hull = cv2.convexHull(hand_contour, returnPoints=False)
            defects = cv2.convexityDefects(hand_contour, hull)

            # Prepare an output image
            output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

            # Find finger tips using convex hull
            hull_points = [hand_contour[i[0]] for i in hull]
            hull_points = np.array(hull_points, dtype=np.int32)

            # Draw convex hull
            # cv2.drawContours(output, [hull_points], -1, (0, 255, 0), 2)

            # Process convexity defects to segment fingers
            fingers = []
            far_points = []
            valid_finger_cnt = 0
            if defects is not None:
                for i in range(defects.shape[0]):
                    start_idx, end_idx, far_idx, _ = defects[i, 0]
                    start = tuple(hand_contour[start_idx][0])
                    end = tuple(hand_contour[end_idx][0])
                    far = tuple(hand_contour[far_idx][0])

                    # Filter defects based on distance (removes noise)
                    if cv2.norm(np.array(start) - np.array(far)) > 20 and cv2.norm(np.array(end) - np.array(far)) > 20:
                        fingers.append((start, end))

                        # Draw fingers (segments)
                        cv2.circle(output, start, 5, (255, 0, 0), -1)
                        cv2.circle(output, end, 5, (255, 0, 0), -1)
                        cv2.circle(output, far, 5, (0, 0, 255), -1)
                        cv2.line(output, start, end, (255, 255, 0), 2)

                        far_points.append(far)
                        finger_ymax = max(start[1], end[1])
                        finger_ymin = far[1]
                        if 0.6 * im_h >= finger_ymax - finger_ymin >= im_h // 3:
                            valid_finger_cnt += 1
            
            if valid_finger_cnt >= 3:
                print('Image valid, splitting ...')
                far_points.sort(key=lambda x: x[0])
                transparent_im = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                for i in range(len(far_points)-1):
                    pt1 = far_points[i]
                    pt2 = far_points[i+1]
                    xmin = pt1[0]
                    xmax = pt2[0]
                    ymin = min(pt1[1], pt2[1])
                    ymax = im_h
                    finger_crop = transparent_im[ymin:ymax, xmin:xmax]
                    finger_bb = get_largest_foreground_region(finger_crop)
                    finger_crop = finger_crop[finger_bb[1]:finger_bb[3], finger_bb[0]:finger_bb[2]]
                    finger_h, finger_w = finger_crop.shape[:2]
                    if finger_h/finger_w < 2 or finger_h/finger_w > 8:
                        continue
                    if min(finger_crop.shape[:2]) < 50:
                        print(f'Finger too small, skipping ...')
                        continue
                    save_name = f'{ip.stem}-finger_{i}.png'
                    cv2.imwrite(os.path.join(out_dir, save_name), finger_crop)
                    print(f'done {save_name}')
            else:
                print('Image not valid')

        except Exception as e:
            print('ERROR: ', e)
            continue

def segment_and_crop_fingers(image_path, output_folder="fingers_output"):
    # Load image with alpha channel (RGBA)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Ensure image has an alpha channel
    if img.shape[2] != 4:
        raise ValueError("Image does not have an alpha channel!")

    # Extract the alpha channel (hand mask)
    hand_mask = img[:, :, 3]

    # Convert mask to binary
    _, binary = cv2.threshold(hand_mask, 128, 255, cv2.THRESH_BINARY)

    # Find contours of the hand
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No hand contour found!")

    hand_contour = max(contours, key=cv2.contourArea)  # Get the largest contour (hand)

    # Find convex hull and defects
    hull = cv2.convexHull(hand_contour, returnPoints=False)
    defects = cv2.convexityDefects(hand_contour, hull)

    if defects is None:
        raise ValueError("No convexity defects found. Ensure fingers are separated!")

    # Prepare an output directory
    os.makedirs(output_folder, exist_ok=True)

    # Extract finger segments using convexity defects
    finger_masks = np.zeros_like(binary)
    finger_contours = []

    for i in range(defects.shape[0]):
        start_idx, end_idx, far_idx, _ = defects[i, 0]
        start = tuple(hand_contour[start_idx][0])
        end = tuple(hand_contour[end_idx][0])
        far = tuple(hand_contour[far_idx][0])

        # Filtering defects (only consider wide gaps)
        if cv2.norm(np.array(start) - np.array(far)) > 20 and cv2.norm(np.array(end) - np.array(far)) > 20:
            # Create a new mask for the individual finger
            single_finger_mask = np.zeros_like(binary)

            # Draw filled contour of the hand
            cv2.drawContours(single_finger_mask, [hand_contour], -1, 255, thickness=cv2.FILLED)

            # Block out the region below the finger (to isolate it)
            cv2.line(single_finger_mask, start, end, 0, thickness=20)

            # Find contours in this new mask (extract just one finger)
            finger_contour, _ = cv2.findContours(single_finger_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if finger_contour:
                finger_contours.append(max(finger_contour, key=cv2.contourArea))  # Get largest finger contour

    # Process each detected finger
    for idx, finger_contour in enumerate(finger_contours):
        # Create a mask for this specific finger
        finger_mask = np.zeros_like(binary)
        cv2.drawContours(finger_mask, [finger_contour], -1, 255, thickness=cv2.FILLED)

        # Extract the specific finger region
        finger_only = cv2.bitwise_and(img, img, mask=finger_mask)

        # Get bounding box of the finger
        x, y, w, h = cv2.boundingRect(finger_contour)

        # Crop to bounding box
        cropped_finger = finger_only[y:y+h, x:x+w]

        # Save the cropped finger with transparency
        output_filename = os.path.join(output_folder, f"finger_{idx+1}.png")
        cv2.imwrite(output_filename, cropped_finger)

        print(f"Saved {output_filename}")

    print("Finger segmentation completed!")



def nothing():
    dir = 'resources/coco2017/coco_objects'
    ipaths = list(Path(dir).glob('*'))
    for _ in range(100): np.random.shuffle(ipaths)
    for ip in ipaths[:100]:
        shutil.copy(ip, 'resources/coco2017/coco_objects_temp')
        print(f'done {ip}')
    

if __name__ == '__main__':
    pass
    # nothing()
    # ElevenkHands.get_hand_roi_images()
    # EgoHand.get_labels()
    # EgoHand.get_hand_images()
    # COCO.get_object_images()
    segment_fingers()