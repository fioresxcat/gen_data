from utils import *


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
        output_dir = 'resources/egohands_data/hands_segmented'
        os.makedirs(output_dir, exist_ok=True)

        with open('resources/egohands_data/new_annos.json') as f:
            annos = json.load(f)
        
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

                # Crop the region of interest
                cropped = transparent[y:y+h, x:x+w]

                # Save as PNG with transparency
                output_path = os.path.join(output_dir, f"{video_id}-{frame_name}-{i}.png")
                cv2.imwrite(output_path, cropped)
                print(f"Saved: {output_path}")



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





def nothing():
    for jp in Path('/data/tungtx2/gen_data/IQ007/resources/egohands_data/_LABELLED_SAMPLES').rglob('*.json'):
        os.remove(jp)
    

if __name__ == '__main__':
    pass
    # nothing()
    ElevenkHands.get_hand_roi_images()
    # EgoHand.get_labels()
    # EgoHand.get_hand_images()
    # COCO.get_object_images()