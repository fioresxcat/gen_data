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


def nothing():
    bg = Image.open('resources/sample_new_eid/new_eid1.png')
    roi = Image.open('test.png')
    bg = bg.convert('RGBA')
    roi = roi.convert('RGBA')
    roi = roi.resize((bg.width//2, bg.height//2))
    pos = (bg.width//3, -50)
    bg.paste(roi, pos, roi)
    bg = bg.convert('RGB')
    bg.save('test1.png')
    

if __name__ == '__main__':
    pass
    # nothing()
    ElevenkHands.get_hand_roi_images()