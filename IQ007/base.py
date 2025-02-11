from utils import *


def get_all_im_paths(list_dirs):
    im_paths = []
    for dir in list_dirs:
        im_paths.extend([fp for fp in Path(dir).rglob('*') if is_image(fp)])
    return im_paths



class ImageFaker:
    def __init__(self, bg_dirs, hand_dirs, object_dirs):
        self.bg_paths = get_all_im_paths(bg_dirs)
        self.hand_paths_dict = self.get_hand_paths_dict(hand_dirs)
        self.object_paths_dict = self.get_object_paths_dict(object_dirs)


    def get_hand_paths_dict(self, hand_dirs):
        d = {'11k_hands': [], 'egohands': []}
        for dir in hand_dirs:
            for ip in Path(dir).rglob('*'):
                if not is_image(ip): continue
                if '11k_hands' in str(ip):
                    d['11k_hands'].append(ip)
                elif 'egohands' in str(ip):
                    d['egohands'].append(ip)
        return d


    def get_object_paths_dict(self, object_dirs):
        d = {}
        for dir in object_dirs:
            for ip in Path(dir).rglob('*'):
                if not is_image(ip): continue
                _, cl_name, _ = ip.name.split('-')
                if cl_name not in d:
                    d[cl_name] = [ip]
                else:
                    d[cl_name].append(ip)
        
        return d
    

    def paste(self, bg: Image, roi: Image, pos: tuple):
        orig_bg_mode = bg.mode
        bg = bg.convert('RGBA')
        roi = roi.convert('RGBA')
        bg.paste(roi, pos, roi)
        bg = bg.convert(orig_bg_mode)
        return bg
    

    def get_bg(self):
        bg_path = np.random.choice(self.bg_paths)
        bg = Image.open(bg_path)
        return bg
    

    def get_hand(self, hand_type: Literal['11k_hands', 'egohands']=None):
        ip = np.random.choice(self.hand_paths_dict[hand_type])
        return Image.open(ip).convert('RGBA'), ip
    

    def get_object(self, cl_name=None):
        if cl_name is None:
            cl_name = np.random.choice(list(self.object_paths_dict.keys()))
        ip = np.random.choice(self.object_paths_dict[cl_name])
        return Image.open(ip), ip
    


    def get_real_object_bb(self, roi, new_bb, bg_w, bg_h):
        inter_bb_abs = [
            int(np.clip(new_bb[0], 0, bg_w)),
            int(np.clip(new_bb[1], 0, bg_h)),
            int(np.clip(new_bb[2], 0, bg_w)),
            int(np.clip(new_bb[3], 0, bg_h)),
        ]
        inter_bb_rel = [
            int(inter_bb_abs[0]-new_bb[0]),
            int(inter_bb_abs[1]-new_bb[1]),
            int(inter_bb_abs[2]-new_bb[0]),
            int(inter_bb_abs[3]-new_bb[1]),
        ]  # relative to roi
        inter_region = roi.crop(inter_bb_rel)
        real_hand_bbox_rel = get_hand_bounding_box(inter_region)  # relative to inter_region

        # draw = ImageDraw.Draw(inter_region)
        # draw.rectangle(real_hand_bbox_rel, outline='red', width=2)
        # inter_region.save('test.png')
        # pdb.set_trace()

        real_hand_bbox_rel = [
            real_hand_bbox_rel[0] + inter_bb_rel[0],
            real_hand_bbox_rel[1] + inter_bb_rel[1],
            real_hand_bbox_rel[2] + inter_bb_rel[0],
            real_hand_bbox_rel[3] + inter_bb_rel[1],
        ]  # real to roi bb

        real_hand_bbox_abs = [
            new_bb[0] + real_hand_bbox_rel[0],
            new_bb[1] + real_hand_bbox_rel[1],
            new_bb[0] + real_hand_bbox_rel[2],
            new_bb[1] + real_hand_bbox_rel[3],
        ]
        return real_hand_bbox_abs


    def paste_roi(self, bg, roi, existing_bbs, paste_side):
        def roi_transform_left_right(roi, bg_h):
            # # resize roi
            # if roi.height > roi.width:
            #     roi = roi.rotate(np.random.choice([90, 270]), expand=True)
            #     # roi = roi.rotate(np.random.choice([90]), expand=True)
            # assert roi.width >= roi.height  # roi is horizontal
            new_roi_h = int(bg_h * np.random.uniform(0.2, 0.8))
            # new_roi_h = int(h * 0.4)
            roi = resize_to_h(roi, new_roi_h)
            return roi
            

        def roi_transform_top_bottom(roi, bg_w):
            # # resize roi
            # if roi.width > roi.height:
            #     roi = roi.rotate(np.random.choice([90, 270]), expand=True)
            # assert roi.height >= roi.width  # roi is vertical
            new_roi_w = int(bg_w * np.random.uniform(0.2, 0.8))
            roi = resize_to_w(roi, new_roi_w)
            return roi


        def check_overlap(new_bb, existing_bbs):
            is_valid = True
            for cur_bb in existing_bbs:
                r1, r2, iou = iou_bbox(new_bb, cur_bb)
                if iou > 0:
                    is_valid = False
                    break
            return is_valid
        
        
        # choose position
        bg_w, bg_h = bg.size
        max_try = 10
        pos, new_bb = None, None
        roi_center_range, display_range = (0.1, 0.9), (0.1, 0.7)
        if paste_side == 'left':
            roi = roi_transform_left_right(roi, bg_h)
            roi_w, roi_h = roi.size
            for num_try in range(max_try):
                # y-axis
                cy_percent = np.random.uniform(*roi_center_range)
                # cy_percent = 0.3
                cy_dist = int(bg_h*cy_percent)
                tmp = roi_h // 2 - cy_dist
                ymin = -tmp
                # x-axis
                display_percent = np.random.uniform(*display_range)
                # display_percent = 0.4
                display_roi_w = int(roi_w * display_percent)
                display_roi_w = min(display_roi_w, int(0.6 * bg_w))  # limit too much occlusion
                xmin = -(roi_w - display_roi_w)
                # get roi bb
                new_bb = (xmin, ymin, xmin+roi_w, ymin+roi_h)
                new_bb = self.get_real_object_bb(roi, new_bb, bg_w, bg_h)
                # check overlap
                is_valid = check_overlap(new_bb, existing_bbs)
                if not is_valid:
                    print(f'Trying {num_try+1} times')
                    continue
                else:
                    pos = (xmin, ymin)
                    break
        elif paste_side == 'right':
            roi = roi_transform_left_right(roi, bg_h)
            roi_w, roi_h = roi.size
            for num_try in range(max_try):
                # y-axis
                cy_percent = np.random.uniform(*roi_center_range)
                cy_dist = int(bg_h*cy_percent)
                tmp = roi_h // 2 - cy_dist
                ymin = -tmp
                # x-axis
                display_percent = np.random.uniform(*display_range)
                display_roi_w = int(roi_w * display_percent)
                display_roi_w = min(display_roi_w, int(0.6 * bg_w))  # limit too much occlusion
                xmin = bg_w - display_roi_w
                # check overlap
                new_bb = (xmin, ymin, xmin+roi_w, ymin+roi_h)
                new_bb = self.get_real_object_bb(roi, new_bb, bg_w, bg_h)
                is_valid = check_overlap(new_bb, existing_bbs)
                if not is_valid:
                    print(f'Trying {num_try+1} times')
                    continue
                else:
                    pos = (xmin, ymin)
                    break
        elif paste_side == 'top':
            roi = roi_transform_top_bottom(roi, bg_w)
            roi_w, roi_h = roi.size
            for num_try in range(max_try):
                # x-axis
                cx_percent = np.random.uniform(*roi_center_range)
                cx_dist = int(bg_w*cx_percent)
                tmp = roi_w // 2 - cx_dist
                xmin = -tmp
                # y-axis
                display_percent = np.random.uniform(*display_range)
                display_roi_h = int(roi_h * display_percent)
                display_roi_h = min(display_roi_h, int(0.6 * bg_h))
                ymin = - (roi_h - display_roi_h)
                # check overlap
                new_bb = (xmin, ymin, xmin+roi_w, ymin+roi_h)
                new_bb = self.get_real_object_bb(roi, new_bb, bg_w, bg_h)
                is_valid = check_overlap(new_bb, existing_bbs)
                if not is_valid:
                    print(f'Trying {num_try+1} times')
                    continue
                else:
                    pos = (xmin, ymin)
                    break
        elif paste_side == 'bottom':
            roi = roi_transform_top_bottom(roi, bg_w)
            roi_w, roi_h = roi.size
            for num_try in range(max_try):
                # x-axis
                cx_percent = np.random.uniform(*roi_center_range)
                cx_dist = int(bg_w*cx_percent)
                tmp = roi_w // 2 - cx_dist
                xmin = -tmp
                # y-axis
                display_percent = np.random.uniform(*display_range)
                display_roi_h = int(roi_h * display_percent)
                display_roi_h = min(display_roi_h, int(0.6 * bg_h))
                ymin = bg_h - display_roi_h
                # check overlap
                new_bb = (xmin, ymin, xmin+roi_w, ymin+roi_h)
                new_bb = self.get_real_object_bb(roi, new_bb, bg_w, bg_h)
                is_valid = check_overlap(new_bb, existing_bbs)
                if not is_valid:
                    print(f'Trying {num_try+1} times')
                    continue
                else:
                    pos = (xmin, ymin)
                    break
        if pos is not None:
            new_bb = [
                int(np.clip(new_bb[0], 0, bg.width)),
                int(np.clip(new_bb[1], 0, bg.height)),
                int(np.clip(new_bb[2], 0, bg.width)),
                int(np.clip(new_bb[3], 0, bg.height)),
            ]
            bg = self.paste(bg, roi, pos)
        else:
            new_bb = None

        return bg, new_bb


    def fake_hand(
        self,
        bg: Image,
        existing_bbs: List,  # set of current bbs
        hand_type: Literal['11k_hands', 'egohands'] = None,
        paste_side: Literal['left', 'top', 'right', 'bottom'] = None,
    ):        
        if hand_type is None:
            hand_type = np.random.choice(['11k_hands', 'egohands'])
        roi, roi_path = self.get_hand(hand_type)
        if paste_side is None:
            paste_side = np.random.choice(['left', 'top', 'right', 'bottom'])
        if hand_type == '11k_hands':
            if paste_side == 'left': # 0, 90, 180, 270
                p = [0.1, 0.6, 0.1, 0.2]
            elif paste_side == 'top':
                p = [0.6, 0.1, 0.2, 0.1]
            elif paste_side == 'right':
                p = [0.1, 0.2, 0.1, 0.6]
            elif paste_side == 'bottom':
                p = [0.2, 0.1, 0.6, 0.1]
            roi = roi.rotate(np.random.choice([0, 90, 180, 270], p=p), expand=True)
            if np.random.rand() < 0.3:
                roi = random_crop(roi, size=(int(0.6*roi.width), int(0.6*roi.height)))
        elif hand_type == 'egohands': 
            roi = roi.rotate(np.random.choice([0, 90, 180, 270]), expand=True)
        
        bg, new_bb = self.paste_roi(bg, roi, existing_bbs, paste_side)
        return bg, new_bb, roi_path
    

    def fake_object(
        self,
        bg: Image,
        existing_bbs: List,  # set of current bbs
        object_name: str = None,
        paste_side: Literal['left', 'top', 'right', 'bottom'] = None,
    ):
        roi, roi_path = self.get_object(object_name)
        if paste_side is None:
            paste_side = np.random.choice(['left', 'top', 'right', 'bottom'])
        bg, new_bb = self.paste_roi(bg, roi, existing_bbs, paste_side)
        return bg, new_bb, roi_path
    