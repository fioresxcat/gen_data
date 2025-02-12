from base import ImageFaker
from PIL import ImageDraw, Image
import os
import numpy as np
from pathlib import Path


def main(args):
    faker = ImageFaker(
        # bg_dirs=['resources/sample_new_eid'],
        bg_dirs=['resources/new_eid'],

        hand_dirs=['resources/11k_hands-transparent_cropped', 'resources/egohands_data/hands_segmented'],
        # hand_dirs=['resources/11k_hands-temp'],

        finger_dirs=['resources/11k_hands-fingers'],
        
        # object_dirs=['resources/coco2017/coco_objects']
        object_dirs=['resources/coco2017/coco_objects_temp']
    )
    if args.fake_type == 'fake_hands':
        save_dir = 'results/fake_hands'
    elif args.fake_type == 'fake_fingers':
        save_dir = 'results/fake_fingers'
    elif args.fake_type == 'fake_objects':
        save_dir = 'results/fake_objects'

    os.makedirs(save_dir, exist_ok=True)
    cnt = 0
    while cnt <= args.num_images:
        try:
            bg, bg_path = faker.get_eid(side=np.random.choice(['front', 'back']))

            # -------- fake hands ---------
            if args.fake_type == 'fake_hands':
                prob = 0.35
                current_bbs = []
                for paste_side in ['left', 'right', 'top', 'bottom']:
                    if np.random.rand() > prob: continue
                    bg, roi_bb, roi_path = faker.fake_hand(bg, current_bbs, hand_type=np.random.choice(['11k_hands', 'egohands'], p=[0.5, 0.5]), paste_side=paste_side)
                    if roi_bb is not None:
                        current_bbs.append(roi_bb)
                if len(current_bbs) == 0:
                    bg, roi_bb, roi_path = faker.fake_hand(bg, current_bbs, hand_type=np.random.choice(['11k_hands', 'egohands'], p=[0.5, 0.5]), paste_side=None)
                    if roi_bb is not None:
                        current_bbs.append(roi_bb)

            # ---------- fake fingers -----------
            elif args.fake_type == 'fake_fingers':
                prob = 0.35
                current_bbs = []
                for paste_side in ['left', 'right', 'top', 'bottom']:
                    if np.random.rand() > prob: continue
                    bg, roi_bb, roi_path = faker.fake_finger(bg, current_bbs, paste_side=paste_side)
                    if roi_bb is not None:
                        current_bbs.append(roi_bb)
                if len(current_bbs) == 0:
                    bg, roi_bb, roi_path = faker.fake_finger(bg, current_bbs, paste_side=None)
                    if roi_bb is not None:
                        current_bbs.append(roi_bb)

            # ---------- fake objects -----------
            elif args.fake_type == 'fake_objects':
                prob = 0.35
                current_bbs = []
                for paste_side in ['left', 'right', 'top', 'bottom']:
                    if np.random.rand() > prob: continue
                    bg, roi_bb, roi_path = faker.fake_object(bg, current_bbs, object_name=None, paste_to_side=np.random.rand() < 0.5, paste_side=paste_side)
                    if roi_bb is not None:
                        current_bbs.append(roi_bb)
                if len(current_bbs) == 0:
                    bg, roi_bb, roi_path = faker.fake_object(bg, current_bbs, object_name=None, paste_to_side=np.random.rand() < 0.5, paste_side=None)
                    if roi_bb is not None:
                        current_bbs.append(roi_bb)

            
            bg = bg.convert('RGB')
            if args.plot:
                draw = ImageDraw.Draw(bg)
                for roi_bb in current_bbs:
                    draw.rectangle(roi_bb, width=2, outline='red')
            bg.save(os.path.join(save_dir, f'fake-{cnt}-{roi_path.stem}.png'))
            # bg.save(f'test.png')

            cnt += 1
            print(f'Done {cnt} images')
            # break

        except Exception as e:
            print(f'ERROR: {e}')
            continue
            # raise e


if __name__ == '__main__':
    pass
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_type', type=str)
    parser.add_argument('--num_images', type=int)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    main(args)