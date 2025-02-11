from base import ImageFaker
from PIL import ImageDraw, Image
import os
from pathlib import Path


def main():
    faker = ImageFaker(
        bg_dirs=['resources/sample_new_eid'],
        # hand_dirs=['resources/11k_hands-transparent_cropped', 'resources/egohands_data/hands_segmented'],
        hand_dirs=['resources/11k_hands-temp'],
        # object_dirs=['resources/coco2017/coco_objects']
        object_dirs=['resources/coco2017/coco_objects_temp']
    )
    save_dir = 'results/paste_objects'
    os.makedirs(save_dir, exist_ok=True)
    NUM_IMAGES = 100
    cnt = 0
    while cnt <= NUM_IMAGES:
        try:
            bg = faker.get_bg()

            # # -------- fake hands ---------
            # current_bbs = []
            # bg, roi_bb, roi_path = faker.fake_hand(bg, current_bbs, hand_type='11k_hands', paste_side='left')
            # if roi_bb is not None:
            #     current_bbs.append(roi_bb)
            # bg, roi_bb, roi_path = faker.fake_hand(bg, current_bbs, hand_type='egohands', paste_side='right')
            # if roi_bb is not None:
            #     current_bbs.append(roi_bb)
            # bg, roi_bb, roi_path = faker.fake_hand(bg, current_bbs, hand_type='11k_hands', paste_side='top')
            # if roi_bb is not None:
            #     current_bbs.append(roi_bb)
            # bg, roi_bb, roi_path = faker.fake_hand(bg, current_bbs, hand_type='egohands', paste_side='bottom')
            # if roi_bb is not None:
            #     current_bbs.append(roi_bb)

            # ---------- fake objects -----------
            current_bbs = []
            bg, roi_bb, roi_path = faker.fake_object(bg, current_bbs, paste_side='left')
            if roi_bb is not None:
                current_bbs.append(roi_bb)
            bg, roi_bb, roi_path = faker.fake_object(bg, current_bbs, paste_side='right')
            if roi_bb is not None:
                current_bbs.append(roi_bb)
            
            bg = bg.convert('RGB')
            draw = ImageDraw.Draw(bg)
            for roi_bb in current_bbs:
                draw.rectangle(roi_bb, width=2, outline='red')
            # bg.save(os.path.join(save_dir, f'fake-{cnt}-{roi_path.stem}.png'))
            bg.save(f'test.png')

            cnt += 1
            print(f'Done {cnt} images')
            break
        except Exception as e:
            print(f'ERROR: {e}')
            raise e


if __name__ == '__main__':
    pass
    main()