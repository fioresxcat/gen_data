from utils import *
from gen_words import WordsImageGen
from copy import deepcopy
from config import GenDataConfig


class ImageFaker:
    """
        This class performs generating the final pasted, blended document images.
        It uses WordsImageGen to get hw images to paste
        It selects the background (bg) by randomly choose from D4LA document images
        It determines where to paste based on the text detection boxes
        Finally it paste the hw images into the bg at the specified location
    """
    def __init__(self, bg_dir, word_dir, save_dir):
        self.word_gen = WordsImageGen(word_dir)
        self.bg_paths_dict = self.get_bg_paths(bg_dir)
        self.save_im_dir = os.path.join(save_dir, 'images')
        self.save_anno_dir = os.path.join(save_dir, 'hw_xmls')
        os.makedirs(self.save_im_dir, exist_ok=True)
        os.makedirs(self.save_anno_dir, exist_ok=True)
        self.tall_characters = list('pgfqyẠạẬậẶặẸẹỆệỊịỌọỘộỢợỤụỰựỴỵ')
        self._init_config()

    def _init_config(self):
        # set all config attribute to this class's attribute
        for k, v in vars(GenDataConfig).items():
            if not k.startswith('__') and not callable(v):  # Ignore dunder and methods
                setattr(self, k, v)


    def get_bg_paths(self, bg_dir):
        bg_paths_dict = {}
        for doc_type in os.listdir(bg_dir):
            doc_dir = os.path.join(bg_dir, doc_type)
            bg_paths_dict[doc_type] = sorted([fp for fp in list(Path(doc_dir).rglob('*')) if is_image(fp) and 'checkpoint' not in str(fp)])
        return bg_paths_dict
    

    def get_background(self, doc_type=None):
        if doc_type is None:
            doc_type = np.random.choice(list(self.bg_paths_dict.keys()))
        bg_path = np.random.choice(self.bg_paths_dict[doc_type])
        bg = Image.open(str(bg_path))
        jp = Path(bg_path).with_suffix('.json')
        with open(jp) as f:
            json_data = json.load(f)
        polys = [shape['points'] for shape in json_data['shapes'] if len(shape['points']) == 4]
        bbs = [poly2box(poly) for poly in polys]
        return bg, bbs


    def paste(self, bg: Image, roi: Image, pos: tuple):
        orig_bg_mode = bg.mode
        bg = bg.convert('RGBA')
        roi = roi.convert('RGBA')
        bg.paste(roi, pos, roi)
        bg = bg.convert(orig_bg_mode)
        return bg


    def is_normal_height_text(self, text):
        return (not any([t in text for t in self.tall_characters])) and (not any([t in unidecode.unidecode(text) for t in self.tall_characters]))
        

    def find_highest_normal_roi(self, rois, texts):
        cand_rois = [roi for roi, text in zip(rois, texts) if self.is_normal_height_text(text)]
        if len(cand_rois) == 0:
            cand_rois = rois
        return max(cand_rois, key=lambda im: im.height)


    def is_overlap_with_existing_boxes(self, bb, existing_bbs):
        for existing_bb in existing_bbs:
            r1, r2, iou = iou_poly(bb, existing_bb)
            if max(r1, r2, iou) >= self.BOX_CHECK_OVERLAP_THRESHOLD:
                return True
        return False
    


    def save(self, bg, hw_bbs, save_name, img_suffix='.jpg'):
        bg = bg.convert('RGB')
        bg.save(os.path.join(self.save_im_dir, save_name+img_suffix))

        write_to_xml(hw_bbs, ['handwriting']*len(hw_bbs), bg.size, os.path.join(self.save_anno_dir, save_name+'.xml'))
    
        

if __name__ == '__main__':
    pass