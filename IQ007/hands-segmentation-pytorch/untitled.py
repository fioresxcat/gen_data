import torch
import pdb
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

def test_model():
    # Imports
    import torch

    # Create the model
    model = torch.hub.load(
        repo_or_dir='guglielmocamporese/hands-segmentation-pytorch', 
        model='hand_segmentor', 
        pretrained=True
    )
    model.eval()
    # pdb.set_trace()

    im = Image.open('/home/fiores/Desktop/VNG/gen_data/IQ007/resources/egohands_data/_LABELLED_SAMPLES/CARDS_COURTYARD_S_H/frame_0037.jpg')
    W, H = im.size
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        lambda x: x if x.shape[0] == 3 else x.repeat(3, 1, 1),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    inp = image_transform(im)
    inp = inp.unsqueeze(0)
    logits = model(inp).detach().cpu()
    preds = F.softmax(logits, 1).squeeze(0)[1] * 255 # [h, w]
    preds = Image.fromarray(preds.numpy().astype(np.uint8), 'L')
    preds = preds.resize((W, H), resample=Image.BICUBIC)
    preds.save(f'test.png')

if __name__ == '__main__':
    pass
    test_model()