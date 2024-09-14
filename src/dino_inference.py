from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

import dino


ADE_MEAN = (np.array([123.675, 116.280, 103.530]) / 255).tolist()
ADE_STD = (np.array([58.395, 57.120, 57.375]) / 255).tolist()


def get_test_augs(img_size):
    return A.Compose(
        [
            A.LongestMaxSize(max_size=max(img_size), interpolation=1),
            A.PadIfNeeded(min_height=img_size[0], min_width=img_size[1], border_mode=cv2.BORDER_REPLICATE),#, border_mode=0, value=(255, 255, 255)),
            #A.Resize(*img_size),
            A.Normalize(mean=ADE_MEAN, std=ADE_STD),
        ]
    )


def imread(img_fpath):
    img = cv2.imread(str(img_fpath), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def remove_small_objects(mask, thresh=50):
    if thresh is not None:
        thresh = np.prod(mask.shape[:2]) * thresh / (512 * 512)

    contours, *_ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    new_mask = np.zeros_like(mask)
    for c in contours:
        area = cv2.contourArea(c)
        if area <= thresh:
            break

        cv2.drawContours(new_mask, [c], -1, (255, ), thickness=cv2.FILLED)

    return new_mask


def inference_model(model, img, transform, device="cpu", dtype=torch.float32):
    if isinstance(img, (Path, str)):
        img = imread(img)

    size = img.shape[:2]
    img = transform(image=img)["image"]
    img = np.transpose(img, (2, 0, 1))
    x = torch.from_numpy(img).to(device=device, dtype=dtype).unsqueeze(0)
    with torch.inference_mode():
        mask = model(x)

    unpad = len(transform) == 3
    if unpad:
        h, w = mask.shape[-2:]
        assert h == w
        if size[0] < size[1]:  # h < w
            r = size[0] / size[1]
            d = int(h * (1 - r) / 2)
            mask = mask[:, :, d:-d]
        elif size[0] > size[1]:  # h > w
            r = size[1] / size[0]
            d = int(w * (1 - r) / 2)
            mask = mask[:, :, :, d:-d]

    mask = torch.nn.functional.interpolate(
        mask,
        size=size,
        mode="bilinear",
        align_corners=False,
    )
    mask = mask.squeeze(0).sigmoid()
    o_mask, mask = mask.split(1, dim=0)
    mask = mask.squeeze(0).cpu().numpy()
    o_mask = o_mask.squeeze(0).cpu().numpy()

    return mask, o_mask


def get_model(model_fpath="./model_best.pth"):
    chkp = torch.load(model_fpath, map_location="cpu")
    args = chkp["args"]
    img_size = args.img_size
    transform = get_test_augs(img_size)
    backbone = args.backbone
    model = dino.Unet(
        backbone_name=backbone,
        img_size=img_size,
        n_classes=args.n_classes,
        pretrained=False,
    ).eval()
    sd = chkp["state_dict"]
    model.load_state_dict(sd)

    return model, transform


def main():
    model, transform = get_model(model_fpath="./model_best.pth")
    for img_fpath in [
        #"/home/kbrodt/projects/udem/inbetweening/inbetweening/data/imgs/1/out-0.png",
        #"/home/kbrodt/projects/udem/inbetweening/inbetweening/data/imgs/2/out-0.png",
        "/home/kbrodt/projects/udem/inbetweening/inbetweening/data/imgs/3/out-0.png",
        "/home/kbrodt/projects/udem/inbetweening/inbetweening/data/imgs/5/out-0.png",
        "/home/kbrodt/projects/udem/inbetweening/inbetweening/data/imgs/8/out-0.png",
        "/home/kbrodt/projects/udem/inbetweening/inbetweening/data/imgs/9/out-95.png",
        "/home/kbrodt/projects/udem/inbetweening/inbetweening/data/imgs/13/out-0.png",
        "/home/kbrodt/projects/udem/inbetweening/inbetweening/data/imgs/14/out-0.png",
    ]:
        img = imread(img_fpath)

        mask = inference_model(model, img, transform)

        plt.subplot(2, 2, 1)
        plt.imshow(img)
        plt.axis("off")

        plt.subplot(2, 2, 2)
        plt.imshow(mask)
        plt.axis("off")

        mask = ((mask > 0.5) * 255).astype("uint8")
        plt.subplot(2, 2, 3)
        plt.imshow(mask)
        plt.axis("off")

        mask = remove_small_objects(mask)
        plt.subplot(2, 2, 4)
        plt.imshow(mask)
        plt.axis("off")

        plt.show()
        plt.close()


if __name__ == "__main__":
    main()
