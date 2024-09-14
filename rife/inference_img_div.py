import argparse
import warnings
from pathlib import Path

import cv2
import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from torch.nn import functional as F

from model.warplayer import warp
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
    parser.add_argument('--img-start', nargs="+", required=True)
    parser.add_argument('--img-end', nargs="+", required=True)
    parser.add_argument('-n', dest='n', default=None, type=int, required=False)
    parser.add_argument('--out-dir', default="output", required=False)
    parser.add_argument('--exp', default=4, type=int)
    parser.add_argument('--ratio', nargs="+", default=None, type=float, help='inference ratio between two images with 0 - 1 range')
    parser.add_argument('--rthreshold', default=0.02, type=float, help='returns image when actual ratio falls in given range threshold')
    parser.add_argument('--rmaxcycles', default=8, type=int, help='limit max number of bisectional cycles')
    parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')
    parser.add_argument('--smooth', action="store_true", help='smooth flow')
    parser.add_argument('--original', action="store_true", help='if original algo')

    args = parser.parse_args()

    return args


def fn(ratio, a=0.3):
    return 0.5
    b = 1 - a
    # ratio : [0, 1]
    if ratio < a:
        return 0
    if ratio > b:
        return 1

    return (ratio - a) / (b - a)


def smooth(x, width=3, sigma=1, theta=0.5):
    assert width % 2 == 1
    rad = width // 2
    distance = torch.arange(
        -rad,
        rad + 1,
        dtype=torch.float,
        device=x.device,
    )
    gaussian = torch.exp(-(distance ** 2) / (sigma ** 2))
    gaussian /= gaussian.sum()

    x_orig = x.clone()  # [T, C, H, W]
    t, c, h, w = x.size()
    x = x.flatten(start_dim=1)  # [t, c * h * w]
    x = x.transpose(0, 1)       # [c * h * w, t]
    x = x.unsqueeze(1)  # [c * h * w, 1, t]
    channels = x.size(1)  # 1
    kernel = gaussian.unsqueeze(0).unsqueeze(0)#.expand(channels, -1, -1)
    print(kernel)
    x_smooth = torch.nn.functional.conv1d(
        x,
        kernel,
        groups=channels,
    )  # [c * h * w, 1, t - 2]
    #print(x_smooth[0])
    #print(x[0, :, 1:-1])
    #print(torch.abs(x_smooth[0] -x[0, :, 1:-1]).max())
    #x_smooth = x_smooth.squeeze(0)  # [c * h * w, t - 2]
    x_smooth = x_smooth.squeeze(1)  # [c * h * w, t - 2]
    assert t - 2 * rad == x_smooth.size(1)
    x_smooth = x_smooth.transpose(0, 1)       # [t, c * h * w]
    x_smooth = x_smooth.view(t - 2 * rad, c, h, w)
    x = torch.cat(
        [
            x_orig[:rad],
            x_smooth,
            x_orig[-rad:],
        ],
        dim=0,
    )#+ torch.randn_like(x_orig)
    assert (t, c, h, w) == x.size(), (t, c, h, w, x.size())
    #x = theta * x_orig + (1 - theta) * x

    return x


# https://stackoverflow.com/questions/11435809/compute-divergence-of-vector-field-using-python
def divergence(f):
    """
    Computes the divergence of the vector field f, corresponding to dFx/dx + dFy/dy + ...
    :param f: List of ndarrays, where every item of the list is one dimension of the vector field
    :return: Single ndarray of the same shape as each of the items in f, which corresponds to a scalar field
    """
    num_dims = len(f)

    return np.ufunc.reduce(
        np.add,
        [
            np.gradient(f[i], axis=i)
            for i in range(num_dims)
        ],
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    args = parse_args()

    try:
        try:
            try:
                from model.RIFE_HDv2 import Model
                model = Model()
                model.load_model(args.modelDir, -1)
                print("Loaded v2.x HD model.")
            except:
                from train_log.RIFE_HDv3 import Model
                model = Model()
                model.load_model(args.modelDir, -1)
                print("Loaded v3.x HD model.")
        except:
            from model.RIFE_HD import Model
            model = Model()
            model.load_model(args.modelDir, -1)
            print("Loaded v1.x HD model")
    except:
        from model.RIFE import Model
        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded ArXiv-RIFE model")
    model.eval()
    model.device()

    img_size = None
    flows_masks = []
    total = 0
    n_steps = len(args.img_start)
    for idx, (img_start, img_end, ratio) in tqdm.tqdm(
        enumerate(
            zip(
                args.img_start,
                args.img_end,
                args.ratio if args.ratio is not None else [None] * len(args.img_start),
            ),
        ),
        total=len(args.img_start),
    ):
        print(img_start, img_end, ratio, fn(ratio))
        ratio = fn(ratio)
        img0 = cv2.imread(img_start, cv2.IMREAD_UNCHANGED)
        img1 = cv2.imread(img_end, cv2.IMREAD_UNCHANGED)
        if img0.ndim != 3:
            img0 = np.stack([img0] * 3, axis=2)
        if img1.ndim != 3:
            img1 = np.stack([img1] * 3, axis=2)

        if img0.shape[0] < img1.shape[0]:
            if img_size is None:
                img_size = img0.shape[:2]
            #img1 = cv2.resize(img1, img_size)
            dh = (img1.shape[0] - img_size[0]) // 2
            dw = (img1.shape[1] - img_size[1]) // 2
            img1 = img1[dh:-dh, dw:-dw].copy()
        elif img1.shape[0] < img0.shape[0]:
            if img_size is None:
                img_size = img1.shape[:2]
            #img0 = cv2.resize(img0, img_size)
            dh = (img0.shape[0] - img_size[0]) // 2
            dw = (img0.shape[1] - img_size[1]) // 2
            img0 = img0[dh:-dh, dw:-dw].copy()

        img0 = (torch.from_numpy(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
        img1 = (torch.from_numpy(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

        _, _, h, w = img0.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)

        if ratio is not None:
            # img_list = [img0]
            img0_ratio = 0.0
            img1_ratio = 1.0
            if ratio <= img0_ratio + args.rthreshold / 2:
                middle = img0
                flow = mask = tmp_img0 = tmp_img1 = None
            elif ratio >= img1_ratio - args.rthreshold / 2:
                middle = img1
                flow = mask = tmp_img0 = tmp_img1 = None
            else:
                tmp_img0 = img0
                tmp_img1 = img1
                for _ in range(args.rmaxcycles):
                    middle = model.inference(tmp_img0, tmp_img1)
                    if isinstance(middle, tuple):
                        flow, mask, middle = middle
                        flow = flow.cpu()
                        mask = mask.cpu()
                    middle_ratio = (img0_ratio + img1_ratio) / 2
                    if ratio - (args.rthreshold / 2) <= middle_ratio <= ratio + (args.rthreshold / 2):
                        break

                    if ratio > middle_ratio:
                        tmp_img0 = middle
                        img0_ratio = middle_ratio
                    else:
                        tmp_img1 = middle
                        img1_ratio = middle_ratio

            img_list = [middle]
            if tmp_img0 is not None:
                tmp_img0 = tmp_img0.cpu()
                tmp_img1 = tmp_img1.cpu()
            flows_masks.append((idx, img_start, flow, mask, tmp_img0, tmp_img1))
            #flows_masks.append((idx, img_start, flow, mask))
        else:
            img_list = [img0, img1]
            for i in range(args.exp):
                tmp = []
                for j in range(len(img_list) - 1):
                    _, _, mid = model.inference(img_list[j], img_list[j + 1])
                    tmp.append(img_list[j])
                    tmp.append(mid)
                tmp.append(img1)
                img_list = tmp

        out_dir = Path(args.out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        img_path = Path(img_start)
        if args.n is None:
            for i, img in enumerate(img_list):
                if args.original:
                    img_name = img_path.with_stem(f"{img_path.stem}_{total:0>3}_{i:0>3}").name
                else:
                    if len(img_list) == 1:
                        img_name = img_path.with_stem(f"{img_path.stem}_{idx % n_steps:0>3}").name
                    else:
                        img_name = img_path.with_stem(f"{img_path.stem}_{i:0>3}").name
                cv2.imwrite(str(out_dir / img_name), (img[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
        else:
            if args.original:
                img_name = img_path.with_stem(f"{img_path.stem}_{total:0>3}_{args.n:0>3}").name
            else:
                img_name = img_path.with_stem(f"{img_path.stem}_{args.n:0>3}").name
            cv2.imwrite(str(out_dir / img_name), (img_list[args.n][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])

        total += 1
        #if idx == n_steps:
            #break

    div_vs_ab = np.full((n_steps, ), fill_value=0, dtype=np.float64)
    div_vs_ba = np.full((n_steps, ), fill_value=0, dtype=np.float64)
    #plt.figure(figsize=(3 * 2, len(flows_masks) * 3))
    for idx, img_start, flow, mask, img0, img1 in flows_masks:
        if flow is None:
            continue

        # flow  # [1, 4, H, W]
        # img0  # [1, 3, H, W]
        flow = flow.squeeze(0)[:, :h:, :w]
        mask = mask.squeeze(0).squeeze(0).cpu().numpy()[:h, :w]
        #img0 = img0.squeeze(0)
        #img0 = (img0 * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
        #plt.imshow(img0)
        #xy = flow[2:4].cpu().numpy()[:, :h:, :w]  # [2, H, W]
        #plt.quiver(xy[0], -xy[1])
        #plt.show()
        #plt.close()

        ab = divergence(flow[2:4].cpu().numpy())
        #plt.subplot(len(flows_masks), 2, 2 * idx + 1)
        #plt.imshow(ab, vmin=-1e-3, vmax=1e-1)
        #plt.axis("off")
        ba = divergence(flow[:2].cpu().numpy())
        #plt.subplot(len(flows_masks), 2, 2 * idx + 2)
        #plt.imshow(ba, vmin=-1e-3, vmax=1e-1)
        #plt.axis("off")
        div_vs_ab[idx] = np.abs(ab * mask).mean()
        div_vs_ba[idx] = np.abs(ba * mask).mean()

    #plt.tight_layout()
    #plt.show()
    #plt.close()

    np.savetxt(str(out_dir / "div_vs_ab.txt"), div_vs_ab)
    np.savetxt(str(out_dir / "div_vs_ba.txt"), div_vs_ba)

    if not args.smooth:
        return

    assert len(flows_masks) > 0

    # smooth
    flows = torch.cat(
        [
            flow
            for _, _, flow, _, _, _ in flows_masks
        ],
        dim=0,
    )  # [T, C, H, W]

    width = 5
    sigma = 2
    #plt.plot(flows.flatten(start_dim=1)[:, 0], label="flow")
    #flows = smooth(flows, width=width, sigma=sigma)
    #plt.plot(flows.flatten(start_dim=1)[:, 0], label=f"gauss {width=} {sigma=}")
    plt.grid(ls="--")
    plt.legend()
    plt.show()
    plt.close()

    masks = torch.cat(
        [
            mask
            for _, _, _, mask, _, _ in flows_masks
        ],
        dim=0,
    )  # [T, C, H, W]
    #masks = smooth(masks, width=width, sigma=sigma)

    print(f"{len(flows_masks)=}")
    for flow, mask, (_, img_start, _, _, img0, img1) in zip(flows, masks, flows_masks):
        flow = flow.unsqueeze(0)
        mask = mask.unsqueeze(0)
        #print(f"{flow.shape=} {mask.shape=}")  # [1, 4, H, w], [1, H, W]
        #print(f"{img0.shape=} {img1.shape=}")  # [1, 4, H, w], [1, H, W]

        flow = flow.to(device)
        mask = mask.to(device)
        warped_img0 = warp(img0.to(device), flow[:, :2])
        warped_img1 = warp(img1.to(device), flow[:, 2:4])
        middle = warped_img0 * mask + warped_img1 * (1 - mask)
        img_list = [middle]

        out_dir = Path(args.out_dir)
        out_dir = out_dir.with_name(f"smooth_{out_dir.name}")
        out_dir.mkdir(exist_ok=True, parents=True)
        img_path = Path(img_start)
        if args.n is None:
            for i, img in enumerate(img_list):
                img_name = img_path.with_stem(f"{img_path.stem}_{i:0>3}").name
                cv2.imwrite(str(out_dir / img_name), (img[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])


if __name__ == "__main__":
    main()
