import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os


PRED_PATH = r"F:\mohu_no_bg.png"
TARGET_PATH = r"F:\yuantu_no_bg.png"
OUTPUT_DIR = r"F:\3D_reconstruction\SSDNeRF-main\edge_viz"
EDGE_THRESHOLD = 0.1


# ==========================================

def get_edges(img_tensor, device='cpu'):

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32, device=device).view(1, 1, 3, 3)

    if img_tensor.size(1) == 3:
        gray = 0.299 * img_tensor[:, 0:1] + 0.587 * img_tensor[:, 1:2] + 0.114 * img_tensor[:, 2:3]
    else:
        gray = img_tensor

    sobel_x_expand = sobel_x.expand(gray.size(1), -1, -1, -1)
    sobel_y_expand = sobel_y.expand(gray.size(1), -1, -1, -1)

    edge_x = F.conv2d(gray, sobel_x_expand, padding=1, groups=gray.size(1))
    edge_y = F.conv2d(gray, sobel_y_expand, padding=1, groups=gray.size(1))

    edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)
    return edge_magnitude


def get_object_mask(img_cv):

    if img_cv.shape[2] == 4:

        return img_cv[:, :, 3]
    else:

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        return mask


def process_and_visualize(pred_path, target_path, output_dir, edge_threshold=0.1):
    print(f"Reading prediction: {pred_path}")
    print(f"Reading target: {target_path}")


    pred_img_raw = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
    target_img_raw = cv2.imread(target_path, cv2.IMREAD_UNCHANGED)

    if pred_img_raw is None or target_img_raw is None:
        print("Error: Could not read images.")
        return


    pred_mask = get_object_mask(pred_img_raw)
    target_mask = get_object_mask(target_img_raw)


    if pred_img_raw.shape[:2] != target_img_raw.shape[:2]:
        target_img_raw = cv2.resize(target_img_raw, (pred_img_raw.shape[1], pred_img_raw.shape[0]))
        target_mask = cv2.resize(target_mask, (pred_img_raw.shape[1], pred_img_raw.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)


    def to_tensor(img_cv):

        if img_cv.shape[2] == 4:
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGB)
        else:
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img = img_rgb.astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    pred_tensor = to_tensor(pred_img_raw)
    target_tensor = to_tensor(target_img_raw)


    print("Computing edges...")
    pred_edges = get_edges(pred_tensor)
    target_edges = get_edges(target_tensor)


    edge_weight = torch.maximum(pred_edges, target_edges)
    edge_mask_loss = (edge_weight > edge_threshold).float()
    weight_map = 1.0 + edge_mask_loss * 2.0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    def save_masked_png(tensor_data, mask_cv, name, normalize=True):

        arr = tensor_data.squeeze().detach().cpu().numpy()


        if normalize:
            min_val = arr.min()
            max_val = arr.max()
            if max_val > min_val:
                arr = (arr - min_val) / (max_val - min_val)
        arr = (arr * 255).astype(np.uint8)


        H, W = arr.shape
        out_img = np.zeros((H, W, 4), dtype=np.uint8)


        out_img[:, :, 0] = arr
        out_img[:, :, 1] = arr
        out_img[:, :, 2] = arr


        out_img[:, :, 3] = mask_cv

        path = os.path.join(output_dir, name)
        cv2.imwrite(path, out_img)
        print(f"Saved: {path}")


    save_masked_png(pred_edges, pred_mask, 'pred_edge_map.png')


    save_masked_png(target_edges, target_mask, 'target_edge_map.png')


    union_mask = cv2.bitwise_or(pred_mask, target_mask)
    save_masked_png(edge_weight, union_mask, 'union_edge_map.png')


    save_masked_png(weight_map, union_mask, 'loss_weight_mask.png')


    def save_masked_heatmap(tensor_data, mask_cv, name):
        arr = tensor_data.squeeze().detach().cpu().numpy()
        min_val = arr.min()
        max_val = arr.max()
        if max_val > min_val:
            arr = (arr - min_val) / (max_val - min_val)
        arr = (arr * 255).astype(np.uint8)


        heatmap_bgr = cv2.applyColorMap(arr, cv2.COLORMAP_INFERNO)


        b, g, r = cv2.split(heatmap_bgr)

        heatmap_bgra = cv2.merge([b, g, r, mask_cv])

        path = os.path.join(output_dir, name)
        cv2.imwrite(path, heatmap_bgra)
        print(f"Saved Heatmap: {path}")

    save_masked_heatmap(pred_edges, pred_mask, 'pred_edge_heatmap.png')
    save_masked_heatmap(target_edges, target_mask, 'target_edge_heatmap.png')
    save_masked_heatmap(edge_weight, union_mask, 'union_edge_heatmap.png')

    print("Masking complete. Backgrounds removed based on object shape.")


if __name__ == "__main__":
    process_and_visualize(PRED_PATH, TARGET_PATH, OUTPUT_DIR, EDGE_THRESHOLD)