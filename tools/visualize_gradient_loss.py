import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os

# ==========================================
# 👇 在这里修改你的图片路径
# ==========================================
PRED_PATH = r"F:\mohu_no_bg.png"
TARGET_PATH = r"F:\yuantu_no_bg.png"
OUTPUT_DIR = r"F:\3D_reconstruction\SSDNeRF-main\gradient_viz"


# ==========================================

def compute_image_gradients(images, device='cpu'):

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32, device=device).view(1, 1, 3, 3)

    if images.size(1) == 3:
        gray = 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
    else:
        gray = images

    sobel_x_expand = sobel_x.expand(gray.size(1), -1, -1, -1)
    sobel_y_expand = sobel_y.expand(gray.size(1), -1, -1, -1)

    grad_x = F.conv2d(gray, sobel_x_expand, padding=1, groups=gray.size(1))
    grad_y = F.conv2d(gray, sobel_y_expand, padding=1, groups=gray.size(1))

    return grad_x, grad_y


def get_object_mask(img_cv):

    if img_cv.shape[2] == 4:

        return img_cv[:, :, 3]
    else:

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        return mask


def process_and_visualize(pred_path, target_path, output_dir):
    print(f"Reading prediction: {pred_path}")
    print(f"Reading target: {target_path}")


    pred_img_raw = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
    target_img_raw = cv2.imread(target_path, cv2.IMREAD_UNCHANGED)

    if pred_img_raw is None or target_img_raw is None:
        print("Error: Could not decode images.")
        return


    pred_mask = get_object_mask(pred_img_raw)
    target_mask = get_object_mask(target_img_raw)

    # Resize target if needed
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


    print("Computing gradients...")
    pred_grad_x, pred_grad_y = compute_image_gradients(pred_tensor)
    target_grad_x, target_grad_y = compute_image_gradients(target_tensor)


    diff_x = (pred_grad_x - target_grad_x) ** 2
    diff_y = (pred_grad_y - target_grad_y) ** 2
    gradient_consistency_error = torch.sqrt(diff_x + diff_y)


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    def save_masked_heatmap(tensor, mask_cv, name):

        arr = tensor.squeeze().detach().abs().numpy()  # 取绝对值


        min_val = arr.min()
        max_val = arr.max()
        if max_val > min_val:
            arr = (arr - min_val) / (max_val - min_val)
        arr = (arr * 255).astype(np.uint8)


        heatmap_bgr = cv2.applyColorMap(arr, cv2.COLORMAP_JET)


        b, g, r = cv2.split(heatmap_bgr)


        heatmap_bgra = cv2.merge([b, g, r, mask_cv])

        output_path = os.path.join(output_dir, name)
        cv2.imwrite(output_path, heatmap_bgra)
        print(f"Saved: {output_path}")


    save_masked_heatmap(pred_grad_x, pred_mask, 'pred_grad_x.png')
    save_masked_heatmap(target_grad_x, target_mask, 'target_grad_x.png')

    save_masked_heatmap(pred_grad_y, pred_mask, 'pred_grad_y.png')
    save_masked_heatmap(target_grad_y, target_mask, 'target_grad_y.png')


    pred_mag = torch.sqrt(pred_grad_x ** 2 + pred_grad_y ** 2)
    target_mag = torch.sqrt(target_grad_x ** 2 + target_grad_y ** 2)

    save_masked_heatmap(pred_mag, pred_mask, 'pred_grad_magnitude.png')
    save_masked_heatmap(target_mag, target_mask, 'target_grad_magnitude.png')


    union_mask = cv2.bitwise_or(pred_mask, target_mask)
    save_masked_heatmap(gradient_consistency_error, union_mask, 'gradient_consistency_error_map.png')

    print("Done! Images check output directory.")


if __name__ == "__main__":
    try:
        import cv2
    except ImportError:
        os.system('pip install opencv-python')
        import cv2

    process_and_visualize(PRED_PATH, TARGET_PATH, OUTPUT_DIR)