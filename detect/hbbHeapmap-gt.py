# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def to_gray_any(img):
    """把任意读入图转成灰度（保留原始bit深度）"""
    if img is None:
        return None
    if img.ndim == 2:
        return img
    # 3通道
    if img.shape[2] == 3:
        # 如果是“RGB灰度”（三个通道完全一样），直接取一个通道
        if np.array_equal(img[:, :, 0], img[:, :, 1]) and np.array_equal(img[:, :, 1], img[:, :, 2]):
            return img[:, :, 0]
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 4通道
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"不支持的图像维度: {img.shape}")

def normalize_percentile_to_uint8(x, p_low=2, p_high=98):
    """
    百分位裁剪 + 归一化到 0~255，适合红外这种灰度强度图（强烈推荐）
    x 可以是 uint8/uint16/float
    """
    x = x.astype(np.float32)
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.uint8)
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo)
    return (x * 255).astype(np.uint8)

def apply_clahe(gray_u8, clipLimit=2.0, tileGridSize=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(gray_u8)

def heatmap_and_overlay(gray_u8, base_bgr, alpha=0.5, cmap=cv2.COLORMAP_JET):
    heat = cv2.applyColorMap(gray_u8, cmap)  # BGR
    base = base_bgr.copy()
    heat = cv2.resize(heat, (base.shape[1], base.shape[0]))
    overlay = cv2.addWeighted(base, 1 - alpha, heat, alpha, 0)
    return heat, overlay

def process_one_image(img_path, out_dir, prefix, use_clahe=False,
                      p_low=2, p_high=98, alpha=0.5, cmap=cv2.COLORMAP_JET):
    """
    对“源图”做强度热力图（无模型、无权重）
    """
    ensure_dir(out_dir)

    # 用 UNCHANGED 读取，尽量保留 16bit 可能性
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"读不到图像: {img_path}")

    gray = to_gray_any(img)

    # 生成用于显示的底图（BGR）
    if img.ndim == 2:
        base_bgr = cv2.cvtColor(normalize_percentile_to_uint8(gray, p_low, p_high), cv2.COLOR_GRAY2BGR)
    else:
        base_bgr = img if img.shape[2] == 3 else cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # 百分位裁剪归一化
    gray_u8 = normalize_percentile_to_uint8(gray, p_low, p_high)

    # 可选：CLAHE增强
    if use_clahe:
        gray_u8 = apply_clahe(gray_u8)

    heat, overlay = heatmap_and_overlay(gray_u8, base_bgr, alpha=alpha, cmap=cmap)

    cv2.imwrite(os.path.join(out_dir, f"{prefix}_gray_u8.png"), gray_u8)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_heatmap.png"), heat)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_overlay.png"), overlay)

    print(f"✅ {prefix} 输出完成 -> {out_dir}")

def main():
    # ========== 你只需要改这三处路径 ==========
    # <<<< 修改1：红外图路径 >>>>
    ir_path = "/home/yuan/my_project/TwoStream_Yolov8-main/TwoStream_Yolov8-main/images/ir/03003.jpg"

    # <<<< 修改2：可见光图路径（如果没有可见光就注释掉相关调用） >>>>
    vis_path = "/home/yuan/my_project/TwoStream_Yolov8-main/TwoStream_Yolov8-main/images/rgb/03003.jpg"

    # <<<< 修改3：输出目录 >>>>
    out_dir = "our-result/05191_heapmap"
    # =====================================

    # 红外：建议用百分位裁剪 + （可选）CLAHE
    process_one_image(
        ir_path, out_dir, prefix="ir",
        use_clahe=False,      # 你觉得不够清晰可以改 True
        p_low=2, p_high=98,   # 红外常用 2~98 或 1~99
        alpha=0.5,
        cmap=cv2.COLORMAP_JET
    )

    # 可见光（如果你有）
    if os.path.exists(vis_path):
        process_one_image(
            vis_path, out_dir, prefix="vis",
            use_clahe=False,
            p_low=2, p_high=98,
            alpha=0.5,
            cmap=cv2.COLORMAP_JET
        )

if __name__ == "__main__":
    main()
