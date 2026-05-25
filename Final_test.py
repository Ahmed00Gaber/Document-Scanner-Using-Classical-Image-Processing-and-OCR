import cv2
import numpy as np
import pytesseract
import os
import matplotlib.pyplot as plt
import re
from difflib import SequenceMatcher

# --- CONFIGURATION ---
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# =========================================================
#                     HELPER FUNCTIONS
# =========================================================

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def word_accuracy(ground_truth, ocr_text):
    gt_words = normalize_text(ground_truth).split()
    ocr_words = normalize_text(ocr_text).split()
    if len(gt_words) == 0: return 0.0
    matcher = SequenceMatcher(None, gt_words, ocr_words)
    correct_words = sum(block.size for block in matcher.get_matching_blocks())
    return (correct_words / len(gt_words)) * 100

def word_error_rate(ground_truth, ocr_text):
    gt_words = normalize_text(ground_truth).split()
    ocr_words = normalize_text(ocr_text).split()
    matcher = SequenceMatcher(None, gt_words, ocr_words)
    substitutions = deletions = insertions = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            substitutions += max(i2 - i1, j2 - j1)
        elif tag == "delete":
            deletions += (i2 - i1)
        elif tag == "insert":
            insertions += (j2 - j1)
    N = len(gt_words)
    if N == 0: return 0.0
    return ((substitutions + deletions + insertions) / N) * 100

# =========================================================
#                     OCR PIPELINE
# =========================================================

def scan_and_ocr(image_path):
    """Scans the document, applies preprocessing, and returns OCR text and images."""
    image = cv2.imread(image_path)
    if image is None:
        return "Error: Image not found.", None, None, None, None, None, None

    orig = image.copy()
    ratio = image.shape[0] / 800.0
    img_display = cv2.resize(image, (int(image.shape[1] / ratio), 800))

    # --- Preprocessing for contour detection ---
    gray = cv2.cvtColor(img_display, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # --- Find Document Contour ---
    contours = cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = contours[0] if len(contours) == 2 else contours[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    doc_contour = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > (img_display.shape[0]*img_display.shape[1]*0.15):
            doc_contour = approx
            break

    # --- Perspective Transform ---
    if doc_contour is not None:
        pts = doc_contour.reshape(4, 2) * ratio
        rect = order_points(pts)
        (tl, tr, br, bl) = rect
        maxWidth = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        maxHeight = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    else:
        warped = orig

    # --- NEW HIGH ACCURACY PREPROCESSING ---
    # 1. Shadow Removal
    rgb_planes = cv2.split(warped)
    result_planes = []
    for plane in rgb_planes:
        dilated = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(plane, bg)
        norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(norm)
    shadow_free = cv2.merge(result_planes)

    # 2. Grayscale + Bilateral Filter
    gray_final = cv2.cvtColor(shadow_free, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray_final, 9, 75, 75)## newww

    # 3. CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)

    # 4. Adaptive Thresholding
    final_processed = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)

    # 5. OCR
    text = pytesseract.image_to_string(final_processed, config="--oem 1 --psm 3 ")

    return text, img_display, gray, blurred, edged, warped, final_processed, shadow_free

# =========================================================
#                      MAIN SCRIPT
# =========================================================

if __name__ == "__main__":

    image_path = 'Images/archive/dataset/Note/11.jpeg'
    ground_truth_path = 'Images/archive/dataset/Note/11.txt'

    if not os.path.exists(image_path):
        print("Image not found.")
        exit()

    ocr_text, img_display, gray, blurred, edged, warped, final_processed, shadow_free = scan_and_ocr(image_path)

    print("\n--- OCR RESULT ---\n")
    print(ocr_text)

    # ---------- LOAD GROUND TRUTH ----------
    if os.path.exists(ground_truth_path):
        with open(ground_truth_path, "r", encoding="utf-8") as f:
            ground_truth = f.read()
        acc = word_accuracy(ground_truth, ocr_text)
        wer = word_error_rate(ground_truth, ocr_text)
        print("\n📊 OCR EVALUATION METRICS")
        print(f"Word Accuracy: {acc:.2f}%")
        print(f"Word Error Rate (WER): {wer:.2f}%")
    else:
        print("\n⚠ Ground truth file not found. Accuracy not calculated.")

    # ---------- VISUALIZATION ----------
    fig, axes = plt.subplots(2, 3, figsize=(24, 10))
    axes = axes.flatten()
    images = [img_display, gray, blurred, edged, shadow_free, final_processed]
    titles = ["Original", "Gray", "Blurred", "Edges", "Shadow Free", "OCR Ready"]

    for i in range(len(images)):
        if len(images[i].shape) == 2:
            axes[i].imshow(images[i], cmap='gray')
        else:
            axes[i].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        axes[i].set_title(titles[i])
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
