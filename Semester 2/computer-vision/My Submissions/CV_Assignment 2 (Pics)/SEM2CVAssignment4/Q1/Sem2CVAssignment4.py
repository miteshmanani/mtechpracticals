import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
img1 = cv2.imread(
    "C:\\Users\\mites\\OneDrive\\Desktop\\CV_Assignment 2 (Pics)\\img1.jpg", cv2.IMREAD_COLOR)
img2 = cv2.imread(
    "C:\\Users\\mites\\OneDrive\\Desktop\\CV_Assignment 2 (Pics)\\img2.jpg", cv2.IMREAD_COLOR)

pts1 = np.array([[188, 234],
                 [429, 225],
                 [245, 402],
                 [159, 612],
                 [479, 500],
                 [299, 567],
                 [335, 314],
                 [293, 228]], dtype="float32")

pts2 = np.array([[164, 237],
                 [396, 194],
                 [249, 390],
                 [188, 612],
                 [484, 453],
                 [338, 535],
                 [371, 426],
                 [323, 291]], dtype="float32")


# Compute Fundamental Matrix using RANSAC
F, mask = cv2.findFundamentalMat(
    pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=3)

if F is None:
    print("Failed to compute a Fundamental Matrix. Check your input points.")
    exit()

print("Fundamental Matrix:\n", F)

# Draw epipolar lines


def draw_epipolar_lines(F, pts1, pts2, img1, img2):
    # Compute the epilines in both images
    lines1 = cv2.computeCorrespondEpilines(
        pts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(
        pts1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)

    # Draw epilines on the images
    def draw_lines(img, lines, pts):
        for r, pt in zip(lines, pts):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(
                int, [img.shape[1], -(r[2] + r[0] * img.shape[1]) / r[1]])
            img = cv2.line(img, (x0, y0), (x1, y1), color, 1)
            img = cv2.circle(img, tuple(pt.astype(int)), 5, color, -1)
        return img

    img1_lines = draw_lines(img1.copy(), lines1, pts1)
    img2_lines = draw_lines(img2.copy(), lines2, pts2)

    return img1_lines, img2_lines


img1_lines, img2_lines = draw_epipolar_lines(F, pts1, pts2, img1, img2)

# Display the results
plt.subplot(121), plt.imshow(cv2.cvtColor(
    img1_lines, cv2.COLOR_BGR2RGB)), plt.title("Epilines in Image 1")
plt.subplot(122), plt.imshow(cv2.cvtColor(
    img2_lines, cv2.COLOR_BGR2RGB)), plt.title("Epilines in Image 2")
plt.show()
