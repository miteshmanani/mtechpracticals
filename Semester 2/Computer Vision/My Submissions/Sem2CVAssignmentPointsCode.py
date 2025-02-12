import cv2
import numpy as np

# Function to collect points from an image


def select_points(image_path, window_name):
    img = cv2.imread(image_path)
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
            points.append([x, y])
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # Mark point on image
            cv2.imshow(window_name, img)
            print(f"Point selected: ({x}, {y})")

    cv2.imshow(window_name, img)
    cv2.setMouseCallback(window_name, click_event)
    print(f"Click on the image to select points. Press 'ESC' when done.")
    cv2.waitKey(0)  # Wait until 'ESC' key is pressed
    cv2.destroyAllWindows()

    return np.array(points, dtype="float32")


# Main execution
if __name__ == "__main__":
    # Select points for img1
    print("Select points in img1.jpg")
    img1_points = select_points(
        "C:\\Users\\mites\\OneDrive\\Desktop\\CV_Assignment 2 (Pics)\\img1.jpg", "Image 1")
    print(f"Points in Image 1: {img1_points}")

    # Select points for img2
    print("Select points in img2.jpg")
    img2_points = select_points(
        "C:\\Users\\mites\\OneDrive\\Desktop\\CV_Assignment 2 (Pics)\\img2.jpg", "Image 2")
    print(f"Points in Image 2: {img2_points}")

    # Save or process the points
    if len(img1_points) == len(img2_points):
        print("Points selected successfully.")
        print("Image 1 Points:", img1_points)
        print("Image 2 Points:", img2_points)
    else:
        print("Error: Mismatch in number of points between the two images.")
