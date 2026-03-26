import cv2
import numpy as np

def interactive_edge_detection(path):
    image = cv2.imread(path)
    if image is None:
        print("Error loading image")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mode = 0

    def nothing(x):
        pass

    cv2.namedWindow("Controls")

    cv2.createTrackbar("Threshold1", "Controls", 100, 255, nothing)
    cv2.createTrackbar("Threshold2", "Controls", 200, 255, nothing)
    cv2.createTrackbar("Kernel", "Controls", 3, 30, nothing)

    while True:
        t1 = cv2.getTrackbarPos("Threshold1", "Controls")
        t2 = cv2.getTrackbarPos("Threshold2", "Controls")
        k = cv2.getTrackbarPos("Kernel", "Controls")

        if k < 1:
            k = 1
        if k % 2 == 0:
            k += 1

        output = image.copy()

        if mode == 1:
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
            output = cv2.magnitude(sobelx, sobely)

        elif mode == 2:
            output = cv2.Canny(gray, t1, t2)

        elif mode == 3:
            output = cv2.Laplacian(gray, cv2.CV_64F)

        elif mode == 4:
            output = cv2.GaussianBlur(image, (k, k), 0)

        elif mode == 5:
            output = cv2.medianBlur(image, k)

        elif mode == 6:
            output = cv2.bilateralFilter(image, d=k, sigmaColor=75, sigmaSpace=75)

        if output.dtype != np.uint8:
            output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX)
            output = np.uint8(output)

        cv2.imshow("Output", output)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        elif key == ord('1'):
            mode = 1
        elif key == ord('2'):
            mode = 2
        elif key == ord('3'):
            mode = 3
        elif key == ord('4'):
            mode = 4
        elif key == ord('5'):
            mode = 5
        elif key == ord('6'):
            mode = 6
        elif key == ord('0'):
            mode = 0

    cv2.destroyAllWindows()

interactive_edge_detection("example.jpg.avif")