from Loader import getAlpha, getBoxes, getSamps, getITD, getITD2
import numpy as np
import cv2


def grey():
    image = getAlpha()

    if image:
        # Convert the PIL image to a NumPy array for OpenCV
        image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Convert the image to grayscale using OpenCV
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

        # Display the grayscale image using OpenCV
        cv2.imshow("Grayscale Image", gray_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No image to process.")


def ycrcb():
    image = getAlpha()

    if image:
        # Convert the PIL image to a NumPy array for OpenCV
        image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2YCrCb)

        # Display the grayscale image using OpenCV
        cv2.imshow("YCRCB Image", image_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No image to process.")


def hls():
    image = getAlpha()

    if image:
        # Convert the PIL image to a NumPy array for OpenCV
        image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HLS)

        # Display the grayscale image using OpenCV
        cv2.imshow("HLS Image", image_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No image to process.")


def hsv():
    image = getAlpha()

    if image:
        # Convert the PIL image to a NumPy array for OpenCV
        image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)

        # Display the grayscale image using OpenCV
        cv2.imshow("HSV Image", image_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No image to process.")


def yuv():
    image = getAlpha()

    if image:
        # Convert the PIL image to a NumPy array for OpenCV
        image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2YUV)

        # Display the grayscale image using OpenCV
        cv2.imshow("YUV Image", image_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No image to process.")


def lab():
    image = getAlpha()

    if image:
        # Convert the PIL image to a NumPy array for OpenCV
        image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2Lab)

        # Display the grayscale image using OpenCV
        cv2.imshow("LAB Image", image_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No image to process.")


def submat():
    image = getAlpha()
    if image:
        # Convert the PIL image to a NumPy array for OpenCV
        image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Define the region of interest (ROI) for submatting
        x, y, w, h = 50, 50, 200, 200  # Example coordinates and size
        roi = image_array[y:y + h, x:x + w]

        # Display the submat image using OpenCV
        cv2.imshow("Submat Image", roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No image to process.")


def blueColorThresh():
    # only blue
    image = getAlpha()
    if image:
        # Convert the PIL image to a NumPy array for OpenCV
        image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for blue color in HSV
        lower_blue = np.array([100, 150, 0])
        upper_blue = np.array([140, 255, 255])

        # Create a mask for blue color
        mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

        # Apply the mask to the original image
        result = cv2.bitwise_and(image_array, image_array, mask=mask)

        # Display the result using OpenCV
        cv2.imshow("Blue Color Threshold", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No image to process.")


def yellowColorThresh():
    # only yellow
    image = getAlpha()
    if image:
        # Convert the PIL image to a NumPy array for OpenCV
        image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for yellow color in HSV
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        # Create a mask for yellow color
        mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

        # Apply the mask to the original image
        result = cv2.bitwise_and(image_array, image_array, mask=mask)

        # Display the result using OpenCV
        cv2.imshow("Yellow Color Threshold", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No image to process.")


def redColorThresh():
    # only red
    image = getAlpha()
    if image:
        # Convert the PIL image to a NumPy array for OpenCV
        image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for red color in HSV
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])

        # Create a mask for red color
        mask = cv2.inRange(hsv_image, lower_red, upper_red)

        # Apply the mask to the original image
        result = cv2.bitwise_and(image_array, image_array, mask=mask)

        # Display the result using OpenCV
        cv2.imshow("Red Color Threshold", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No image to process.")


def colorThresh(input, lower, upper):
    image = input
    if image is not None:  # Check if the image is valid
        # Convert the PIL image to a NumPy array for OpenCV
        image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)

        # Create a mask for the specified color range
        mask = cv2.inRange(hsv_image, lower, upper)

        # Apply the mask to the original image
        result = cv2.bitwise_and(image_array, image_array, mask=mask)
        return result
    else:
        print("No image to process.")
        return None


def contours(input):
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    image = colorThresh(input, lower_blue, upper_blue)

    if image is not None:
        # Convert the thresholded image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find contours on the grayscale image
        contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original image
        original_image = getAlpha()
        image_array = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        cv2.drawContours(image_array, contours, -1, (0, 255, 0), 3)

        # Display the image with contours
        cv2.imshow("Contours", image_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No image to process.")


def contoursMax():
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    image = colorThresh(getAlpha(), lower_red, upper_red)

    if image is not None:
        # Convert the thresholded image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find contours on the grayscale image
        contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original image
        original_image = getAlpha()
        image_array = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        cv2.drawContours(image_array, contours, -1, (0, 255, 0), 3)
        # draw the biggest one a different color
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(image_array, [largest_contour], -1, (255, 0, 0), 3)

        # Display the image with contours
        cv2.imshow("Contours Max Red", image_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No image to process.")


def submatArea():
    image = getAlpha()
    if image:
        # Convert the PIL image to a NumPy array for OpenCV
        image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Define submat regions
        sub1 = image_array[0:100, 0:100]  # Top-left
        sub2 = image_array[100:200, 100:200]  # Top-right
        sub3 = image_array[200:300, 200:300]  # Bottom-left
        sub4 = image_array[300:400, 300:400]  # Bottom-right

        # Calculate mean colors for each submat
        mean1 = cv2.mean(sub1)
        mean2 = cv2.mean(sub2)
        mean3 = cv2.mean(sub3)
        mean4 = cv2.mean(sub4)

        # Draw rectangles with mean colors
        cv2.rectangle(image_array, (0, 0), (100, 100), (int(mean1[0]), int(mean1[1]), int(mean1[2])), 2)
        cv2.rectangle(image_array, (100, 100), (200, 200), (int(mean2[0]), int(mean2[1]), int(mean2[2])), 2)
        cv2.rectangle(image_array, (200, 200), (300, 300), (int(mean3[0]), int(mean3[1]), int(mean3[2])), 2)
        cv2.rectangle(image_array, (300, 300), (400, 400), (int(mean4[0]), int(mean4[1]), int(mean4[2])), 2)

        # Display the image with rectangles
        cv2.imshow("Submat Areas", image_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No image to process.")


def practicalExample():
    image = getBoxes()
    if image:
        # Convert the PIL image to a NumPy array for OpenCV
        image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Define the region of interest (ROI) for submatting
        x1, y1, w, h = 150, 475, 225, 225  # Example coordinates and size
        roi1 = image_array[y1:y1 + h, x1:x1 + w]
        x2, y2 = 500, y1
        roi2 = image_array[y2:y2 + h, x2:x2 + w]
        x3, y3 = 900, y1
        roi3 = image_array[y3:y3 + h, x3:x3 + w]

        mean1 = cv2.mean(roi1)
        mean2 = cv2.mean(roi2)
        mean3 = cv2.mean(roi3)

        image_mod = image_array.copy()
        cv2.circle(image_mod, (x1 + w // 2, y1 - h // 2), 50, (int(mean1[0]), int(mean1[1]), int(mean1[2])), -1)
        cv2.circle(image_mod, (x2 + w // 2, y2 - h // 2), 50, (int(mean2[0]), int(mean2[1]), int(mean2[2])), -1)
        cv2.circle(image_mod, (x3 + w // 2, y3 - h // 2), 50, (int(mean3[0]), int(mean3[1]), int(mean3[2])), -1)

        # Draw rectangles around the ROIs
        cv2.rectangle(image_mod, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        cv2.rectangle(image_mod, (x2, y2), (x2 + w, y2 + h), (0, 255, 0), 2)
        cv2.rectangle(image_mod, (x3, y3), (x3 + w, y3 + h), (0, 255, 0), 2)

        # Display the submat image using OpenCV
        cv2.imshow("Submat Image", image_mod)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No image to process.")


def findSamplesP1(isBlue):
    image = getSamps()
    # need to get blue (0,1) and yellow
    red_lower = np.array([0, 100, 100])
    red_upper = np.array([10, 255, 255])
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    blue_lower = np.array([100, 150, 0])
    blue_upper = np.array([140, 255, 255])
    image = colorThresh(getSamps(), yellow_lower, yellow_upper)
    if isBlue == 1:
        secondary = colorThresh(getSamps(), blue_lower, blue_upper)
    else:
        secondary = colorThresh(getSamps(), red_lower, red_upper)
    final = image
    cv2.bitwise_or(image, secondary, final)
    cv2.imshow("Final", final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def findSamplesP2(isBlue):
    # Define color thresholds
    red_lower = np.array([0, 100, 100])
    red_upper = np.array([10, 255, 255])
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    blue_lower = np.array([100, 150, 0])
    blue_upper = np.array([140, 255, 255])

    # Apply yellow threshold
    image = colorThresh(getSamps(), yellow_lower, yellow_upper)

    # Apply secondary threshold based on `isBlue`
    if isBlue == 1:
        secondary = colorThresh(getSamps(), blue_lower, blue_upper)
    else:
        secondary = colorThresh(getSamps(), red_lower, red_upper)

    # Perform bitwise OR to combine masks
    final = cv2.bitwise_or(image, secondary)

    # Ensure `final` is not empty before proceeding
    if final is not None and final.size > 0:
        # Convert to grayscale
        gray_image = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)

        # Find contours
        contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original image
        original_image = getSamps()
        image_array = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        cv2.drawContours(image_array, contours, -1, (0, 255, 0), 3)

        # Display the result
        cv2.imshow("Final", image_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No valid image to process.")


def findSamplesP3(isBlue):
    # Define color thresholds
    red_lower = np.array([0, 100, 100])
    red_upper = np.array([10, 255, 255])
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    blue_lower = np.array([100, 150, 0])
    blue_upper = np.array([140, 255, 255])

    # Apply yellow threshold
    image = colorThresh(getSamps(), yellow_lower, yellow_upper)

    # Apply secondary threshold based on `isBlue`
    if isBlue == 1:
        secondary = colorThresh(getSamps(), blue_lower, blue_upper)
    else:
        secondary = colorThresh(getSamps(), red_lower, red_upper)

    # Perform bitwise OR to combine masks
    final = cv2.bitwise_or(image, secondary)

    # Ensure `final` is not empty before proceeding
    if final is not None and final.size > 0:
        # Convert to grayscale
        gray_image = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)

        # Find contours
        contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original image
        original_image = getSamps()
        image_array = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        cv2.drawContours(image_array, contours, -1, (0, 255, 0), 3)
        # get rotated bounding box
        for contour in contours:
            # Get the rotated bounding box
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)  # Convert to integer coordinates

            # Draw the rotated bounding box
            cv2.polylines(image_array, [box], isClosed=True, color=(0, 0, 0), thickness=2)

        # Display the result
        cv2.imshow("Final", image_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No valid image to process.")


def findSamplesProblematic(isBlue):
    # Define color thresholds
    red_lower = np.array([0, 100, 100])
    red_upper = np.array([10, 255, 255])
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    blue_lower = np.array([100, 150, 0])
    blue_upper = np.array([140, 255, 255])

    # Apply yellow threshold
    image = colorThresh(getITD(), yellow_lower, yellow_upper)

    # Apply secondary threshold based on `isBlue`
    if isBlue == 1:
        secondary = colorThresh(getITD(), blue_lower, blue_upper)
    else:
        secondary = colorThresh(getITD(), red_lower, red_upper)

    # Perform bitwise OR to combine masks
    final = cv2.bitwise_or(image, secondary)

    # Ensure `final` is not empty before proceeding
    if final is not None and final.size > 0:
        # Convert to grayscale
        gray_image = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)

        # Find contours
        contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original image
        original_image = getITD()
        image_array = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        cv2.drawContours(image_array, contours, -1, (0, 255, 0), 3)
        # get rotated bounding box
        for contour in contours:
            # Get the rotated bounding box
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)  # Convert to integer coordinates

            # Draw the rotated bounding box
            cv2.polylines(image_array, [box], isClosed=True, color=(0, 0, 0), thickness=2)

        # Display the result
        cv2.imshow("Final", image_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No valid image to process.")


import cv2
import numpy as np


def findSamplesProblematicZOOM(isBlue, step):
    red_lower = np.array([0, 100, 100])
    red_upper = np.array([10, 255, 255])
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    blue_lower = np.array([100, 150, 0])
    blue_upper = np.array([140, 255, 255])
    if step > 1:
        yellow_lower = np.array([10, 110, 50])
        yellow_upper = np.array([35, 255, 255])
        blue_lower = np.array([90, 70, 40])
        blue_upper = np.array([135, 255, 255])

    image = getITD2()
    image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2RGBA)

    if step > 0:
        kernel = np.ones((5, 5), np.uint8)
        image2 = cv2.erode(image_array, kernel, iterations=1)
        image3 = cv2.dilate(image2, kernel, iterations=1)
        image = cv2.GaussianBlur(image3, (5, 5), 0)
        if step > 3:
            clean_mask = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            image = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)

    yellow = colorThresh(image, yellow_lower, yellow_upper)

    if isBlue == 1:
        secondary = colorThresh(image, blue_lower, blue_upper)
        # cv2.imshow('Secondary', secondary)
    else:
        if step > 1:
            red_lower2 = np.array([170, 100, 100])
            red_upper2 = np.array([180, 255, 255])
            mask1 = colorThresh(image, red_lower, red_upper)
            mask2 = colorThresh(image, red_lower2, red_upper2)
            secondary = cv2.bitwise_or(mask1, mask2)
        else:
            secondary = colorThresh(image, red_lower, red_upper)

    final = cv2.bitwise_or(yellow, secondary)

    if final is not None and final.size > 0:
        if step < 4:
            gray_image = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            # === Advanced Watershed without skimage ===
            gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)

            # === Manual local maxima: connected components ===
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            unknown = cv2.subtract(sure_bg, sure_fg)
            markers[unknown == 255] = 0

            image_bgr = cv2.cvtColor(np.array(getITD2()), cv2.COLOR_RGB2BGR)
            markers = cv2.watershed(image_bgr, markers)

            contours = []
            for label in np.unique(markers):
                if label <= 1:
                    continue
                mask = np.uint8(markers == label)
                cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours.extend(cs)

        # Draw bounding boxes
        output = cv2.cvtColor(np.array(getITD2()), cv2.COLOR_RGB2BGR)
        closestCenter = None
        for contour in contours:
            if step > 2 and cv2.contourArea(contour) < 1000:
                continue
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.polylines(output, [box], isClosed=True, color=(0, 0, 0), thickness=2)
            if (step > 4):
                # Draw the center of the bounding box
                center = (int(rect[0][0]), int(rect[0][1]))
                cv2.circle(output, center, 5, (255, 0, 0), -1)
                # Find the closest center to the center of the image
                if closestCenter is None or (
                        abs(center[0] - output.shape[1] // 2) < abs(closestCenter[0] - output.shape[1] // 2) and abs(
                        center[1] - output.shape[0] // 2) < abs(closestCenter[1] - output.shape[0] // 2)):
                    closestCenter = center
        if closestCenter is not None:
            cv2.circle(output, closestCenter, 10, (0, 255, 0), -1)

        cv2.imshow("Final", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No valid image to process.")


if __name__ == "__main__":
    findSamplesProblematicZOOM(0, 10)
