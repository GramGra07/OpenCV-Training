from Loader import getAlpha, getBoxes
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
def submat():
    image = getAlpha()
    if image:
        # Convert the PIL image to a NumPy array for OpenCV
        image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Define the region of interest (ROI) for submatting
        x, y, w, h = 50, 50, 200, 200  # Example coordinates and size
        roi = image_array[y:y+h, x:x+w]

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
def colorThresh(lower,upper):
    image = getAlpha()
    if image:
        # Convert the PIL image to a NumPy array for OpenCV
        image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)

        # Create a mask for red color
        mask = cv2.inRange(hsv_image, lower, upper)

        # Apply the mask to the original image
        result = cv2.bitwise_and(image_array, image_array, mask=mask)
        return result
    else:
        print("No image to process.")
def contours():
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    image = colorThresh(lower_blue, upper_blue)

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
    image = colorThresh(lower_red, upper_red)

    if image is not None:
        # Convert the thresholded image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find contours on the grayscale image
        contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original image
        original_image = getAlpha()
        image_array = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        cv2.drawContours(image_array, contours, -1, (0, 255, 0), 3)
        #draw the biggest one a different color
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
        roi1 = image_array[y1:y1+h, x1:x1+w]
        x2, y2 = 500, y1
        roi2 = image_array[y2:y2+h, x2:x2+w]
        x3, y3 = 900, y1
        roi3 = image_array[y3:y3+h, x3:x3+w]

        mean1 = cv2.mean(roi1)
        mean2 = cv2.mean(roi2)
        mean3 = cv2.mean(roi3)

        image_mod = image_array.copy()
        cv2.circle(image_mod, (x1+w//2, y1-h//2), 50, (int(mean1[0]), int(mean1[1]), int(mean1[2])), -1)
        cv2.circle(image_mod, (x2+w//2, y2-h//2), 50, (int(mean2[0]), int(mean2[1]), int(mean2[2])), -1)
        cv2.circle(image_mod, (x3+w//2, y3-h//2), 50, (int(mean3[0]), int(mean3[1]), int(mean3[2])), -1)

        # Draw rectangles around the ROIs
        cv2.rectangle(image_mod, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)
        cv2.rectangle(image_mod, (x2, y2), (x2+w, y2+h), (0, 255, 0), 2)
        cv2.rectangle(image_mod, (x3, y3), (x3+w, y3+h), (0, 255, 0), 2)

        # Display the submat image using OpenCV
        cv2.imshow("Submat Image", image_mod)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No image to process.")
practicalExample()