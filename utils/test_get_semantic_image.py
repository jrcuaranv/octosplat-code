from matplotlib import pyplot as plt
import numpy as np
import cv2


def get_semantic_image(image, add_seg_noise = False, conf = True):
    # Convert BGR to HSI
    hsi_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for red hues (fruits)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([20, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([179, 255, 255])

    # Threshold the HSV image to get only red hues
    mask1 = cv2.inRange(hsi_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsi_image, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask1, mask2)

    # Define lower and upper bounds for green hues (leaves)
    lower_green = np.array([40, 20, 20])  # Adjust these values based on the specific green hues in your images
    upper_green = np.array([80, 255, 255])  # Adjust these values based on the specific green hues in your images

    # Threshold the HSV image to get only green hues
    mask_green = cv2.inRange(hsi_image, lower_green, upper_green)

    green_color = (0, 255, 0)  # BGR color format: (B, G, R)
    red_color = (0, 0, 255)  # BGR color format: (B, G, R)
    
    # Adding noise to segmentation masks
    green_wrong = False
    red_wrong = False
    if (add_seg_noise):
        aux1 = np.random.sample()
        if (0.7 < aux1 and aux1 < 0.85):
            green_color = (0, 0, 255) # red
            green_wrong = True
            
        if (0.85 < aux1 and aux1 < 1.0):
            green_color = (0, 0, 0) # black
            green_wrong = True

        aux2 = np.random.sample()
        if (0.7 < aux2 and aux2 < 0.85):
            red_color = (0, 255, 0) # green
            red_wrong = True
        if (0.85 < aux2 and aux2 < 1.0):
            red_color = (0, 0, 0) # black
            red_wrong = True
        


    green_image = np.full(image.shape, green_color, dtype=np.uint8)
    red_image = np.full(image.shape, red_color, dtype=np.uint8)
    
    if conf:
        confidence_map = np.ones((image.shape[0],image.shape[1]))
        if green_wrong:
            confidence_map[mask_green==255] = 0.3
        if red_wrong:
            confidence_map[mask_red==255] = 0.3

    semantic_image = cv2.bitwise_and(green_image, green_image, mask=mask_green) + cv2.bitwise_and(red_image, red_image, mask=mask_red)
    # return image_bgr8
    if conf:
        return semantic_image.astype(np.uint8), mask_red, mask_green, confidence_map
    else:
        return semantic_image.astype(np.uint8), mask_red, mask_green
image = cv2.imread("/home/jose/Downloads/tomato.png")

semantic_img, mask_red, mask_green, conf_map = get_semantic_image(image, add_seg_noise=True, conf=True)

print(mask_red.shape)
print(mask_green)
plt.figure(1)
plt.imshow(semantic_img)

plt.figure(2)
plt.imshow(conf_map)
plt.show()