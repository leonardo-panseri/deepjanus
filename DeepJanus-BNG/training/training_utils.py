import cv2
import numpy as np
import matplotlib.image as mpimg


# IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 160, 320, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def load_image(image_path):
    """Load RGB images from a file."""
    try:
        return mpimg.imread(image_path)
    except FileNotFoundError as e:
        raise e


def crop(image):
    """Crops the image (removing the sky at the top and the car front at the bottom)."""
    return image[80:-1, :, :]  # remove the sky and the car front


def resize(image):
    """Resize the image to the input shape used by the network model."""
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """Convert the image from RGB to YUV (this is what the NVIDIA model does)."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    """Preprocesses an image before feeding it to the ML model."""
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image


def choose_image(center, left, right, steering_angle):
    """Randomly choose an image from the center, left or right, and adjust the steering angle."""
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(left), steering_angle + 0.2
    elif choice == 1:
        return load_image(right), steering_angle - 0.2
    return load_image(center), steering_angle


def random_flip(image, steering_angle):
    """Randomly flip the image left <-> right, and adjust the steering angle."""
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def random_translate(image, steering_angle, range_x, range_y):
    """Randomly shift the image vertically and horizontally (translation)."""
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def random_shadow(image):
    """Generates and adds random shadow."""
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is upside down.  So, the above the line:
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
    """Randomly adjusts brightness of the image."""
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:, :, 2] = hsv[:, :, 2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augment(center, left, right, steering_angle, range_x=100, range_y=10):
    """Generates an augmented image and adjust steering angle (the steering angle is associated with
    the center image)."""
    image, steering_angle = choose_image(center, left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    image = random_shadow(image)
    image = random_brightness(image)
    return image, steering_angle


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2)

    data = r'..\data\training_recordings\2023-12-08_19-30-32'
    ct = data + r'\z00003_0.09701476693753461_center.jpg'
    lt = data + r'\z00003_0.09701476693753461_left.jpg'
    rt = data + r'\z00003_0.09701476693753461_right.jpg'
    img, angle = augment(ct, lt, rt, 0.08609010383969817)
    img = preprocess(img)
    ax1.imshow(img)

    img2 = load_image(ct)
    img2 = preprocess(img2)
    ax2.imshow(img2)

    fig.show()
