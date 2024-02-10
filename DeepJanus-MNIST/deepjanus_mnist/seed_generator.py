import numpy as np
import tf_keras

from deepjanus_mnist.image_tools import bitmap_to_svg
from .mnist_member import MNISTMember


class BitmapDistanceMemoized:
    """Class providing an efficient way to calculate distances between greyscale bitmap images of a dataset"""

    def __init__(self, images: np.ndarray):
        """Creates a new instance for efficient distance calculation between images of the given dataset.
        Note that the pixels must be normalized floats to obtain valid results."""
        self.images = images
        self.cache = {}

    def get_distance(self, index1: int, index2: int):
        """Calculates the distance between two grayscale bitmap images at the given indices.
        If the distance was already calculated, returns it from cache without recalculating it."""
        index_str = tuple(sorted([index1, index2]))
        if index_str in self.cache:
            return self.cache[index_str]

        img1 = self.images[index1]
        img2 = self.images[index2]
        distance = np.linalg.norm(img2 - img1)

        self.cache.update({index_str: distance})

        return distance

    def calculate_min_distance_from_others(self, index: int, other_indices: list[int]):
        """Calculates the minimum distance of the image at the given index from a group of other images."""
        n = len(other_indices)
        distances = np.zeros((n,))
        for i in range(n):
            distances[i] = self.get_distance(index, other_indices[i])
        return distances.min()


def seed_candidate_generator(expected_label: int = None):
    """Generates candidate members for seed creation."""
    # Load the MNIST dataset
    mnist = tf_keras.datasets.mnist
    (images, labels), (_, _) = mnist.load_data()

    # Build list of indices that can be used to generate members
    if expected_label is None:
        image_indices = np.arange(len(images))
    else:
        image_indices = np.nonzero(labels == expected_label)[0]

    generated_indices = []
    # The first element is chosen randomly
    starting_point = np.random.default_rng().choice(image_indices)
    generated_indices.append(starting_point.item())
    image_indices = image_indices[image_indices != starting_point]

    # To generate a diverse set of seeds, yield images that are different from those already returned
    distance_calculator = BitmapDistanceMemoized(images.astype(np.float32) / 255.)
    while image_indices.size > 0:
        max_distance = 0
        best_index = image_indices[0]
        for index in image_indices:
            distance = distance_calculator.calculate_min_distance_from_others(index, generated_indices)
            if distance > max_distance:
                max_distance = distance
                best_index = index
        generated_indices.append(best_index.item())
        image_indices = image_indices[image_indices != best_index]

        svg_image = bitmap_to_svg(images[best_index])
        yield MNISTMember(svg_image, labels[best_index].item())


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    g = seed_candidate_generator(5)
    fig, axs = plt.subplots(10, 1)
    for j in range(10):
        mbr = next(g)
        print(mbr)
        mbr.to_image(axs[j])
    fig.show()
