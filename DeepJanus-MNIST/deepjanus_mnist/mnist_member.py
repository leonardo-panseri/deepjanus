import hashlib

from deepjanus.member import Member
from matplotlib import pyplot as plt

from .image_tools import svg_to_bitmap, calculate_bitmap_distance


class MNISTMember(Member):
    """Member for DeepJanus-MNIST"""

    def __init__(self, svg: str, expected_label: int, name: str = None):
        super().__init__(name)
        self.svg = svg
        self.expected_label = expected_label
        self.bitmap = svg_to_bitmap(svg)

        self.predicted_label = None
        self.prediction_quality = None

    def clone(self, name: str = None):
        res = MNISTMember(self.svg, self.expected_label, name)
        res.satisfy_requirements = self.satisfy_requirements
        res.predicted_label = self.predicted_label
        res.prediction_quality = self.prediction_quality
        return res

    def clear_evaluation(self):
        super().clear_evaluation()
        self.predicted_label = None
        self.prediction_quality = None

    def distance(self, other: 'MNISTMember'):
        return calculate_bitmap_distance(self.bitmap, other.bitmap)

    def to_image(self, ax: plt.Axes):
        ax.imshow(self.bitmap, cmap='Greys')

    def to_tuple(self) -> tuple[float, float]:
        # TODO: tuple repr (needed?)
        raise NotImplementedError()

    def to_dict(self) -> dict:
        return {
            'svg': self.svg,
            'expected_label': self.expected_label,
            'satisfy_requirements': self.satisfy_requirements,
            'predicted_label': self.predicted_label,
            'prediction_quality': self.prediction_quality
        }

    @classmethod
    def from_dict(cls, d: dict, name: str = None) -> 'MNISTMember':
        res = MNISTMember(d['svg'],
                          d['expected_label'],
                          name)
        res.satisfy_requirements = d['satisfy_requirements']
        res.predicted_label = d['predicted_label']
        res.prediction_quality = d['prediction_quality']
        return res

    def __eq__(self, other):
        if isinstance(other, MNISTMember):
            return self.svg == other.svg and self.expected_label == other.expected_label
        return False

    def __ne__(self, other):
        if isinstance(other, MNISTMember):
            return self.svg != other.svg or self.expected_label != other.expected_label
        return True

    def member_hash(self):
        return hashlib.sha256((self.svg + str(self.expected_label)).encode('UTF-8')).hexdigest()

    def __str__(self):
        prediction = 'na'
        if self.predicted_label is not None:
            prediction = self.predicted_label
        return f'{super().__str__()} p={prediction}'
