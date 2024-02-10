import re
from xml.etree import ElementTree as ET
from random import choice, getrandbits, uniform

from deepjanus.mutator import Mutator

from .image_tools import svg_to_bitmap, create_svg_xml, calculate_bitmap_distance
from .mnist_member import MNISTMember


class MNISTDigitMutator(Mutator):
    """Mutation strategy for DeepJanus-MNIST members"""

    # Matches all start, end, and middle points of an SVG path
    PATTERN_VERTICES = re.compile('([\d.]+),([\d.]+)\s[MCLZ]')
    # Matches all Bézier curve control points of an SVG path
    PATTERN_CONTROL_POINTS = re.compile('C([\d.]+),([\d.]+)\s([\d.]+),([\d.]+)\s')

    def mutate_point(self, point: tuple):
        """Applies a random displacement to one of the two coordinates of a point."""
        # Mutation is valid if the displaced coordinate is still inside the image
        valid_mutation = False
        mutated_point = point
        while not valid_mutation:
            # Randomly select displacement and sign
            displacement = self.get_random_mutation_extent()

            # Choose which coordinate to mutate
            if getrandbits(1):
                mutated_x = float(point[0]) + displacement
                mutated_point = (str(mutated_x), point[1])
                valid_mutation = 0 <= mutated_x <= 28
            else:
                mutated_y = float(point[1]) + displacement
                mutated_point = (point[0], str(mutated_y))
                valid_mutation = 0 <= mutated_y <= 28

        return mutated_point

    @staticmethod
    def mutate_svg_path(svg_path: str, match_to_mutate: re.Match,
                        mutated_point: tuple, x_group: int, y_group: int):
        """Modifies a SVG path, substituting the mutated point to the original."""
        mutated_path = (svg_path[:match_to_mutate.start(x_group)]
                        + ','.join(mutated_point)
                        + svg_path[match_to_mutate.end(y_group):])
        return mutated_path

    def mutate_vertex(self, svg_path: str):
        """Applies a displacement to one of the two coordinates of a random start, end, or middle point of the path."""
        matches = list(self.PATTERN_VERTICES.finditer(svg_path))

        # Choose a random vertex to mutate
        match_to_mutate = choice(matches)

        # 2 groups: the first is the x and the other is the y
        vertex = match_to_mutate.groups()

        mutated_vertex = self.mutate_point(vertex)
        assert mutated_vertex != vertex

        return self.mutate_svg_path(svg_path, match_to_mutate, mutated_vertex, 1, 2)

    def mutate_control_point(self, svg_path: str):
        """Applies a displacement to one of the two coordinates of a random Bézier curve control point of the path."""
        matches = list(self.PATTERN_CONTROL_POINTS.finditer(svg_path))

        # Choose a random Bézier curve to mutate
        # 4 groups: the first and second are x,y of the first control point,
        # the third and fourth are x,y of the second control point
        match_to_mutate = choice(matches)
        # Choose which control point to mutate
        if getrandbits(1):
            x_group, y_group = 1, 2
        else:
            x_group, y_group = 3, 4
        control_point = (match_to_mutate.group(x_group), match_to_mutate.group(y_group))

        mutated_control_point = self.mutate_point(control_point)
        assert mutated_control_point != control_point

        return self.mutate_svg_path(svg_path, match_to_mutate, mutated_control_point, x_group, y_group)

    def mutate(self, member: MNISTMember):
        """Mutates a DeepJanus-MNIST member: applies a random displacement LB<=|disp|<=UB to a random vertex or Bézier
         curve control point of the SVG path of this member."""
        svg_root: ET.Element = ET.fromstring(member.svg)
        svg_path = svg_root.find('{http://www.w3.org/2000/svg}path').get('d')

        # Randomly select the mutation operator to apply
        if getrandbits(1):
            operator = self.mutate_vertex
        else:
            operator = self.mutate_control_point

        # A mutation is valid when its digit bitmap representation is different from the original
        valid_mutation = False
        mutated_svg = None
        mutated_bitmap = None
        while not valid_mutation:
            mutated_svg_path = operator(svg_path)
            mutated_svg = create_svg_xml(mutated_svg_path)
            mutated_bitmap = svg_to_bitmap(mutated_svg)

            distance = calculate_bitmap_distance(member.bitmap, mutated_bitmap)

            if distance != 0:
                valid_mutation = True

        if mutated_svg is not None and mutated_bitmap is not None:
            member.svg = mutated_svg
            member.bitmap = mutated_bitmap
            member.clear_evaluation()
        else:
            raise Exception('Error during member mutation')


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from .mnist_config import MNISTConfig

    mbr = MNISTMember(
        "<svg version=\"1.0\" xmlns=\"http://www.w3.org/2000/svg\" height=\"28\" width=\"28\"><path d=\" M12,24 C8,23 4,20 5,18 C5,15 8,15 8,17 C8,20 11,21 15,20 C19,19 19,17 14,16 C10,14 9,13 9,10 C9,9 8,8 8,8 C7,8 6,7 6,6 C6,5 7,5 12,5 C19,5 24,6 24,7 C24,8 22,9 20,9 C17,9 15,9 14,10 C12,12 13,14 15,13 C16,13 23,19 23,20 C23,23 16,24 12,24 Z\" /></svg>",
        5)

    plt.imshow(mbr.bitmap, cmap='Greys')
    plt.show()

    cfg = MNISTConfig('.')
    mtr = MNISTDigitMutator(cfg)
    for i in range(100):
        mbr.mutate(mtr)

    plt.imshow(mbr.bitmap, cmap='Greys')
    plt.show()
