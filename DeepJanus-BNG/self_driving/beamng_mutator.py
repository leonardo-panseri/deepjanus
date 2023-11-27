import random

from core.mutator import Mutator
from self_driving.beamng_member import BeamNGMember
from self_driving.catmull_rom import catmull_rom


class BeamNGRoadMutator(Mutator):
    """Mutation strategy for DeepJanus-BNG members"""

    NUM_UNDO_ATTEMPTS = 20

    def __init__(self, lower_bound: int, upper_bound: int):
        self.lower_bound = lower_bound
        self.higher_bound = upper_bound

    def mutate(self, member: BeamNGMember):
        """Mutates a DeepJanus-BNG member: picks a random road control node (excluding the first
        and last three) and adds a random value in the range [lower_bound,upper_bound]. If the road sampled from the
        new control nodes is not valid, tries modifying a different control node."""
        original_nodes = tuple(member.control_nodes)

        genes_to_mutate = list(range(3, len(original_nodes) - 3))

        def next_gene_index() -> int:
            if not genes_to_mutate:
                return -1
            gene = random.choice(genes_to_mutate)
            genes_to_mutate.remove(gene)
            return gene

        def find_valid_mutate(index: int) -> bool:
            coord_index, mut_value = self.mutate_gene(member, index)

            attempt = 0

            is_mutation_valid = member.is_valid()
            while not is_mutation_valid and attempt < BeamNGRoadMutator.NUM_UNDO_ATTEMPTS:
                self.undo_mutation(member, index, coord_index, mut_value)
                coord_index, mut_value = self.mutate_gene(member, index)
                attempt += 1
                is_mutation_valid = member.is_valid()
            return is_mutation_valid

        gene_index = next_gene_index()

        while gene_index != -1:
            if find_valid_mutate(gene_index):
                break
            else:
                gene_index = next_gene_index()

        if gene_index == -1:
            raise ValueError("No gene can be mutated")

        assert member.control_nodes != original_nodes

    def mutate_gene(self, member: BeamNGMember, index: int, xy_prob=0.5) -> tuple[int, int]:
        gene = list(member.control_nodes[index])
        # Choose the mutation extent
        mut_value = random.randint(self.lower_bound, self.higher_bound)
        # Avoid to choose 0
        if mut_value == 0:
            mut_value += 1
        coord_index = 0
        if random.random() < xy_prob:
            coord_index = 1
        gene[coord_index] += mut_value
        member.control_nodes[index] = tuple(gene)
        member.sample_nodes = catmull_rom(member.control_nodes, member.num_spline_nodes)
        return coord_index, mut_value

    @classmethod
    def undo_mutation(cls, member: BeamNGMember, index: int, coord_index: int, mut_value: int):
        gene = list(member.control_nodes[index])
        gene[coord_index] -= mut_value
        member.control_nodes[index] = tuple(gene)
        member.sample_nodes = catmull_rom(member.control_nodes, member.num_spline_nodes)
