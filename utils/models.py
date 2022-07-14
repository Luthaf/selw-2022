import numpy as np
from equistore import Labels, TensorBlock, TensorMap


def sum_over_structures(tensor):
    """Sum over the per-atom representation & create a per-structure
    representation"""

    new_blocks = []
    for _, block in tensor:
        unique_structures = np.unique(block.samples["structure"])

        new_values = np.zeros((len(unique_structures),) + block.values.shape[1:])
        new_values[block.samples["structure"]] += block.values

        new_samples = Labels(
            ["structure"], values=unique_structures.view(dtype=np.int32).reshape(-1, 1)
        )

        new_block = TensorBlock(
            new_values, new_samples, block.components, block.properties
        )

        if block.has_gradient("positions"):
            gradient = block.gradient("positions")

            n_properties = gradient.data.shape[-1]
            # new shape is n_structures * n_atom_per_structures * n_neighbor_per_atom * xyz * properties
            gradient_data = gradient.data.reshape(
                len(unique_structures), 3, 3, 3, n_properties
            )

            new_data = np.sum(gradient_data, axis=1).reshape(
                3 * len(unique_structures), 3, n_properties
            )

            unique_structure_atoms = np.unique(gradient.samples[["structure", "atom"]])
            new_samples = Labels(
                ["sample", "structure", "atom"],
                values=np.array(
                    [
                        [structure_i, structure_i, atom_i]
                        for structure_i, atom_i in unique_structure_atoms
                    ],
                    dtype=np.int32,
                ),
            )

            new_block.add_gradient(
                "positions", new_data, new_samples, gradient.components
            )

        new_blocks.append(new_block)

    return TensorMap(tensor.keys, new_blocks)
