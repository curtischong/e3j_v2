# for a single set of poin clouds (one training example), convert it into a graph
from o3.irrep import Irreps


# used when we want to aggregate all of the irreps coming from neighbouring nodes (right after when the messages are passed)
def avg_irreps_with_same_id(irreps_list: list[Irreps]) -> Irreps:
    ids = [irreps.id() for irreps in irreps_list]
    assert (
        len(set(ids)) == 1
    ), f"All ids for irreps in irreps_list MUST be the same (so we can avg them). ids: {ids}"

    # the first irreps will be where we store the averaged irreps
    summed_data = irreps_list[0].data()
    for irreps in irreps_list[1:]:
        for idx, irrep in enumerate(irreps.irreps):
            summed_data[idx] += irrep.data

    for i in range(len(summed_data)):
        summed_data[i] /= len(irreps_list)
        # summed_data[i] *= math.sqrt(1 / len(irreps_list)) # this doesn't lower equivariance error :'(
    return Irreps.from_id(irreps_list[0].id(), summed_data)
