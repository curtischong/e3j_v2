from e3nn.o3._wigner import _so3_clebsch_gordan
from e3x.so3._symbolic import _clebsch_gordan
import e3nn_jax

def verify_coefficents_are_same():
    for l1 in range(2):
        for l2 in range(2):
            for l3 in range(2):
                for m1 in range(-l1, l1 + 1):
                    for m2 in range(-l2, l2 + 1):
                        for m3 in range(-l3, l3 + 1):
                            e3nn_cg = _so3_clebsch_gordan(l1, l2, l3)[m1, m2, m3].tolist()
                            e3nn_jax_cg = e3nn_jax.clebsch_gordan(l1, l2, l3)[m1, m2, m3]
                            # e3x_cg_mat = clebsch_gordan_for_degrees(l1, l2, l3, cartesian_order=True) # this returns the clebsch gordan for all the degrees up to l1+l2+l3. we only need the one for l1+l2+l3
                            e3x_cg = _clebsch_gordan(l1, l2, l3, m1,m2,m3).evalf()
                            # print("e3x_cg_mat shape", e3x_cg_mat.shape)
                            # e3x_cg = e3x_cg_mat[m1 + l1 - (l1**2),m2 + l2 - (l2**2),m3 + l3 - (l3**2)]

                            # print(f"l1={l1}, l2={l2}, l3={l3}, m1={m1}, m2={m2}, m3={m3} cg={cg}")
                            print(e3nn_cg, e3nn_jax_cg, e3x_cg)


def view_coefficients_size():
    l1 = 2
    l2 = 4
    l3 = 6
    print(e3nn_jax.clebsch_gordan(l1, l2, l3).shape)

if __name__ == "__main__":
    view_coefficients_size()