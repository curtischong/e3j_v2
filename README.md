# e3j

This is my equivariant graph neural network library. I have one goal only:
- to make the implementation so simple that I can come back to it in a few months and understand how it works

Why am I doing this?

Equivariant Graph Neural Network libraries are pretty complex and not well-explained. I'm doing this so I can learn the math and the minute details.


### What is equivariance? And why do we want it?
- Read this! https://docs.google.com/presentation/d/1ZNM52MDDc183y5j4AIX27NjePoJP1qLnAhYsyKaBzqI/edit?usp=sharing

### How are features represeted?:

Irreps are represented in the same manner as e3x:

 <p align="center">
  <img width="90%" src="https://e3x.readthedocs.io/stable/_images/features_shape_diagram.png" alt="irrep tensor layout"/>
</p>


E3j forbids mixing irreps of different orders. (e3nn is really generous and lets you mix irreps of different orders - but implementing this is harder to understand)

for a given max L, we will always have one l=0 feature, one l=1 feature, one l=2 feature, ..., one l=L feature
- why can't we just have one l=L feature? Because spherical harmonics acts like a fourier transform. the higher level Ls just get you more precision. But we need the lower level Ls as well. 
- e.g. To represent a point in three-dimensional space using spherical harmonics up to degree L, we need 4 coefficients:
- 1 spherical harmonic at l=0
- 3 spherical harmonics at l=1 (but they have diff m: m=-1, 0, 1)


### How to decide what parity?
- for intermediate layers, as long as you have even/odd parity flowing through the layers it's fine
- You just need to make sure that the inputs of your features are the correct parity
    - e.g. if reflecting your input is a completely different shape, ensure you have odd parity. (e.g. tetris tiles)
    - but if reflections don't matter (like the position of inputs to atoms), then you can have even parity
- not too sure about outputs? I don't think it matters, but I haven't done any testing


THESE ARE the parity of the inputs for spherical harmonics that you specify
    Irreps* irreps_sh = irreps_create("1x1o + 1x2e + 1x3o");


### Where are the weights?
- e3nn and e3x offers linear layers where they perform tensor products AND HAVE WEIGHTS inside the tensor product.
    - basically, when you multiply each irrep coefficient with the clebsch gordan coefficients, you also multiply with the corresponding weight
- This makes the tensor product implementation more complicated, and harder to debug
- I'm NOT storing the weights here. instead, the weights will be simple linear layers on the irreps (similar to Allegro's implementation)



### Gotchas I had when implementing
- make sure you're using cartesian or NOT cartesian order for the spherical harmonics coefficients
- When getting the clebsch gordan coefficients, check the shape of the matrix you're reading it from. Make sure you're only
reading the coefficients for degrees l1,l2,l3 NOT all the degrees up to l1+l2+l3 (which is a larger matrix).
- make sure you normalize the vectors before you calculate the spherical harmonics coefficients to get the irreps


### TODO:
- rename features to num channels
    - since features may be confused with the irreps array
- make jax compilation work
- support equivariant activation functions
- maybe add a channel for the length of 3D features (in addition to getting the L1 tensors for 3D features?)