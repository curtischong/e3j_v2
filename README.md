# e3j

This is my equivariant graph neural network library. I have one goal only:
- to make the implementation so simple that I can come back to it in a few months and understand how it works

Why am I doing this?

Equivariant Graph Neural Network libraries are pretty complex and not well-explained. I'm doing this so I can learn the math and the minute details.



### The formulation:

To make it simple, all of the feature tensors that are passed around are defined by 3 properties:
- the number of irreps
- the max l of the irreps
- the parity of the irreps

This forbids mixing irreps of different orders. (e3nn is really generous and lets you mix irreps of different orders - but it's more complex)

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