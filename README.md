# e3simple

This is my equivariant graph neural network library. I have one goal only:
- to make the implementation so simple that I can come back to it in a few months and understand how it works

Why am I doing this?

Equivariant Graph Neural Network libraries are pretty complex and not well-explained. I'm doing this so I can learn the math and the minute details.

### What is equivariance? And why do we want it?
- Read this! https://docs.google.com/presentation/d/1ZNM52MDDc183y5j4AIX27NjePoJP1qLnAhYsyKaBzqI/edit?usp=sharing

### How exactly does equivariance work? How does it differ from traditional ML models?

- Read this! https://docs.google.com/presentation/d/1tuhAtmkWthONETgRxBx1pVVXcoYxj4ooa6HvpOHFsVw/edit?usp=sharing

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
- make sure you're using cartesian order in all places (when retrieving the spherical harmonics coefficients, the clebsch-gordan coefficients, and setting the coefficients that the tensor product outputs)
- When getting the clebsch gordan coefficients, check the shape of the matrix you're reading it from. Make sure you're only
reading the coefficients for degrees l1,l2,l3 NOT all the degrees up to l1+l2+l3 (which is a larger matrix).
- make sure you normalize the vectors before you calculate the spherical harmonics coefficients to get the irreps


### things I did to make the implimentation simpler:
- pass one graph at a time into the model to train
- custom message passing scheme


### TODO:
- support equivariant activation functions
- maybe add a channel for the length of 3D features (in addition to getting the L1 tensors for 3D features?)
