# e3simple

This is my equivariant graph neural network library. I have one goal only:
- to make the implementation so simple that I can come back to it in a few months and understand how it works

### Special Thanks

This repo was heavily inspired by code in [e3nn](https://github.com/e3nn), [e3nn-jax](https://github.com/e3nn/e3nn-jax), [e3nn.c](https://github.com/teddykoker/e3nn.c), and [e3x](https://github.com/google-research/e3x)

### Why am I doing this?

Equivariant Graph Neural Network libraries are pretty complex and not well-explained. I'm doing this so I can learn the math and the minute details.

### What is equivariance? And why do we want it?
- Read this! https://docs.google.com/presentation/d/1ZNM52MDDc183y5j4AIX27NjePoJP1qLnAhYsyKaBzqI/edit?usp=sharing

### How exactly does equivariance work? How does it differ from traditional ML models?

- Read this! https://docs.google.com/presentation/d/1tuhAtmkWthONETgRxBx1pVVXcoYxj4ooa6HvpOHFsVw/edit?usp=sharing

### Gotchas I had when implementing
- make sure you're using cartesian order in all places (when retrieving the spherical harmonics coefficients, the clebsch-gordan coefficients, and setting the coefficients that the tensor product outputs)
- When getting the clebsch gordan coefficients, check the shape of the matrix you're reading it from. Make sure you're only
reading the coefficients for degrees l1,l2,l3 NOT all the degrees up to l1+l2+l3 (which is a larger matrix).
- make sure you normalize the vectors before you calculate the spherical harmonics coefficients to get the irreps


### Things I did to make the implimentation simpler:
- I made custom message passing functions since we are not taking advantage of nice tensors (with consistent shapes during message passing)
  - It also showcases the simplicity of message passing and how we can maintain equivariance at the same time
- There is no batch dimension into the model. We pass one graph at a time into the model to train.


### TODO:
- LinearLayer tests
- Add an equivariance test for 3D outputs
- support adding scalar features as features
