from sympy import symbols, expand
import sympy as sp

x, y = symbols('x y')
expr = expand((x + y)**2)

# This would expand to x**2 + 2*x*y + y**2
terms_list = expr.as_ordered_terms()

print(terms_list)

my_term = sp.Poly(sp.S(5) + x + 2.5*y)
# in ((1, 0), 1):
#   (1, 0) indicates the term has degree 1 in x and degree 0 in y
#   1 indicates the coefficient is 1
print(my_term)
print(my_term.terms())
for monomial, coefficient in my_term.terms():
    print("Monomial:", monomial)
    print("Coefficient:", coefficient)
    print()