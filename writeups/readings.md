
# Background
We are interested in Fermionic system. It could be described via creation and anhilation operators that obeys the cannonical commutation relation. They claim that it is easier to work with Majorana fermions. TODO: why? Under this representation, we could define a basis for all unitaries with Majorana monomial. There are two type of such monomials, one that acts on the odd parity subspace (by fermion occupation number) and etc.

Gaussian unitaries are such that they modify the majorana operator excitation on a single mode into a linear combination of all the modes. This linear combination could be represented as a rotation matrix. We could further decompose this rotation matrix into givens rotation in the subspace spanned by related single modes. 

We could then represent any Gaussian state via two parameters.
$\theta$ the global phase and $R$ the matrix that represents the rotation in the single mode subspace.

There is a necessary and sufficient condition to determine if a mixed state is a Gaussian state. However, it is not clear how to find the constant $K$ and $A$ matrix given the state $\rho$. 

From an arbitrary state, we could compute the covariance matrix by taking the expectation value of two gaussian state from the equation of $\Gamma(U_R \Psi) = R\Gamma(\Psi)R^T$. Computing the expectation value of arbitrary Hamiltonian equals to computing the Pfaffian of a submatrix of the covariance matrix.

In Section 2.7 you should find formulas for state description and measurement probability.
- [Why Fermion's wavefunction are antisymmetric](https://qr.ae/psA0BW)
