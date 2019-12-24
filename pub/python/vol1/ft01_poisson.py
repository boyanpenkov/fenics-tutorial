"""
FEniCS tutorial demo program: Poisson equation with Dirichlet conditions.
Test problem is chosen to give an exact solution at all nodes of the mesh.

  -Laplace(u) = f    in the unit square
            u = u_D  on the boundary

  u_D = 1 + x^2 + 2y^2
    f = -6
"""

from __future__ import print_function
import numpy
import fenics
#fenics.parameters["num_threads"] = 8
import matplotlib.pyplot as plt

# Create mesh and define function space
mesh = fenics.UnitSquareMesh(32, 32)
V = fenics.FunctionSpace(mesh, 'P', 1)

# Define boundary condition
u_D = fenics.Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

def boundary(x, on_boundary):
    return on_boundary

bc = fenics.DirichletBC(V, u_D, boundary)

# Define variational problem
u = fenics.TrialFunction(V)
v = fenics.TestFunction(V)
f = fenics.Constant(-6.0)
a = fenics.dot(fenics.grad(u), fenics.grad(v))*fenics.dx
L = f*v*fenics.dx

# Compute solution
u = fenics.Function(V)
fenics.solve(a == L, u, bc)

# Plot solution and mesh
fenics.plot(u)
fenics.plot(mesh)

# Save solution to file in VTK format
vtkfile = fenics.File('poisson/solution.pvd')
vtkfile << u

# Compute error in L2 norm
error_L2 = fenics.errornorm(u_D, u, 'L2')

# Compute maximum error at vertices
vertex_values_u_D = u_D.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)
error_max = numpy.max(numpy.abs(vertex_values_u_D - vertex_values_u))

# Print errors
print('error_L2  =', error_L2)
print('error_max =', error_max)

# Hold plot
plt.show()
