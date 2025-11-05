# Linear Algebra for Machine Learning

'''
- Some of the linear operations are:
    - Vectors and Operations 
    - Matrix and Matrix Operations 
    - Determinants, inverse, rank and trace 
    - Eigenvalues and eigenvectors 
    - Vector norms and orthogonality 
    - LU, QR, and SVD Decomposition 
    - Solving system of linear equations(Ax=b)

- Library Used 
    - Numpy - For matrix and vector operation 
    - SciPy - For LU decompositon 
'''
 

import numpy as np 
from scipy.linalg import lu

#1. Vector operations 

print("Vector Oeprations:\n")

#Creating a vector 

v1 = np.array([2,3,4])
v2 = np.array([1,0,-1])

print("\nVector v1 : ", v1)
print("\nVector 2:", v2)

#Vector addition and subtraction 
print("\n Addition operation",v1+v2)
print("\n Subtraction operation", v1-v2)

#scalar multiply 

print("Scalar multiply for vector 2 with 3", 3*v1)

#dot product 
dot_product = np.dot(v1,v2)
print(f"The dot product of {v1} and {v2}: ", dot_product)

#cross product 
cross_product = np.cross(v1,v2)
print(f"Cross product of {v1} and {v2}:", cross_product)

#magnitude or Norm(length of a vector)

norm_v1 = np.linalg.norm(v1)
print("\n Norm of v1 (||v1||)", norm_v1)

unit_v1 = v1/norm_v1
print("Unit vector of v1: ", unit_v1)


#Angle between vectors 

cos_theta = np.dot(v1,v2)/(np.linalg.norm(v1)* np.linalg.norm(v2))
angle_deg = np.degrees(np.arccos(np.clip(cos_theta,-1,1)))
print("\n Angle between v1 and v2:", round(angle_deg,2,"degrees"))