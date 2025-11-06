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
print("\n Addition operation \n",v1+v2)
print("\n Subtraction operation \n", v1-v2)

#scalar multiply 

print("\nScalar multiply for vector 2 with 3 \n", 3*v1)

#dot product 
dot_product = np.dot(v1,v2)
print(f"\n The dot product of {v1} and {v2}: \n ", dot_product)

#cross product 
cross_product = np.cross(v1,v2)
print(f"\nCross product of {v1} and {v2}:\n", cross_product)

#magnitude or Norm(length of a vector i.e magnitude)

norm_v1 = np.linalg.norm(v1)
print("\n Norm of v1 (||v1||)", norm_v1)

unit_v1 = v1/norm_v1
print("\n Unit vector of v1: \n", unit_v1)


#Angle between vectors 

cos_theta = np.dot(v1,v2)/(np.linalg.norm(v1)* np.linalg.norm(v2))
angle_deg = np.degrees(np.arccos(np.clip(cos_theta,-1,1)))
# [-1,1 ] here clamp the value of cos theta between -1 and 1 
print("\n Angle between v1 and v2: \n", round(angle_deg,2),"degrees")



#Matrix creation and basic operations 

print(" \n Matrix Operations \n ")

#Define matrix i.e 2D Array 

A = np.array([[1,2,3],[4,5,6],[7,8,9]])
B = np.array([[9,8,7],[6,5,4],[3,2,1]])

print("\n Matrix A :", A)
print("\n Matrix B :",B)

print("\n Matrix Addition, subtraction, and scalar multiplication \n")
print("\n A+B = \n", A+B)
print("\n A-B = \n ",A-B )
print("\n A*B = \n", np.cross(A,B)) # cross product of two matrices 
print("\n 3*B = \n ",3*B )

#Element wise multiplication 
print("\n A*B = \n", A*B)

#Transpose of matrix 
print("\n Transopose of A = \n", A.T)

#====================================
         #MATRIX PROPERTIES 
#====================================

print("Matrix Properties")

# Matrix properties like determinant, rank, trace and inverse 

M = np.array([[2,1],[5,3]])
print("Matrix M: \n",M)

#Determinant non zero invertible 

det_M = np.linalg.det(M)
print("\n The determinant of M is \n", det_M)

#Rank the number of independant rows or column 
rank_M = np.linalg.matrix_rank(M)
print("\n The rank of the matrix is \n", rank_M)


#Inverse (Only if determanant not equal to 0)
if det_M!=0:
    inv_M = np.linalg.inv(M)
    print("\n Inverse of M:",inv_M)
    
else: 
    print("Matrix is singular no inverse")


#Trace(Sum of diagonal element)

trace_M = np.trace(M)
print("Trace of M: ", trace_M)


#=================================================
# Eigon values and eigon vectors 
#=================================================

print("\n Eigon values and eigon vectors\n")

C = np.array([[4,2],[1,3]])


#Eigon decomposition 

eigen_values, eigen_vectors = np.linalg.eig(C)

print("\n The matrix is \n", C)
print("\n The eigen value is \n", eigen_values), 
print("\n The Eigon vector is ",eigen_vectors)


#Verify A-V = Lambda -V for each pair 

for i in range(len(eigen_values)):
    lhs = np.dot(C,eigen_vectors[:,i])
    rhs = eigen_values[i] * eigen_vectors[:,i]
    
    print(f"\n Eigen pair {i+1}")
    print("A-v =", lhs)
    print("Lambda -v =", rhs)



