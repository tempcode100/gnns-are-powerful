import numpy as np

S = np.array([[0,1,1,0,1,0], [1,0,1,0,0,1],[1,1,0,1,0,0],[0,0,1,0,1,1],[1,0,0,1,0,1],
[0,1,0,1,1,0]])

S_hat = np.array([[0,1,0,1,0,1], [1,0,1,0,1,0],[0,1,0,1,0,1], [1,0,1,0,1,0],[0,1,0,1,0,1], [1,0,1,0,1,0]])

N = S.shape[0]

L1, U1 = np.linalg.eig(S)
L2, U2 = np.linalg.eig(S_hat)

one = np.ones((N,1))
print("Eigenvalue and eigenvector information for G")
print(np.round(L1,3))
print(np.round(one.T@U1,3))
print("Eigenvalue and eigenvector information for G_hat")
print(np.round(L2,3))
print(np.round(one.T@U2,3))

h = [10,1,-1/2, 1/3, -1/4, 1/5]
Z = np.zeros((N,N))
tmp = np.eye(N)
for i in range(len(h)):
    Z += h[i] * tmp
    tmp = tmp @ S
    

y1 = np.round(np.diagonal(Z),2)
print("GNN output for G")
print('y = ',y1)

Z_hat = np.zeros((N,N))
tmp = np.eye(N)
for i in range(len(h)):
    Z_hat += h[i] * tmp
    tmp = tmp @ S_hat
    


y1_hat = np.round(np.diagonal(Z_hat),2)
print("GNN output for G_hat")
print('y_hat = ',y1_hat)
