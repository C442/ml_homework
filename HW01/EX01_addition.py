import numpy as np

b = -1
x = np.array([1,1,1])
w = np.array([1,-2, 3])
d_x = 0

#calculating distance d(x), formula is given
numerateur = abs(np.dot(w, x) + b)
denominateur = np.linalg.norm(w)
d_x = numerateur/denominateur

print(np.linalg.norm(d_x))

#calculating the orthogonal projection(closest dist to plane) of x onto the decision boundary

# https://math.libretexts.org/Bookshelves/Linear_Algebra/Interactive_Linear_Algebra_%28Margalit_and_Rabinoff%29/06%3A_Orthogonality/6.03%3A_Orthogonal_Projection

proj = x - (numerateur/np.dot(w,w) * w)
print(proj)

#answer to d
#https://medium.com/@abhaysingh71711/support-vector-machine-svm-algorithm-fc0d3595de4c
#https://medium.com/@abhaysingh71711/support-vector-machine-svm-algorithm-fc0d3595de4c
