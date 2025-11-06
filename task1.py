import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin

X1 = np.array([-5.0,
               -4.473684210526316,
               -3.947368421052632,
               -3.4210526315789473,
               -2.8947368421052633,
               -2.368421052631579,
               -1.8421052631578947,
               -1.3157894736842106,
               -0.7894736842105265,
               -0.2631578947368425,
               0.2631578947368416,
               0.7894736842105257,
               1.3157894736842106,
               1.8421052631578947,
               2.3684210526315788,
               2.894736842105263,
               3.421052631578947,
               3.947368421052632,
               4.473684210526315,
               5.0])

Y1 = np.array([-3.8214130281395384,
               -3.4193887110325143,
               -2.5661253640159583,
               -1.775098291142789,
               -1.0856313118090484,
               -2.281145922997222,
               -1.0925080217035064,
               0.04196956346189595,
               -0.7073390282943666,
               1.010429892882658,
               2.237370616506767,
               1.9245548728985702,
               2.79729220264899,
               2.3823270467509614,
               4.061315773975142,
               3.0683027990952865,
               3.936954278819055,
               4.568106293943185,
               5.778550917642228,
               6.528404914496181])


X2 = np.array([
    10.0,
    8.0,
    13.0,
    9.0,
    11.0,
    14.0,
    6.0,
    4.0,
    12.0,
    7.0,
    5.0
])

Y2 = np.array([
    [8.04, 9.14, 7.46],
    [6.95, 8.14, 6.77],
    [7.58, 8.74, 12.74],
    [8.81, 8.77, 7.11],
    [8.33, 9.26, 7.81],
    [9.96, 8.1, 8.84],
    [7.24, 6.13, 6.08],
    [4.26, 3.1, 5.39],
    [10.84, 9.13, 8.15],
    [4.82, 7.26, 6.42],
    [5.68, 4.74, 5.73]
])

tol = 10**-6


def func(v: np.ndarray, *args):
    x = args[0]
    y = args[1]
    yhatt = v[0] * x + v[1] * np.ones(np.shape(x)[0])
    kost = np.linalg.norm(yhatt - y)
    return kost


def gen_p(x, y, cofs):
    yhatt = cofs[0] * x + cofs[1] * np.ones(np.size(x))

    A = np.outer(yhatt, yhatt.T)
    B = np.outer(y, yhatt.T)

    P = np.dot(A, np.linalg.pinv(B))

    print("Is projection? :", np.allclose(np.dot(P, P), P, atol=10 ** -1))
    print("Is orthogonal? :", np.allclose(P, P.T))

    return P


print("Data set 1")
cofs1 = fmin(func, x0=np.array([1, 1]), args=(X1, Y1), xtol=tol)
print("Coefficients:", cofs1)  # Coefficients: [0.97754289 1.07934658]

print("Data set 2")
cofs2 = fmin(func, x0=np.array([1, 1]), args=(X2, Y2[:, 0]), xtol=tol)
print("Coefficients:", cofs2)

print("Data set 3")
cofs3 = fmin(func, x0=np.array([1, 1]), args=(X2, Y2[:, 1]), xtol=tol)
print("Coefficients:", cofs3)

print("Data set 4")
cofs4 = fmin(func, x0=np.array([1, 1]), args=(X2, Y2[:, 2]), xtol=tol)
print("Coefficients:", cofs4)

fig, axs = plt.subplots(2, 2)

# top left plot
axs[0, 0].scatter(X1, Y1, color='lightblue', label='y')
axs[0, 0].plot(X1, cofs1[0]*X1+cofs1[1], color='red', label='yhat(fmin)')  # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
axs[0, 0].set_title("Data set 1")

# top right plot
axs[0, 1].scatter(X2, Y2[:, 0], color='lightblue', label='y')  # why do we have to crop Y like this for 2-4 but not 1 me no comprendo
axs[0, 1].plot(X2, cofs2[0]*X2+cofs2[1], color='red', label='yhat(fmin)')
axs[0, 1].set_title("Data set 2")

# bottom left plot
axs[1, 0].scatter(X2, Y2[:, 1], color='lightblue', label='y')
axs[1, 0].plot(X2, cofs3[0]*X2+cofs3[1], color='red', label='yhat(fmin)')
axs[1, 0].set_title("Data set 3")

# bottom right plot
axs[1, 1].scatter(X2, Y2[:, 2], color='lightblue', label='y')
axs[1, 1].plot(X2, cofs4[0]*X2+cofs4[1], color='red', label='yhat(fmin)')
axs[1, 1].set_title("Data set 4")

for ax in axs.flat:
    ax.set(xlabel='x', ylabel='y')

fig.tight_layout()

# second part

P1 = gen_p(X1, Y1, cofs1)
P2 = gen_p(X2, Y2[:, 0], cofs2)
P3 = gen_p(X2, Y2[:, 1], cofs3)
P4 = gen_p(X2, Y2[:, 2], cofs4)

# top left plot
axs[0, 0].scatter(X1, np.dot(P1, Y1), color='blue', marker='+')

# top right plot
axs[0, 1].scatter(X2, np.dot(P2, Y2[:, 0]), color='blue', marker='+')

# bottom left plot
axs[1, 0].scatter(X2, np.dot(P3, Y2[:, 1]), color='blue', marker='+')

# bottom right plot
axs[1, 1].scatter(X2, np.dot(P4, Y2[:, 2]), color='blue', marker='+')

R1 = 2*P1 - np.eye(20)
R2 = 2*P2 - np.eye(11)
R3 = 2*P3 - np.eye(11)
R4 = 2*P4 - np.eye(11)

# top left plot
axs[0, 0].scatter(X1, np.dot(R1, Y1), color='lavender')

# top right plot
axs[0, 1].scatter(X2, np.dot(R2, Y2[:, 0]), color='lavender')

# bottom left plot
axs[1, 0].scatter(X2, np.dot(R3, Y2[:, 1]), color='lavender')

# bottom right plot
axs[1, 1].scatter(X2, np.dot(R4, Y2[:, 2]), color='lavender')

# create surface plot

# data set 1
rand_a = np.linspace(-5, 5, 20)
rand_b = np.linspace(-5, 5, 20)

# err = []
points = np.zeros((20, 20))
for i in range(len(rand_a)):
    for j in range(len(rand_b)):
        err = np.linalg.norm(rand_a[i]*X1[i] + rand_b[j] - Y1[i])
        points[i, j] = err      # (np.array([rand_a[i], rand_b[j], err]))


#ax = plt.subplot(projection = '3d', xlabel = 'a', ylabel = 'b', zlabel='y')
#ax.scatter3D(cofs1[0], cofs1[1], np.array([1]))          # vilket y-värde ska vi ha???
#ax.plot_surface(rand_a, rand_b, points)  # argument must be 2d
plt.show()


