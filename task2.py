import numpy as np
import matplotlib.pyplot as plt


def recurrence(vec, mat, steps):
    z0 = vec
    for i in range(steps):
        z1 = np.inner(mat, z0)
        z0 = z1
        if not i % 10:
            print(f"After {i+1} iterations:", z0)
    print(f"After {steps} iterations:", z0)
    return z0


def normrecurrence(vec, mat, steps):
    z0 = vec/np.linalg.norm(vec)
    for i in range(steps):
        z1 = np.inner(mat, z0)
        z0 = z1/np.linalg.norm(z1)
        if not i % 10:
            print(f"After {i + 1} iterations:", z0)
    print(f"After {steps} iterations:", z0)
    return z0


def qrecurrence(vec, mat, steps):
    z0 = vec / np.linalg.norm(vec)
    q = np.inner(np.inner(z0, A), z0)
    for i in range(steps):
        z1 = np.inner(mat, z0)
        z0 = z1 / np.linalg.norm(z1)
        q = np.inner(np.inner(z0, A), z0)
        if not i % 10:
            print(f"After {i + 1} iterations:", q)
    print(f"After {steps} iterations:", q)
    return q


def epsilonrecurrence(vec, con_vec, mat, max_steps, ep):
    z0 = vec / np.linalg.norm(vec)
    for i in range(max_steps):
        if np.linalg.norm(z0 - con_vec) < ep:
            return i
        z1 = np.inner(mat, z0)
        z0 = z1 / np.linalg.norm(z1)
    return -1


def epsilonqrecurrence(vec, con_q, mat, max_steps, ep):
    z0 = vec / np.linalg.norm(vec)
    q = np.inner(np.inner(z0, A), z0)
    for i in range(max_steps):
        if abs(q-con_q) < ep:
            return i
        z1 = np.inner(mat, z0)
        z0 = z1 / np.linalg.norm(z1)
        q = np.inner(np.inner(z0, A), z0)
    return -1


a = np.array([8, 3, 12])
b = np.array([1/19, -12/19, 19/19])
c = np.array([1/19, 12/19, -19/19])

A = np.array([[1, 3, 2],
              [-3, 4, 3],
              [2, 3, 1]])


# Regular iterates
print("z0 =", a)
recurrence(a, A, 100)  # diverges
print("-")
print("z0 =", b)  # diverges
recurrence(b, A, 100)
print("-")
print("z0 =", c)  # diverges with eigenvalue -1 for few iterations
recurrence(c, A, 100)

print("-"*20)
# normalized iterations
print("z0 =", a/np.linalg.norm(a))
v1 = normrecurrence(a, A, 1000)  # converges on [0.6882472016116795  0.22941573387059622 0.6882472016116795]
print(v1[0])
print(v1[1])
print(v1[2])
print("-")
print("z0 =", b/np.linalg.norm(b))
v2 = normrecurrence(b, A, 1000)  # converges on [-0.6882472016116675  -0.2294157338706683 -0.6882472016116676]
print(v2[0])
print(v2[1])
print(v2[2])
print("-")
print("z0 =", c/np.linalg.norm(c))
v3 = normrecurrence(c, A, 1000)  # converges on [0.6882472016116683  0.22941573387066477 0.6882472016116683]
print(v3[0])
print(v3[1])
print(v3[2])

print("-"*20)
# q iterations
print("q0 =", np.inner(np.inner(a, A), a))
qrecurrence(a, A, 100)                      # converges to eigenvalue 4
print("-")
print("q0 =", np.inner(np.inner(b, A), b))
qrecurrence(b, A, 100)                      # converges to eigenvalue 4
print("-")
print("q0 =", np.inner(np.inner(c, A), c))
qrecurrence(c, A, 100)                      # converges to eigenvalue 4
print("-")

print("-"*20)
# epsilon
i1 = epsilonrecurrence(a, v1, A, 1000, 10**-8)
print(f"Condition satisfied after {i1} iterations")
print("-")
i2 = epsilonrecurrence(b, v2, A, 1000, 10**-8)
print(f"Condition satisfied after {i2} iterations")
print("-")
i3 = epsilonrecurrence(c, v3, A, 1000, 10**-8)
print(f"Condition satisfied after {i3} iterations")

print("-"*20)
E = np.logspace(-1, -14)
Y1 = np.array([epsilonrecurrence(a, v1, A, 1000, e) for e in E])
Y2 = np.array([epsilonrecurrence(b, v2, A, 1000, e) for e in E])
Y3 = np.array([epsilonrecurrence(c, v3, A, 1000, e) for e in E])

Y4 = np.array([epsilonqrecurrence(a, 4, A, 1000, e) for e in E])
Y5 = np.array([epsilonqrecurrence(b, 4, A, 1000, e) for e in E])
Y6 = np.array([epsilonqrecurrence(c, 4, A, 1000, e) for e in E])

plt.plot(E, Y1, label="v1")
plt.plot(E, Y2, label="v2")
plt.plot(E, Y3, label="v3")
plt.plot(E, Y4, label="q1")
plt.plot(E, Y5, label="q2")
plt.plot(E, Y6, label="q3")

plt.xscale("log")
plt.title("Convergence speed by epsilon")
plt.ylabel("Number of iterations")
plt.xlabel("Epsilon")
plt.legend()

plt.show()

# can see in the plot that v always converges faster than q
