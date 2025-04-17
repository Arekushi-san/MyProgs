import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy import stats 

# Parameters
Niter = 1000 #dépend de c
n = 100 #dépend de h
c=1
tau=0.1
h=0.3
r = (c*tau/h)**2

# Generate matrix M
M = np.diag((r) * np.ones(n - 1), -1) + np.diag((1 - 2 * r) * np.ones(n)) + np.diag((r) * np.ones(n - 1), 1)

# Generate initial fij values
def Fgenfij(n, Txt, L=1, mu=0.5, sigma=0.1):
    fij = np.zeros(n)
    if Txt == "Tri":
        fij = np.linspace(0, L, n)
        for j in range(n):
            if fij[j] > L / 2:
                fij[j] = L - fij[j]
    if Txt=="Sin":
        fij=np.linspace(0,2*3.14*mu,n)
        fij=np.sin(fij)
    if Txt=="Lin":
        fij=np.linspace(0, 1,n)
        fij[-1]=0
        return fij
    if Txt=="Gauss":
        fij=np.linspace(0, L,n)
        fij = stats.norm.pdf(fij, mu, sigma) 
        print(fij)
        """
        fij=np.linspace(0,1,n)
        fij=(1/(sigma*np.sqrt(2*np.pi)))*np.exp(-((fij-2.5)/sigma)**2*1/2)
        #fij=np.random.normal(2.5, 0.1, n)"""
    return fij

fij = Fgenfij(n, "Gauss", L=5, mu=2.5, sigma=0.05)
ft = fij.copy()
fijm1 = fij.copy()

F = [fij.copy()]
for i in range(1, Niter):
    fij_new = M @ fij - fijm1
    fij_new[0], fij_new[-1] = ft[0], ft[-1]  # Boundary conditions
    F.append(fij_new.copy())
    fijm1, fij = fij, fij_new

F = np.array(F)  # Convert list to NumPy array for consistency

# Animation setup
fig, ax = plt.subplots()
x = np.linspace(0, 1, n)
line, = ax.plot(x, F[0])
ax.set_ylim(F.min(), F.max())

def update(frame):
    line.set_ydata(F[frame])
    return line,

ani = FuncAnimation(fig, update, frames=Niter, interval=100, blit=True)
plt.show()
