import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Paramètres de la figure
fig, ax = plt.subplots()
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)
line, = ax.plot(x, y)

# Fonction d'initialisation
def init():
    line.set_ydata(np.sin(x))
    return line,

# Fonction de mise à jour
def update(frame):
    y = np.sin(x + frame * 0.1)  # Déplace la sinusoïde vers la droite
    line.set_ydata(y)
    return line,

# Création de l'animation
ani = FuncAnimation(
    fig, update, frames=np.arange(0, 100, 1), init_func=init, blit=True, interval=50
)

# Sauvegarde de l'animation en MP4
filename="sinusoidal_animation"
writer = FFMpegWriter(fps=24, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
#ani.save(f"{filename}.mp4", writer=writer, dpi=200)
#ani.save("sinusoidal_animation.mp4", writer=writer, dpi=200)

#ani.save(f"{filename}.mp4", writer=FFMpegWriter(fps=24, codec='libx264', extra_args=['-pix_fmt', 'yuv420p']), dpi=200)

# Affichage de l'animation
plt.show()
