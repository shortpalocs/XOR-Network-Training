import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from PIL.Image import init

from training import confidence_history
from training import train_step







# Set up the figure


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(10,6))
    fig.canvas.manager.set_window_title("XOR Network Training")

    labels = ['[0,0]', '[0,1]', '[1,0]', '[1,1]']
    colors = ['red', 'green', 'blue', 'orange']

    # Line for each of the 4 x inputs

    lines = []
    for i in range(4):
        line, = ax.plot([], [], color=colors[i], label=labels[i], linewidth=3)
        lines.append(line)


    # Setup the plot


    # Target lines

    ax.axhline(y=0, color='yellow', linestyle='--', label="0 (False)")
    ax.axhline(y=1, color='yellow', linestyle='--', label="1 (True)")


    ax.set_xlim(0, 10000) # X lim
    ax.set_ylim(-0.1, 1.1) # Y lim
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Confidence (Accuracy)")
    ax.set_title("Xor Network Learning")
    ax.grid(True)
    ax.legend(loc='best')


    def animate(frame):
        # Each time animate func gets called, you train neural ten times. (AKA 10 steps per frame)
        for i in range(30):
            train_step()



        data = list(range(len(confidence_history[0])))
        for i, line in enumerate(lines):
            line.set_data(data, confidence_history[i])

            # data is x, iteration number
            # confidence history is y axis

        ax.set_title(f"Iteration: {len(confidence_history[0])}")
        return lines



    def init():
        for line in lines:
            line.set_data([], [])
        return lines


    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=11000, interval=1)



    plt.show()











