import matplotlib.pyplot as plt
import numpy as np

def main():
    print("hallo")

    x = np.linspace(0, 10, 100)

    fig = plt.figure()
    plt.plot(x, np.sin(x))

    fig2 = plt.figure()
    plt.plot(x, np.cos(x))


    fig.add_subplot(121, facecolor='b')
    fig.add_subplot(122, facecolor='g')
    fig.add_subplot(211, facecolor='r')
    fig.add_subplot(212, facecolor='y')

    plt.show()


if __name__ == "__main__":
    main()