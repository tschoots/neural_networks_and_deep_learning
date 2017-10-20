import matplotlib.pyplot as plt
import numpy as np

import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QAction, QLineEdit, QMessageBox
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot


class theApp(QMainWindow):

    def __init__(self, fig, ax):
        super().__init__()
        self.title="PyQt testbox"
        self.left = 10
        self.top = 10
        self.width = 500
        self.height = 140
        self.initUI()
        self.fig = fig
        self.ax = ax

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        #Create textbox
        self.textbox = QLineEdit(self)
        self.textbox.move(20,20)
        self.textbox.resize(380, 40)

        # Create a button in the window
        self.button = QPushButton('Show graph', self)
        self.button.move(20, 75)

        # connect button to funtion on_click
        self.button.clicked.connect(self.on_click)
        self.show()

    @pyqtSlot()
    def on_click(self):
        textboxValue = self.textbox.text()
        #QMessageBox.question(self, 'Message ton', "you typed : " + textboxValue, QMessageBox.Ok, QMessageBox.Ok)
        #self.textbox.setText("")
        x = np.linspace(-5, 5, 100)

        self.ax.plot(x, np.cos(x), '--b')
        self.fig.show()
        #plt.show()

def main():
    print("hallo")

    x = np.linspace(-5,5, 100)

    fig = plt.figure()
    ax = plt.axes()

    ax.plot(x, np.sin(x), '-g')


    plt.legend()



    app = QApplication(sys.argv)
    ex = theApp(fig, ax)
    sys.exit(app.exec_())




if __name__ == "__main__":
    main()