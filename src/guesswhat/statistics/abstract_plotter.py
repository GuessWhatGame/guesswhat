import os

import matplotlib.pyplot as plt

class AbstractPlotter(object):
    def __init__(self, path, name, suffix):
        self.path = path
        self.name = suffix + "." + name

    def save_as_pdf(self):
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(os.path.join(self.path, self.name+str(".pdf"))) as pdf:
            pdf.savefig()
            plt.close()

    def plot(self):
        plt.plot()