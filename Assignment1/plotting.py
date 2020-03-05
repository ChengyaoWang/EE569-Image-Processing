import numpy, csv, collections
import matplotlib.pyplot as plt

class ee569_hw1_plotting():
    x_axis = [x for x in range(256)]
    def __init__(self):
        self.dataDic = collections.defaultdict(list)
        order = ['red_pdf', 'red_tf', 'green_pdf', 'green_tf', 'blue_pdf', 'blue_tf']
        with open('output.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter = ',')
            for rowPointer, rowName in zip(csv_reader, order):
                for item in rowPointer[:-1]:
                    self.dataDic[rowName].append(int(item))

    def tf_plot(self):
        for channel in ['red', 'green', 'blue']:
            plt.plot(self.x_axis, self.dataDic[channel + '_tf'], c = channel)
        plt.legend(['R_before', 'G_before', 'B_before'], loc = 'lower right')
        plt.xlim((-1, 256))
        plt.ylim((-1, 256))
        plt.title('Empirical Transfer Function (CDF) of R/G/B')
        plt.savefig('tf.png', dpi = 800)
        plt.close()


    def pdf_plot(self):
        for channel in ['red', 'green', 'blue']:
            plt.plot(self.x_axis, self.dataDic[channel + '_pdf'], c = channel)
        plt.legend(['R_before', 'G_before', 'B_before'], loc = 'upper right')
        plt.xlim((-1, 256))
        plt.title('Empirical PDF of R/G/B')
        plt.savefig('pdf.png', dpi = 800)
        plt.close()
    
    def cdf_plot(self):
        # CDF Before Op
        for channel in ['red', 'green', 'blue']:



            plt.plot(self.x_axis, [x * 875 for x in self.dataDic[channel + '_tf']], c = channel)
        plt.legend(['R_before', 'G_before', 'B_before'], loc = 'lower right')
        plt.xlim((-1, 256))
        plt.ylim((-1, 224100))
        plt.title('Cumulative Histogram of Before Op')
        plt.savefig('cdf_beforeOp.png', dpi = 800)
        plt.close()
        # CDF After Op
        for channel in ['red', 'green', 'blue']:
            plt.plot(self.x_axis, [875 * x for x in range(256)], c = channel)
        plt.legend(['R_After', 'G_After', 'B_After'], loc = 'lower right')
        plt.xlim((-1, 256))
        plt.ylim((-1, 224100))
        plt.title('Cumulative Histogram of After Op')
        plt.savefig('cdf_afterOp.png', dpi = 800)
        plt.close()

        

        



if __name__ == '__main__':
    dummy = ee569_hw1_plotting()
    dummy.tf_plot()
    dummy.pdf_plot()
    dummy.cdf_plot()