import os

import scipy.signal
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class LossHistory():
    def __init__(self, log_dir, val_loss_flag = True):
        import datetime
        self.time_str       = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        self.save_path      = os.path.join(log_dir, "loss_" + str(self.time_str))  
        self.val_loss_flag  = val_loss_flag

        self.losses         = []
        if self.val_loss_flag:
            self.val_loss   = []
        
        os.makedirs(self.save_path)

    def append_loss(self, loss, val_loss = 0):
        self.losses.append(loss)
        with open(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")

        if self.val_loss_flag:
            self.val_loss.append(val_loss)
            with open(os.path.join(self.save_path, "epoch_val_loss_" + str(self.time_str) + ".txt"), 'a') as f:
                f.write(str(val_loss))
                f.write("\n")
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        try:
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, 5 if len(self.losses) < 25 else 15, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
        except:
            pass

        if self.val_loss_flag:
            plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
            try:
                plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, 5 if len(self.losses) < 25 else 15, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
            except:
                pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".png"))

        plt.cla()
        plt.close("all")
