"""Result visualization utilities."""
import matplotlib.pyplot as plt


def plot_history(history):
    plt.figure()
    plt.plot(history.get('loss', []), label='loss')
    plt.plot(history.get('val_loss', []), label='val_loss')
    plt.legend()
    plt.show()
