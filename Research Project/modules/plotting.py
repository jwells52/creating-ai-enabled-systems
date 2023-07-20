import matplotlib.pyplot as plt

def fsl_plots(dict, title, xlabel='Epochs', ylabel='Loss', save_plot=False, save_path=None):
  fig, ax = plt.subplots()

  plt.title(title)

  for key, values in dict.items():
    ax.plot(values, label=f'{key}-shot')

  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)

  ax.legend()
  if save_plot:
    plt.savefig(save_path)