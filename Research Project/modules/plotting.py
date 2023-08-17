import matplotlib.pyplot as plt

def fsl_plots(
        d:dict,
        title:str,
        parameter:str = 'way',
        xlabel:str = 'Epochs',
        ylabel:str='Loss',
        save_plot:bool=False,
        save_path:str=None
    ):
    '''
    Helper function for plotting Learn/Validation curves of various FSL models
    '''
    _, ax = plt.subplots()

    plt.title(title)

    for key, values in d.items():
        ax.plot(values, label=f'{key}-{parameter}')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.legend()
    if save_plot:
        plt.savefig(save_path)
