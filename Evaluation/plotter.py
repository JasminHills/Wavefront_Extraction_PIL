import json
import matplotlib.pyplot as plt
def open_metrics(model_name):
    with open(model_name+'/metrics.json') as f:
        file=json.load(f)
    #     print(file)
        fl=f.read()
        trainLoss=file['train_int_loss']
        valLoss= file['val_int_loss']
        lr=file['learning_rate']
        return trainLoss, valLoss, lr

def plot_metric(model_name, plot=False):
    train, val, lr=open_metrics(model_name)
    lngth_train=np.linspace(0, len(train), len(train))
    lngth_val=np.linspace(0, len(val), len(val))
    plt.plot(lngth_train, train, label='Training' )
    plt.plot(lngth_val, val, label='Valuation' )
    plt.ylabel('Loss (Relative)', size=14)
    plt.xlabel('Epochs', size=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.legend(fontsize=14)

    if plot:
        plt.savefig('Trainin1g.svg', bbox_inches='tight')
    plt.show()