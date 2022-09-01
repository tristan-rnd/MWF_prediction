import matplotlib.pyplot as plt
import torch


def samples(loader, norm):
    T1_T2, MWF = next(iter(loader))
    fig_trained, ax_trained = plt.subplots(1, 3, figsize=(10, 15))
    if norm == 0:
        fig_trained.suptitle("T1, T2 and MWF")
    else:
        fig_trained.suptitle("T1, T2 normalized and MWF")

    fig_trained.tight_layout()
    fig_trained.subplots_adjust(top=0.92)

    ax_trained[0].imshow(T1_T2[0][0], cmap="gray")
    ax_trained[0].set_title("T1")

    ax_trained[1].imshow(T1_T2[0][1], cmap="gray")
    ax_trained[1].set_title("T2")

    ax_trained[2].imshow(MWF[0][0], cmap="gray")
    ax_trained[2].set_title("MWF")
    if norm == 0:
        plt.savefig('Samples_display.pdf')
    else:
        plt.savefig('Normalized_samples_display.pdf')
    return


def loss(train_loss, validation_loss, epochs):
    fig_acc, ax_acc = plt.subplots()
    ax_acc.plot(range(0, epochs), train_loss, label="Train data")
    ax_acc.plot(range(0, epochs), validation_loss, label="validation data")
    ax_acc.set_title("Loss over epochs")
    ax_acc.set_xlabel("Epochs")
    ax_acc.set_ylabel("Loss")
    ax_acc.legend()
    ax_acc.text(0.8, 0.2, "Final loss: " + "{:.10f}".format(validation_loss[-1]), horizontalalignment='center',
                verticalalignment='center', transform=ax_acc.transAxes)
    plt.savefig('Loss.pdf')


def prediction(net, loader, device="cuda", nb=8):
    T1_T2, MWF = next(iter(loader))
    fig_trained, ax_trained = plt.subplots(nb, 4, figsize=(10, 15))
    fig_trained.suptitle("MWF created from T1 & T2")

    fig_trained.tight_layout()
    fig_trained.subplots_adjust(top=0.92)

    for i in range(nb):
        if device == "cuda":
            output = net(T1_T2[i].to(device).type(torch.cuda.FloatTensor).unsqueeze(dim=0)).squeeze().cpu().detach()
        else:
            output = net(T1_T2[i].to(device).float().unsqueeze(dim=0)).item()

        ax_trained[i][0].imshow(T1_T2[i][0], cmap="gray")
        ax_trained[i][0].set_title("T1")

        ax_trained[i][1].imshow(T1_T2[i][1], cmap="gray")
        ax_trained[i][1].set_title("T2")

        ax_trained[i][2].imshow(MWF[i][0], cmap="gray")
        ax_trained[i][2].set_title("True MWF")

        ax_trained[i][3].imshow(output, cmap="gray")
        ax_trained[i][3].set_title("Predicted MWF")
    plt.savefig('Prediction.pdf')
    plt.show()

    return output