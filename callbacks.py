from transformers import TrainerCallback
import matplotlib.pyplot as plt

class LossHistory(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.epochs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        epoch = logs.get("epoch")
        if "loss" in logs:
            self.train_losses.append(logs["loss"])
            self.epochs.append(epoch)
        if "eval_loss" in logs:
            self.eval_losses.append(logs["eval_loss"])

    def plot(self):
        plt.plot(self.epochs[:len(self.train_losses)], self.train_losses, label='Train Loss')
        plt.plot(self.epochs[:len(self.eval_losses)], self.eval_losses, label='Eval Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Evaluation Loss")
        plt.legend()
        plt.grid(True)
        plt.show()
