import pandas as pd
import matplotlib.pyplot as plt


def graph_result(train_history):
    history_df = pd.DataFrame(train_history)
    print(("Best Validation Loss: {:0.5f}" +
           "\nBest Validation Accuracy: {:0.5f}")
          .format(history_df['val_loss'].min(),
                  history_df['val_binary_accuracy'].max()))
    history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
    history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(
        title="Accuracy")
    plt.show()
