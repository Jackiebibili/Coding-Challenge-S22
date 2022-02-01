import dataprep
import training
import graph

def main():
   data = dataprep.prepare_data()
   history = training.train_binary_classification_model(*data)
   graph.graph_result(history.history)

main()
