class Trainer:
    def __init__(self, model, dataset_name, model_type):
        ...
    def fit(self, train_loader, val_loader):
        ...

# run.py - volá to takto:
trainer = Trainer(mlp_model, dataset="tess",  model_type="mlp")
trainer = Trainer(cnn_model, dataset="wesad", model_type="cnn")