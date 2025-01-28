import torch

class ModelConfig():
    def __init__(self, 
                 name_model="mBERTA",
                 num_epochs=10,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 batch_size=32,
                 gradient_accumulation=1,
                 learning_rate=1e-5,
                 dropout=[0.25],
                 mix_precision=True,
                 max_length=250):
        
        self.name_model = name_model
        self.num_epochs = num_epochs
        self.device = device
        self.batch_size = batch_size
        self.gradient_accumulation = gradient_accumulation
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.mix_precision = mix_precision
        self.max_length = max_length

        self.all_labels = None
        self.classes = ['descriptive', 'direct', 'non-offensive', 'offensive', 'reporting']


# Initialize configuration
config = ModelConfig()