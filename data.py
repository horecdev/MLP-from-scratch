class TinyShakespeareDataset:
    def __init__(self, path, seq_len, batch_size):
        with open(path, "r") as file:
            self.text = file.read()
            
        self.seq_len = seq_len
        self.batch_size = batch_size
        
        