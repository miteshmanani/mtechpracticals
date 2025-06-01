from config import Config
from train import GANTrainer

if __name__ == "__main__":
    config = Config()  # Create an instance of the Config class
    trainer = GANTrainer(config)
    trainer.train()
