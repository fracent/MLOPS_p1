import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger  # optional but nice
from mlops_p1.data import corrupt_mnist
from mlops_p1.lightning_model import MyAwesomeModel


def main():
    # Prepare data
    train_set, test_set = corrupt_mnist()
    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=32)

    # Instantiate model
    model = MyAwesomeModel()

    #  Optional: connect to Weights & Biases logger
    wandb_logger = WandbLogger(project="lightning-test")

    # Instantiate Trainer — this is where all flags go
    trainer = Trainer(
        max_epochs=10,              
        limit_train_batches=0.2,    
        logger=wandb_logger,        
        default_root_dir="reports", 
        accelerator="auto",         
    )

    # Train!
    trainer.fit(model, train_dataloader)

    # (optional) test after training
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    main()
