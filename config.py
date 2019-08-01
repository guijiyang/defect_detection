class Config():
    """ configuration for train and test """
    batch_size=32
    learning_rate=0.0005
    max_epochs=30
    adjust_iter=3
    lr_decay=0.8
    image_size=512

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
