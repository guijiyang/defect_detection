class Config():
    """ configuration for train and test """
    batch_size=1
    learning_rate=1e-3
    max_epochs=30
    image_size=None

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

class detectConfig(Config):
    batch_size=2
    learning_rate=1e-4
    max_epochs=30
    adjust_iter=3
    lr_decay=0.8
    image_size=(512, 512)

class ClassifyConfig(Config):
    batch_size=8
    learning_rate=0.001
    max_epochs=30
    weight_decay=5e-5
    momentum=0.9
    image_size=(227,227)
    num_classes=4