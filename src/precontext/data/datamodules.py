class SSTDataModule():

    def __init__(
        self,
        train_infos,
        val_infos,
        transforms,
        use_multimodal,
        batch_size,
        num_workers,
    ):
        super().__init__()
