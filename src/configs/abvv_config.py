from configs.base import Config as BaseConfig


class Config(BaseConfig):
    # Base
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.add_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_args(self, **kwargs):
        self.batch_size = 1
        self.num_epochs = 1  # Quick test run to verify functionality

        self.loss_type = "CrossEntropyLoss"

        self.checkpoint_dir = "working/checkpoints/ABVV"

        self.model_type = "MemoCMT"

        self.text_encoder_type = "bert"  # [bert, roberta]
        self.text_encoder_dim = 768
        self.text_unfreeze = False

        self.audio_encoder_type = "hubert_base"
        self.audio_encoder_dim = 768
        self.audio_unfreeze = False

        self.fusion_dim: int = 768

        # Dataset
        self.data_name: str = "IEMOCAP"  # Using IEMOCAP loader for ABVV (same format)
        self.data_root: str = "/Users/puchudexter/Research/MemoCMT/ABVV_preprocessed"
        self.data_valid: str = "val.pkl"
        self.text_max_length: int = 297
        self.audio_max_length: int = 128000

        # Number of classes for ABVV
        self.num_classes: int = 4

        # Config name
        self.name = (
            f"{self.model_type}_{self.text_encoder_type}_{self.audio_encoder_type}_ABVV"
        )

        for key, value in kwargs.items():
            setattr(self, key, value)

