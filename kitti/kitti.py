from .registry import DATASETS
from .custom import CustomDataset
from .video_custom import VideoCustomDataset


@DATASETS.register_module
class KittiDataset(CustomDataset):
    CLASSES = ('Car', 'Van', 'Truck', 'Pedestrian', 'Person', 'Cyclist', 'Tram', 'Misc',)
    def __init__(self, **kwargs):
        super(KittiDataset, self).__init__(**kwargs)

@DATASETS.register_module
class KittiVideoDataset(VideoCustomDataset):
    CLASSES = ('Car', 'Van', 'Truck', 'Pedestrian', 'Person', 'Cyclist', 'Tram', 'Misc',)
    def __init__(self, **kwargs):
        super(KittiVideoDataset, self).__init__(**kwargs)