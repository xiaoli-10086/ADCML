from config import get_config
from data_loader import get_test_loader, get_train_loader
from train_independent import Trainer
# from train_DML import Trainer
# from train_independent_other import Trainer
from utils import prepare_dirs, save_config
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


config, unparsed = get_config()
prepare_dirs(config)

train_dataset = get_train_loader(
            config.traindata_dir, config.batch_size,
            config.random_seed, config.input_size, config.shuffle, config.num_workers)

test_dataset = get_test_loader(config.predictdata_dir,config.batch_size,config.input_size,config.num_workers)

data_loader = (train_dataset, test_dataset)

trainer = Trainer(config, data_loader)

trainer.train()
