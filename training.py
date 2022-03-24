import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from sc_g_net_utils import g_net, show_final_history
from create_training_date import custom_data_gen
import pathlib

current_path = pathlib.Path.cwd()
train_data_dir = current_path / "Dataset/Baseline/"
valid_data_dir = current_path / "Dataset/Cross-validation/First three"

train_data, valid_data, labels, in_shape = custom_data_gen(train_data_dir, valid_data_dir, color_type='gray')
print(in_shape)
model = g_net(in_shape, n_cls=train_data.num_classes, desp=True)

history=model.fit(train_data, steps_per_epoch=2, epochs=5, validation_data=valid_data)
show_final_history(history)