import pathlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sc_gnet import create_googlenet
# from tl_models_util import make_inception_v3_model, make_xception_model
from create_training_data import custom_data_gen, preprocess_img
from pretty_confusion_matrix import pp_matrix_from_data



current_path = pathlib.Path.cwd()
train_data_dir = current_path / "Dataset/Baseline/"
valid_data_dir = current_path / "Dataset/Other batches"
train_data, valid_data, labels, total_images, image_shape = custom_data_gen(train_data_dir, valid_data_dir , n_batch= 64, color_type='gray')

model = create_googlenet(len(labels), image_shape)
# model = make_xception_model(len(labels), image_shape)
model.summary()

EPOCHS = 50
BATCH_SIZE = 64
STEPS_PER_EPOCH = total_images // BATCH_SIZE
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
history = model.fit(
    train_data,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=valid_data, callbacks=[callback])
  
model.save("trained_weights/gray_googlenet_Others_batches.h5")


"""
preform detection
"""
# images_path = train_data.filepaths
# model.load_weights('trained_weights/tl_xception_Others_batches.h5')

# true_labels = train_data.classes
# predicted_labels = []
# for img in images_path:
#     img1, img2 = preprocess_img(img)
#     predictions = model.predict(img1)
#     predicted_class_id = np.argmax(predictions)
#     # predicted_label = labels[predicted_class_id]
#     predicted_labels.append(predicted_class_id)

# with open('Y_pred.txt', 'w') as f:
#     for item in predicted_labels:
#         f.write("%s," % item)
# print(predicted_labels)
# pp_matrix_from_data(true_labels, predicted_labels, labels, cmap='Greens')


"""
Draw Images and resutls
"""

# plt.figure(figsize=(10,7))
# plt.subplots_adjust(hspace=0.5)
# for i in range(6):
#     plt.subplot(2,3,i+1)
#     #preprocess Image to feed image into the model for prediction
#     img1, img2 = preprocess_img(images_path[i])
#     predictions = model.predict(img1)
#     predicted_class_id = np.argmax(predictions)
#     predicted_label = labels[predicted_class_id]
#     true_label_id = train_data.classes
#     t_id = true_label_id[i]
#     true_label = labels[t_id]

#     plt.imshow(img2)
#     plt.title("True_class = {0} \n Prediction = {1}".format(true_label, predicted_label))
#     plt.axis('off')
#     plt.tight_layout
#     plt.suptitle('Model predictions')

# plt.show()