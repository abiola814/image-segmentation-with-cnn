# Update CUDA for TF 2.5
!wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcudnn8_8.1.0.77-1+cuda11.2_amd64.deb
!dpkg -i libcudnn8_8.1.0.77-1+cuda11.2_amd64.deb
# Check if package has been installed
!ls -l /usr/lib/x86_64-linux-gnu/libcudnn.so.*
# Upgrade Tensorflow
!pip install --upgrade tensorflow==2.5.0

!nvidia-smi



!unzip cnn.zip
import sys
sys.path.append("/content/maskrcnn_colab/mrcnn_demo")
from m_rcnn import *
from visualize import random_colors, get_mask_contours, draw_mask
%matplotlib inline

# Extract Images
images_path = "dataset.zip"
annotations_path = "v-name.json"

extract_images(os.path.join("/content/",images_path), "/content/dataset")

dataset_train = load_image_dataset(os.path.join("/content/", annotations_path), "/content/dataset", "train")
dataset_val = load_image_dataset(os.path.join("/content/", annotations_path), "/content/dataset", "val")
class_number = dataset_train.count_classes()
print('Train: %d' % len(dataset_train.image_ids))
print('Validation: %d' % len(dataset_val.image_ids))
print("Classes: {}".format(class_number))


# Load image samples
display_image_samples(dataset_train)


# Load Configuration
config = CustomConfig(class_number)
# config.display()
model = load_training_model(config)

# Start Training
# This operation might take a long time.
train_head(model, dataset_train, dataset_train, config)

# Load Test Model
# The latest trained model will be loaded
test_model, inference_config = load_test_model(class_number)

# Test on a random image
test_random_image(test_model, dataset_val, inference_config)
