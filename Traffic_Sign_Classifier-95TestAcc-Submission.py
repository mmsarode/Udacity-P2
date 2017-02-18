
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Build a Traffic Sign Recognition Classifier
# 
# In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 
# 
# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 
# 
# In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.
# 
# The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.
# 
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ---
# ## Step 0: Load The Data

# In[1]:

# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = "train.p"
testing_file = "test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
print(X_train.shape)


# ---
# 
# ## Step 1: Dataset Summary & Exploration
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
# 
# Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

# ### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas

# In[2]:

### Replace each question mark with the appropriate value.

# TODO: Number of training examples
n_train = len(y_train)

# TODO: Number of testing examples.
n_test = len(y_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape[1:]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = 43

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# ### Include an exploratory visualization of the dataset

# Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc.
# 
# The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.
# 
# **NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections.

# In[3]:

import matplotlib.pyplot as plt
import cv2
# import tensorflow as tf
import numpy as np

get_ipython().magic('matplotlib inline')
filename1 = "P2-Images/Traffic-signs-from-web/120_02.jpg"
filename2 = "P2-Images/Traffic-signs-from-web/stop.jpg"
filename3 = "P2-Images/Traffic-signs-from-web/60.jpg"
filename4 = "P2-Images/Traffic-signs-from-web/keep_right.jpg"
filename5 = "P2-Images/Traffic-signs-from-web/ahead_only.jpg"
filename6 = "P2-Images/Traffic-signs-from-web/turn_left.jpg"
filename7 = "P2-Images/Traffic-signs-from-web/school.jpg"

# im1 = cv2.imread(filename1)[::1]
im1 = cv2.imread(filename1)[...,::-1]
im2 = cv2.imread(filename2)[...,::-1]
im3 = cv2.imread(filename3)[...,::-1]
im4 = cv2.imread(filename4)[...,::-1]
im5 = cv2.imread(filename5)[...,::-1]
im6 = cv2.imread(filename6)[...,::-1]
im7 = cv2.imread(filename7)[...,::-1]





# x = LoadImageM(filename1)
# im2 = np.asarray(im)
print(type(im1))
print(im1.shape)
plt.figure(figsize=(1,1))
plt.imshow(im1)
# print(max(im2))


# In[4]:

new_size = (32,32)

resized_image1 = cv2.resize(im1, new_size) 
resized_image2 = cv2.resize(im2, new_size) 
resized_image3 = cv2.resize(im3, new_size) 
resized_image4 = cv2.resize(im4, new_size) 
resized_image5 = cv2.resize(im5, new_size) 
resized_image6 = cv2.resize(im6, new_size) 
resized_image7 = cv2.resize(im7, new_size) 

plt.figure(figsize=(1,1))
plt.imshow(resized_image7)


# In[5]:

### Data exploration visualization goes here.
### Feel free to use as many code cells as needed.
import random
import numpy as np
import matplotlib.pyplot as plt

# import tflearn

# from tflearn.data_preprocessing import ImagePreprocessing
# from tflearn.data_augmentation import ImageAugmentation

# import tflearn.data_augmentation.ImageAugmentation
# Visualizations will be shown in the notebook.
get_ipython().magic('matplotlib inline')

# # image.convertTo(new_image, -1, alpha, beta);
# image2 = image
# cv2.convertScaleAbs(image, image2, alpha, beta)

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.figure(figsize=(1,1))
plt.imshow(image)
# plt.imshow(image2)
print(y_train[index])
print(image.shape)
# print (image)


# ----
# 
# ## Step 2: Design and Test a Model Architecture
# 
# Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# 
# There are various aspects to consider when thinking about this problem:
# 
# - Neural network architecture
# - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
# - Number of examples per label (some have more than others).
# - Generate fake data.
# 
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.
# 
# **NOTE:** The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play!

# ### Pre-process the Data Set (normalization, grayscale, etc.)

# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.

# In[6]:

import cv2
def generate_augmented_data(image, maxrot, maxtrans, maxshear, contrast, brightness):
    
    nrow = 32
    ncol = 32
#     parameters
#     rot = 30 #deg
    rand_alpha = np.random.uniform(-contrast, contrast) + 1
    rand_beta = np.random.uniform(-brightness, brightness) + 0
    
#     image_new = image*rand_alpha + rand_beta

    rand_rot = np.random.uniform(maxrot) - maxrot/2
    
    tr_x = maxtrans * np.random.uniform() - maxtrans/2
    tr_y = maxtrans * np.random.uniform() - maxtrans/2
    
    Rot_matrix = cv2.getRotationMatrix2D((ncol/2,nrow/2),rand_rot,1)
    Trans_matrix = np.float32([[1,0,tr_x],[0,1,tr_y]])
    
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5 + maxshear*np.random.uniform() - maxshear/2
    pt2 = 20 + maxshear*np.random.uniform() - maxshear/2

    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_matrix = cv2.getAffineTransform(pts1,pts2)
    
    image_new = cv2.warpAffine(image, Rot_matrix,(ncol, nrow))
    image_new = cv2.warpAffine(image_new, Trans_matrix,(ncol, nrow))
    image_new = cv2.warpAffine(image_new, shear_matrix,(ncol, nrow))
#     print (rand_alpha, rand_beta)
    image_new = image_new*rand_alpha + rand_beta
    return image_new
    





# In[7]:

aa = generate_augmented_data(image, 15, 4, 5, 0.025, 2)

plt.imshow(aa)


# In[ ]:




# In[8]:

### Preprocess the data here. Preprocessing steps could include normalization, converting to grayscale, etc.
### Feel free to use as many code cells as needed.


# In[9]:

from sklearn.utils import shuffle

print(X_train.shape)
# bb = np.array([generate_augmented_data(img, 40, 5, 5, 1.5, 30) for img in X_train ])
X_train_aug = X_train
y_train_aug = y_train

# bb = np.array([generate_augmented_data(img, 40, 5, 5, 1.5, 30) for img in X_train ])
# X_train_aug = np.concatenate((X_train_aug, bb), axis=0)
# y_train_aug = np.concatenate((y_train_aug, y_train), axis=0) 

for i in range(1):
    bb = np.array([generate_augmented_data(img, 20, 4, 5, 0.025, 2) for img in X_train ])
    X_train_aug = np.concatenate((X_train_aug, bb), axis=0)
    y_train_aug = np.concatenate((y_train_aug, y_train), axis=0) 
#     bb = np.concatenate((bb, X_train), axis=0)
       
    X_train_aug, y_train_aug = shuffle(X_train_aug, y_train_aug)
    

# print(size(bb))
print(X_train_aug.shape)
# print(aa.shape)
# plt.imshow(aa)


# In[10]:

# print(type(bb))
# cc = np.concatenate((bb, X_train), axis=0)
# # bb = np.array([bb, X_train])
# # bb2 = bb + image
# # np.append(bb, [image], 0)
# print(bb.shape)
# print(X_train.shape)
# print(cc.shape)
# plt.imshow(bb[-1])
# plt.imshow(image)


# In[11]:

# Preprocess


# In[12]:

def normalize_01(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    a = -0.5
    b = 0.5
    Xmin = 0
    Xmax = 255
    return a + (image_data - Xmin)*(b - a)*1./(1.*Xmax - Xmin)
#     return [ a + (X - Xmin)*(b - a)*1./(1.*Xmax - Xmin) for X in image_data]


# In[13]:

### Preprocess the data here.
### Feel free to use as many code cells as needed.

X_train_n = normalize_01(X_train_aug)
y_train_n = y_train_aug
X_test_n = normalize_01(X_test)
# X_validation_n = normalize_01(X_validation )
# y_validation_n = y_validation

print(X_train_n[2][5][0])
print(X_train_aug[2][5][0])

# print(X_validation_n[3][5][0], y_validation_n[3])
# print(X_validation[3][5][0], y_validation[3])

# X_train_n, y_train_n = shuffle(X_train_n, y_train_n)


# ### Split Data into Training, Validation and Testing Sets

# In[14]:

### Split the data into training/validation/testing sets here.
### Feel free to use as many code cells as needed.


# In[15]:

from sklearn.model_selection import train_test_split
X_train_n, X_validation_n, y_train_n, y_validation_n = train_test_split(X_train_n, y_train_n, test_size = 0.25, random_state=0)

n_train = len(y_train_n)

n_validation = len(y_validation_n)

print("Updated Image Shape: {}".format(X_train[0].shape))
print("Updated Number of training examples: {}".format(n_train))
print("Updated Number of validation examples: {}".format(n_validation))


# ### Model Architecture

# In[16]:

### Define your architecture here.
### Feel free to use as many code cells as needed.


# In[17]:

import tensorflow as tf

EPOCHS = 10
BATCH_SIZE = 128*8*2
print(X_train[2][5][0])
print(X_train_n[2][5][0])


# In[18]:

### Define your architecture here.
### Feel free to use as many code cells as needed.

from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    keep_prob = 0.5
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 30x30x10.
    
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 23), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(23))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)
    
    
    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    # SOLUTION: Pooling. Input = 28x28x6. Output = 15x15x10.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    # SOLUTION: Layer 2: Convolutional. Output = 12x12x25.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 23, 40), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(40))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)
    
    
    
#     hidden_layer = tf.nn.relu(hidden_layer)
#     hidden_layer = tf.nn.dropout(hidden_layer, keep_prob)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    # SOLUTION: Pooling. Input = 12x12x16. Output = 6x6x25.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(1000, 490), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(490))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)
#     fc1 = tf.nn.dropout(fc1, keep_prob)
        # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1add_W = tf.Variable(tf.truncated_normal(shape=(490, 220), mean = mu, stddev = sigma))
    fc1add_b = tf.Variable(tf.zeros(220))
    fc1add   = tf.matmul(fc1, fc1add_W) + fc1add_b
    
    # SOLUTION: Activation.
    fc1add    = tf.nn.relu(fc1add)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(220, 90), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(90))
    fc2    = tf.matmul(fc1add, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)
#     fc2 = tf.nn.dropout(fc2, keep_prob)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(90, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits


# In[19]:

from tensorflow.contrib.layers import flatten

def GooglNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    keep_prob = 0.5
    
#         32*32 to 30*30
    conv0_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 3, 25), mean = mu, stddev = sigma))
    conv0_b = tf.Variable(tf.zeros(25))
    conv0   = tf.nn.conv2d(x, conv0_W, strides=[1, 1, 1, 1], padding='VALID') + conv0_b
#     20
    
    conv0 = tf.nn.relu(conv0)
        
#         30*30 to 28*28      
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 30x30x10.    
    conv1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 25, 35), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(35))
    conv1   = tf.nn.conv2d(conv0, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
#     30

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)
    
    
    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    # SOLUTION: Pooling. Input = 28x28x6. Output = 15x15x10.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    conv1 = tf.nn.dropout(conv1, keep_prob)
    
#   14*14 to 12*12
    conv2a_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 35, 45), mean = mu, stddev = sigma))
    conv2a_b = tf.Variable(tf.zeros(45))
    conv2a   = tf.nn.conv2d(conv1, conv2a_W, strides=[1, 1, 1, 1], padding='VALID') + conv2a_b
#     40
    
    conv2a = tf.nn.relu(conv2a)
    
    
#   12*12 to 10*10
    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    # SOLUTION: Layer 2: Convolutional. Output = 12x12x25.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 45, 60), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(60))
    conv2   = tf.nn.conv2d(conv2a, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)
#     50
    
    
    
#     hidden_layer = tf.nn.relu(hidden_layer)
#     hidden_layer = tf.nn.dropout(hidden_layer, keep_prob)
#   10*10 to 5*5
    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    # SOLUTION: Pooling. Input = 12x12x16. Output = 6x6x25.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(1500, 1000), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(1000))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
#     1250 800
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)
    
#     fc1 = tf.nn.dropout(fc1, keep_prob/2)
#     fc1 = tf.nn.dropout(fc1, keep_prob)
        # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1add_W = tf.Variable(tf.truncated_normal(shape=(1000, 520), mean = mu, stddev = sigma))
    fc1add_b = tf.Variable(tf.zeros(520))
    fc1add   = tf.matmul(fc1, fc1add_W) + fc1add_b
#     800 420
    # SOLUTION: Activation.
    fc1add    = tf.nn.relu(fc1add)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(520, 250), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(250))
    fc2    = tf.matmul(fc1add, fc2_W) + fc2_b
#     420 200 90 43
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)
    
    fc2b_W  = tf.Variable(tf.truncated_normal(shape=(250, 110), mean = mu, stddev = sigma))
    fc2b_b  = tf.Variable(tf.zeros(110))
    fc2b    = tf.matmul(fc2, fc2b_W) + fc2b_b
    
    # SOLUTION: Activation.
    fc2b    = tf.nn.relu(fc2b)
#     fc2 = tf.nn.dropout(fc2, keep_prob)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(110, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2b, fc3_W) + fc3_b
    
    return logits


# ### Train, Validate and Test the Model

# A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
# sets imply underfitting. A high accuracy on the test set but low accuracy on the validation set implies overfitting.

# In[20]:

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.


# In[21]:

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)


# In[22]:

rate = 0.001

# logits = LeNet(x)
logits = GooglNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)


# In[23]:

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# In[25]:

web_image_array = np.array([normalize_01(resized_image1), normalize_01(resized_image2),                            normalize_01(resized_image3),normalize_01(resized_image4),                           normalize_01(resized_image5),normalize_01(resized_image6),                           normalize_01(resized_image7)])

print(X_train_n.shape)
print(web_image_array.shape)


# In[28]:

from sklearn.utils import shuffle
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train_n)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train_n, y_train_n = shuffle(X_train_n, y_train_n)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_n[offset:end], y_train_n[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_validation_n, y_validation_n)
        training_accuracy = evaluate(X_train_n, y_train_n)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.5f}".format(validation_accuracy))
        print("Training Accuracy = {:.5f}".format(training_accuracy))
        print()

    testing_accuracy = evaluate(X_test_n, y_test)
    print("Testing Accuracy = {:.5f}".format(testing_accuracy))
    
    aa  = sess.run(logits, feed_dict={x: web_image_array })
    print(aa)
    
    saver.save(sess, './P2AWS_web_images_17Feb_normalize_1augmented_15')
    print("Model saved")
    


# In[38]:

print(np.argmax(aa[0])) #right of way at next intersection
print(np.argmax(aa[1])) #road work
print(np.argmax(aa[2])) #slippery road
print(np.argmax(aa[3])) # Keep right
print(np.argmax(aa[4])) # Ahead only
print(np.argmax(aa[5])) # Speed limit 30
print(np.argmax(aa[6])) #	Vehicles over 3.5 metric tons prohibited
# print(aa[0])
plt.imshow(im7)


# In[ ]:

# testing_accuracy = evaluate(X_test_n, y_test)
# print("Training Accuracy = {:.5f}".format(testing_accuracy))


# ---
# 
# ## Step 3: Test a Model on New Images
# 
# To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.
# 
# You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

# ### Load and Output the Images

# In[ ]:

### Load the images and plot them here.
### Feel free to use as many code cells as needed.


# ### Predict the Sign Type for Each Image

# In[ ]:

### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.


# ### Analyze Performance

# In[ ]:

### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.


# ### Output Top 5 Softmax Probabilities For Each Image Found on the Web

# For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 
# 
# The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.
# 
# `tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.
# 
# Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tk.nn.top_k` is used to choose the three classes with the highest probability:
# 
# ```
# # (5, 6) array
# a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
#          0.12789202],
#        [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
#          0.15899337],
#        [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
#          0.23892179],
#        [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
#          0.16505091],
#        [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
#          0.09155967]])
# ```
# 
# Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:
# 
# ```
# TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
#        [ 0.28086119,  0.27569815,  0.18063401],
#        [ 0.26076848,  0.23892179,  0.23664738],
#        [ 0.29198961,  0.26234032,  0.16505091],
#        [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
#        [0, 1, 4],
#        [0, 5, 1],
#        [1, 3, 5],
#        [1, 4, 3]], dtype=int32))
# ```
# 
# Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.

# In[ ]:

### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.


# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the IPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run. You can then export the notebook by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 

# ### Project Writeup
# 
# Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 
