# import datasets, classifiers and performance metrices
from  sklearn import datasets,svm
import matplotlib.pyplot as plt

# the digits dataset
digits = datasets.load_digits()
print "digits : ", digits.keys()

print " digits.target------- :",digits.target

images_and_labels = list(zip(digits.images, digits.target))
print " len (images_and_labels) " , len(images_and_labels)

for index, [image,label] in enumerate(images_and_labels[:5]):
    print "index :", index, "image : \n ", image, "label :", label
    plt.subplot(2,5,index+1)
    plt.axis('on')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')         # image processing  grayscale
    plt.title('Training : %i ' % label)



# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples,feature) matrix:
n_samples= len(digits.images)
print "n_sample : ",n_samples
imageData = digits.images.reshape((n_samples, -1))

print "after reshaped : len(imagedata[0]) :",len(imageData[0])

# create a classifier : a support vector classifier
classifier = svm.SVC(gamma=0.001)

# we learn the digits on the first half of the digits
classifier.fit(imageData[  : n_samples//2] ,
               digits.target[ : n_samples//2])

# now predict the value of the digit on the second half:
expected= digits.target[n_samples //2 :]
predicted = classifier.predict(imageData[n_samples //2 :])  # testing data

images_and_predictions = list(zip(digits.images[n_samples // 2 :],predicted))
for index,[image , prediction] in enumerate(images_and_predictions[:5]):
    plt.subplot(2, 5, index+6)
    plt.axis('off')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title('Prediction: %i'% prediction)
print "original values : ",digits.target[n_samples // 2: (n_samples//2)+5 ]



# install pillow library
from scipy.misc import imread,imresize,bytescale
img = imread("Three2.jpeg")
img = imresize(img,(8,8))
img = img.astype(digits.images.dtype) # conversion of the input image to the given data

img= bytescale(img, high=16.0 , low=0) # to represent ebery pixel to max 16 not more than that

print "img :",img
x_testData =[]

for c in img:
    for r in c:
        x_testData.append(sum(r)/3.0) # since  r is 1 pixel every point is made up of three colors
print "x_testData :" , x_testData

x_testData = [x_testData]
print  "len(x_testData) :",len(x_testData)
print "machine learning output :", classifier.predict(x_testData)
plt.show()