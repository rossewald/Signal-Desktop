import numpy as np
import skimage.io
import skimage.transform
import os
import sys
from caffe2.python import core, workspace, models
import operator
from caffe2.python.models import squeezenet
from PIL import Image


CAFFE_MODELS = os.getcwd()+"/caffe_models"
IMAGE_LOCATION = "MaybeCat.png"
MODEL = 'squeezenet', 'init_net.pb', 'predict_net.pb', 'ilsvrc_2012_mean.npy', 227
mean = 128
INPUT_IMAGE_SIZE = MODEL[4]
INIT_NET = os.path.join(CAFFE_MODELS, MODEL[0], MODEL[1])
PREDICT_NET = os.path.join(CAFFE_MODELS, MODEL[0], MODEL[2])
       
# Read in the raw image data from Image_Raw_Data_Output.txt.
def getRawImageData():
    readImageData = open("Image_Raw_Data_Output.txt","r")
    data_string = readImageData.read()
    readImageData.close()
    return data_string
    
# Format the raw image data into a 2d pixel array.
def formatRawImageData(data_string,image_width):

    # Turn comma-separated string of numbers into a list of ints.
    curr_int_string = ""
    num_list = []
    for indv_char in data_string:
        if indv_char == ",":
            num_list.append(int(curr_int_string))
            curr_int_string = ""
        else: curr_int_string = curr_int_string+indv_char
    initial_line_list = num_list
    
    # Turn the list of ints into a list of pixels, where each pixel is 4 RGBA int values.
    count = 1; this_list = []; whole_list = []
    for value in initial_line_list:
        this_list.append(value)
        if count == 4:
            whole_list.append(this_list)
            this_list = []; count = 0
        count = count + 1
      
    # Sort the pixels into rows.
    width_count = 1; this_pixel_row = []; image_list = []
    for this_pixel in whole_list:
        this_pixel_row.append(this_pixel)
        if width_count == image_width:
            image_list.append(this_pixel_row)
            this_pixel_row = []; width_count = 0
        width_count = width_count + 1
        
    return image_list

# Turn the 2d pixel array into a PNG image.
def createPNGImage(image_list):
    test_image_data = np.array(image_list)
    test_image = Image.fromarray(test_image_data.astype('uint8'))
    test_image.save("MaybeCat.png")
       
# Function to crop the center cropX x cropY pixels from the input image
def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

# Function to rescale the input image to the desired height and/or width. This function will preserve
#   the aspect ratio of the original image while making the image the correct scale so we can retrieve
#   a good center crop. This function is best used with center crop to resize any size input images into
#   specific sized images that our model can use.
def rescale(img, input_height, input_width):
    # Get original aspect ratio
    aspect = img.shape[1]/float(img.shape[0])
    if(aspect>1):
        # landscape orientation - wide image
        res = int(aspect * input_height)
        imgScaled = skimage.transform.resize(img, (input_width, res))
    if(aspect<1):
        # portrait orientation - tall image
        res = int(input_width/aspect)
        imgScaled = skimage.transform.resize(img, (res, input_height))
    if(aspect == 1):
        imgScaled = skimage.transform.resize(img, (input_width, input_height))
    return imgScaled

def getImage():
    # Load the image as a 32-bit float, rescale it to the desired input size, and crop it so we can feed it to our model.
    img = skimage.img_as_float(skimage.io.imread(IMAGE_LOCATION)).astype(np.float32)
    img = rescale(img, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
    img = crop_center(img, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)

    # switch to CHW (HWC --> CHW)
    img = img.swapaxes(1, 2).swapaxes(0, 1)

    # switch to BGR (RGB --> BGR)
    img = img[(2, 1, 0), :, :]

    # remove mean for better results
    img = img * 255 - mean

    # add batch size axis which completes the formation of the NCHW shaped input that we want
    img = img[np.newaxis, :, :, :].astype(np.float32)
    return img
    
def getResults(img):

    # Read the contents of the input protobufs into local variables
    with open(INIT_NET, "rb") as f:
        init_net = f.read()
    with open(PREDICT_NET, "rb") as f:
        predict_net = f.read()

    # Initialize the predictor from the input protobufs
    p = workspace.Predictor(init_net, predict_net)

    # Run the net and return prediction
    results = p.run({'data': img})

    # Turn it into something we can play with and examine which is in a multi-dimensional array
    results = np.asarray(results)
    return results
    
def getPrediction(results):

    # Quick way to get the top-1 prediction result
    # Squeeze out the unnecessary axis. This returns a 1-D array of length 1000
    preds = np.squeeze(results)
    # Get the prediction and the confidence by finding the maximum value and index of maximum value in preds array
    curr_pred, curr_conf = max(enumerate(preds), key=operator.itemgetter(1))
    return curr_pred,curr_conf

# Return True if the ML model predicts that the image is of a cat.
def Is_Pred_Cat_ID(pred):
    # The ID list is comprised of the ImageNet object codes that designate cats. The full list of codes is available here: https://gist.githubusercontent.com/aaronmarkham/cd3a6b6ac071eca6f7b4a6e40e6038aa/raw/9edb4038a37da6b5a44c3b5bc52e448ff09bfe5b/alexnet_codes.
    id_list = [281,282,283,284,285]
    if pred in id_list: return True
    return False

def Is_Cat(img):
    results = getResults(img)
    pred, conf = getPrediction(results)
    if Is_Pred_Cat_ID(pred): return True
    return False
    
def doImagePreprocessing(width):
    image_data=getRawImageData()
    formatted_image_data=formatRawImageData(image_data,IMAGE_WIDTH)
    createPNGImage(formatted_image_data)
    
def returnPrediction():
    img = getImage()
    is_cat=Is_Cat(img)
    if is_cat: print ("True")
    else: print ("False")

IMAGE_WIDTH=int(sys.argv[1])
doImagePreprocessing(IMAGE_WIDTH)
returnPrediction()
