
# coding: utf-8

# # Super Resolution

# ## Imports

# In[1]:


import numpy as np
import sys, getopt
from PIL import Image
from tqdm import tqdm_notebook
import scipy.misc
from skimage import color
from skimage import io
from skimage import transform
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
global_learning_rate = .01
epsRMS = .000001
epsAdam = .000000001
beta1 = .9
beta2 = .999


# ## Conv Net Layer

# In[2]:


class conv_layer(object):
    def __init__(self, kernel_shape, hparameters):
        self.lr = global_learning_rate
        self.pad, self.stride = hparameters['pad'], hparameters['stride']
        self.weight, self.bias = self.initialize(kernel_shape)
        self.v_w, self.v_b = np.zeros(kernel_shape), np.zeros((1,1,1,kernel_shape[-1]))
        self.m_w, self.m_b = np.zeros(kernel_shape), np.zeros((1,1,1,kernel_shape[-1]))
        self.cacheW, self.cacheB = np.zeros(kernel_shape), np.zeros((1,1,1,kernel_shape[-1]))
        self.t = 1
    
    def initialize(self, kernel_shape):
        weight = np.random.randn(3, 3, kernel_shape[2], kernel_shape[3]) * np.sqrt(2.0/kernel_shape[3])
        bias = np.random.randn(1, 1, 1, kernel_shape[3])
        return weight, bias
    
    def update(self, weight, bias, dW, db, vw, vb, lr, momentum=0, weight_decay=0):
        vw_u = momentum*vw - weight_decay*lr*weight - lr*dW
        vb_u = momentum*vb - weight_decay*lr*bias   - lr*db
        weight_u = weight + vw_u
        bias_u   = bias   + vb_u
        return weight_u, bias_u, vw_u, vb_u
    
    def RMSpropUpdate(self, weight, bias, dW, db, cacheW, cacheB, lr, weight_decay=0):
        cacheW += weight_decay * cacheW + (1 - weight_decay) * dW**2
        cacheB += weight_decay * cacheB + (1 - weight_decay) * db**2
        
        weight = weight - lr*dW / (np.sqrt(cacheW) + epsRMS)
        bias = bias - lr*db / (np.sqrt(cacheB) + epsRMS)
        
        return weight, bias, cacheW, cacheB
    
    def AdamUpdate(self, weight, bias, dW, db, m_w, m_b, v_w, v_b, t, lr):
        m_w = m_w * beta1 + (1 - beta1) * dW
        mt_w = m_w / (1 - beta1**t)
        v_w = beta2*v_w + (1 -beta2) * (dW**2)
        vt_w = v_w / (1 - beta2**t)
        
        m_b = m_b * beta1 + (1 - beta1) * db
        mt_b = m_b / (1 - beta1**t)
        v_b = beta2*v_b + (1 -beta2) * (db**2)
        vt_b = v_b / (1 - beta2**t)
        
        weight = weight - lr * mt_w / (np.sqrt(vt_w) + epsAdam)
        bias = bias - lr * mt_b / (np.sqrt(vt_b) + epsAdam)
        
        t += 1
        
        return weight, bias, m_w, m_b, v_w, v_b, t
        
        
        
    def forward_prop(self, input_map):
        output_map, self.cache = self.conv_forward(input_map, self.weight, self.bias, self.pad, self.stride)
        return output_map
    
    def back_prop(self, dZ, momentum, weight_decay, method):
        dA_prev, dW, db = self.conv_backward(dZ, self.cache)
        if method == 'Adam':
            self.weight, self.bias, self.m_w, self.m_b, self.v_w, self.v_b, self.t = self.AdamUpdate(self.weight, self.bias, dW, db, self.m_w, self.m_b, self.v_w, self.v_b, self.t, self.lr)
        elif method == 'RMSprop':
            self.weight, self.bias, self.cacheW, self.cacheB = self.RMSpropUpdate(self.weight, self.bias, dW, db, self.cacheW, self.cacheB, weight_decay) 
        else:
            self.weight, self.bias, self.v_w, self.v_b = self.update(self.weight, self.bias, dW, db, self.v_w, self.v_b, self.lr, momentum, weight_decay)
        return dA_prev
        
    def zero_pad(self, X, pad):
        X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), 'constant', constant_values=(0,0))
        return X_pad
    
    def conv_single(self, a_slice_prev, W, b):
        s = a_slice_prev * W
        Z = np.sum(s)
        Z = Z + float(b)
        return Z
    
    def conv_forward(self, A_prev, W=None, b=None, pad=1, stride=1):
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        (f, f, n_C_prev, n_C) = W.shape
        
        n_H = int(((n_H_prev - f + 2*pad) / stride)) + 1
        n_W = int(((n_W_prev - f + 2*pad) / stride)) + 1
        
        Z = np.zeros((m, n_H, n_W, n_C))
        A_prev_pad = self.zero_pad(A_prev, pad)
        for h in range(n_H):                            
            for w in range(n_W):
                
                A_slice_prev = A_prev_pad[:, h*stride:h*stride+f, w*stride:w*stride+f, :]
                Z[:, h, w, :] = np.tensordot(A_slice_prev, W, axes=([1,2,3],[0,1,2])) + b

        assert(Z.shape == (m, n_H, n_W, n_C))
        cache = (A_prev, W, b, pad, stride)
        return Z, cache
    
    
    
    def conv_backward(self, dZ, cache):
        (A_prev, W, b, pad, stride) = cache
        
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        
        (f, f, n_C_prev, n_C) = W.shape
        
        (m, n_H, n_W, n_C) = dZ.shape
        
        dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
        dW = np.zeros((f, f, n_C_prev, n_C))
        db = np.zeros((1, 1, 1, n_C))

        if pad != 0:
            A_prev_pad = self.zero_pad(A_prev, pad)
            dA_prev_pad = self.zero_pad(dA_prev, pad)
        else:
            A_prev_pad = A_prev
            dA_prev_pad = dA_prev

        for h in range(n_H):                  
            for w in range(n_W):    
                vert_start, horiz_start = h*stride, w*stride
                vert_end,   horiz_end  = vert_start+f, horiz_start+f

                A_slice = A_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end, :]

                dA_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end, :] += np.transpose(np.dot(W, dZ[:, h, w, :].T), (3,0,1,2))

                dW += np.dot(np.transpose(A_slice, (1,2,3,0)), dZ[:, h, w, :])
                db += np.sum(dZ[:, h, w, :], axis=0)

        dA_prev = dA_prev_pad if pad == 0 else dA_prev_pad[:,pad:-pad, pad:-pad, :]

        assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

        return dA_prev, dW, db
    
    


# ## Relu Layer

# In[3]:


class relu(object):
    def relu_forward(self, A_prev):
        Z = np.maximum(A_prev, 0)
        return Z
    
    def relu_backward(self, dZ):
        return np.where(dZ>0, 1, 0)


# ## Loss

# In[4]:


class loss(object):
    def loss_forward(self, image, label_image, mode):
        if mode == 'train':
            self.image = image
            self.label_image = label_image
#             return normalize(label_image - image), np.square(self.image - self.label_image).mean()
            return label_image - image, np.square(self.image - self.label_image).mean()
        if mode == 'test':
            self.image = image
            self.label_image = label_image
            return normalize(image - label_image)
        
    def loss_backward(self):
        return normalize(self.image - self.label_image)


# ## Super Resolution

# In[5]:


class super_resolution2(object):
    def __init__(self):
        kernel_shape = {"C1": (3,3,3,64),
                        "C2": (3,3,64,64),   
                        "C3": (3,3,64,64), 
                        "C4": (3,3,64,64),
                        "C5": (3,3,64,64),
                        "C6": (3,3,64,64),
                        "C7": (3,3,64,64),
                        "OUTPUT": (3,3,64,3)}
        
        
        hparameters = {"pad": 1, "stride": 1}
        
        self.C1 = conv_layer(kernel_shape['C1'], hparameters)
        self.relu1 = relu()
        
        self.C2 = conv_layer(kernel_shape['C2'], hparameters)
        self.relu2 = relu()
        
        self.C3 = conv_layer(kernel_shape['C3'], hparameters)
        self.relu3 = relu()
        
        self.C4 = conv_layer(kernel_shape['C4'], hparameters)
        self.relu4 = relu()
        
        self.C5 = conv_layer(kernel_shape['C5'], hparameters)
        self.relu5 = relu()
        
        self.C6 = conv_layer(kernel_shape['C6'], hparameters)
        self.relu6 = relu()
        
        self.C7 = conv_layer(kernel_shape['C7'], hparameters)
        self.relu7 = relu()
        
        self.C8 = conv_layer(kernel_shape['OUTPUT'], hparameters)
        
        self.L = loss()
        
    def forward(self, image, label_image, mode):
        print('Starting Conv Forward...')
        self.label_image = label_image
        self.C1_FP = self.C1.forward_prop(image)
        self.relu1_FP = self.relu1.relu_forward(self.C1_FP)
        
        self.C2_FP = self.C2.forward_prop(self.relu1_FP)
        self.relu2_FP = self.relu2.relu_forward(self.C2_FP)

        self.C3_FP = self.C3.forward_prop(self.relu2_FP)
        self.relu3_FP = self.relu3.relu_forward(self.C3_FP)
        
        self.C4_FP = self.C4.forward_prop(self.relu3_FP)
        self.relu4_FP = self.relu4.relu_forward(self.C4_FP)
        
        self.C5_FP = self.C5.forward_prop(self.relu4_FP)
        self.relu5_FP = self.relu5.relu_forward(self.C5_FP)
        
        self.C6_FP = self.C6.forward_prop(self.relu5_FP)
        self.relu6_FP = self.relu6.relu_forward(self.C6_FP)
        
        self.C7_FP = self.C7.forward_prop(self.relu6_FP)
        self.relu7_FP = self.relu7.relu_forward(self.C7_FP)
        
        self.C8_FP = self.C8.forward_prop(self.relu7_FP)
        
        loss = self.L.loss_forward(self.C8_FP, (label_image), mode)
        
        return loss
    
    def backward(self, momentum, weight_decay, update_method=None):
        print('Starting Conv Backward...')
        dZ = self.L.loss_backward()
        
        self.C8_BP = self.C8.back_prop(dZ, momentum, weight_decay, update_method)
        
        self.relu7_BP = self.relu7.relu_backward(self.C8_BP)
        self.C7_BP = self.C7.back_prop(self.relu7_BP, momentum, weight_decay, update_method)
        
        self.relu6_BP = self.relu6.relu_backward(self.C7_BP)
        self.C6_BP = self.C6.back_prop(self.relu6_BP, momentum, weight_decay, update_method)
        
        self.relu5_BP = self.relu5.relu_backward(self.C6_BP)
        self.C5_BP = self.C5.back_prop(self.relu5_BP, momentum, weight_decay, update_method)
        
        self.relu4_BP = self.relu4.relu_backward(self.C5_BP)
        self.C4_BP = self.C4.back_prop(self.relu4_BP, momentum, weight_decay, update_method)
        
        self.relu3_BP = self.relu3.relu_backward(self.C4_BP)
        self.C3_BP = self.C3.back_prop(self.relu3_BP, momentum, weight_decay, update_method)
        
        self.relu2_BP = self.relu2.relu_backward(self.C3_BP)
        self.C2_BP = self.C5.back_prop(self.relu2_BP, momentum, weight_decay, update_method)
        
        self.relu1_BP = self.relu1.relu_backward(self.C2_BP)
        self.C1_BP = self.C1.back_prop(self.relu1_BP, momentum, weight_decay, update_method)
        


# ## Image Handling

# In[6]:


def load_image( infilename ) :
    img=mpimg.imread(infilename)
    return img

def save_image( npdata, outfilename ):
    assert_msg = 'Input shall be a HxWx3 ndarray'
    assert isinstance(npdata, np.ndarray), assert_msg
    assert len(npdata.shape) == 3, assert_msg
    assert npdata.shape[2] == 3, assert_msg
    
    rescaled = (255.0 / npdata.max() * (npdata - npdata.min())).astype(np.uint8)

    im = Image.fromarray(rescaled)
    im.save(outfilename)


# ## Main

# In[10]:


def normalize(x):
    return ( x - x.min()) / ( x.max() - x.min())

def main(argv):
    imgName = 'image_0729.jpg'
    mode = 'test'
    dimensions = 32
    try:
        opts, args = getopt.getopt(argv,"hi:m:d",["image=","mode=", "dimensions="])
    except getopt.GetoptError:
        print 'SuperResolution.py -i <image> -m <mode> -d <dimensions>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'SuperResolution.py -i <image> -m <mode> -d <dimensions>'
            sys.exit()
        elif opt in ("-i", "--image"):
             imgName = arg
        elif opt in ("-m", "--mode"):
             mode = arg
        elif opt in ("-m", "--mode"):
             dimensions = arg
    
    
    
    filename = './ass3images/' + imgName
    outfilename = './ass3images/resized-' + imgName
    numOfImages = 1
    img = load_image( filename )
    imgResized = transform.resize(img, (img.shape[0] + 1,img.shape[1] + 1), mode='reflect')
    
    mode = 'test'
    epoches = 14
    momentum = 2
    global_learning_rate = .001
    weight_decay = .9
    SuperResolution = super_resolution2()
    method = 'Adam'
    desiredResolution = dimensions
    numberOfIterations = desiredResolution - img.shape[0]
    SuperResCNNs = []
    
    imgplot= plt.imshow(img)
    plt.show()
    
    img = img.astype('float64') / 255
    
    imgResizedCrop = img[0: img.shape[0], 0:img.shape[0]]

    imgResizedCrop = np.repeat(imgResizedCrop[np.newaxis, :, :, :], numOfImages, axis=0)
    imgOriginal = np.repeat(img[np.newaxis, :, :, :], numOfImages, axis=0)
    
    if desiredResolution <= img.shape[0]:
        print('Error this program is meant to upscale images, please insert a higher desired resolution')
        print('Current image shape: ', img.shape)
        sys.exit()
    if mode == 'train':
        for i in range(numberOfIterations):
            loss = []
            cost = 0
            sub_sample = transform.resize(img, (img.shape[0] - (i + 1),img.shape[1] - (i + 1)), mode='reflect')
            up_sample = transform.resize(sub_sample, (sub_sample.shape[0] + 1, sub_sample.shape[1] + 1), mode='reflect')
            sub_sample = np.repeat(sub_sample[np.newaxis, :, :, :], numOfImages, axis=0)
            up_sample = np.repeat(up_sample[np.newaxis, :, :, :], numOfImages, axis=0)
            epoches = int( up_sample.shape[1] / 2.0 )

            SuperResolution = super_resolution2()
            
            for j in range(epoches):
                print('Epoch: ', j + 1, 'in CNN ', j, ' Start')
                print(imgOriginal.shape)
                print(up_sample.shape)
                print(sub_sample.shape)
                
                imgNext, lossTemp = SuperResolution.forward(up_sample, imgOriginal, mode)  
                
                imgplot= plt.imshow(imgNext)
                plt.show()
                
                imgNext = (imgNext + imgOriginal) / 2
                outImg = imgNext.reshape(up_sample.shape[1],up_sample.shape[2], up_sample.shape[3]) 
                
                loss.append(lossTemp)
                cost += loss[j]

                imgplot= plt.imshow(outImg)
                plt.show()
                
                print('Loss: ',loss[j])
                SuperResolution.backward(momentum, weight_decay, method)
                print('Epoch: ', i  + 1, ' End')
            
            SuperResCNNs.append(SuperResolution)
            imgOriginal = sub_sample
            sub_sample.reshape(sub_sample.shape[1],sub_sample.shape[2], sub_sample.shape[3])
            up_sample.reshape(up_sample.shape[1],up_sample.shape[2], up_sample.shape[3])
            print('Total training cost: ', cost)
            
        imgOriginal = np.repeat(img[np.newaxis, :, :, :], numOfImages, axis=0)
        
        for i in range(numberOfIterations):
            loss = []
            sub_sample = transform.resize(img, (img.shape[0] - (i + 1),img.shape[1] - (i + 1)), mode='reflect')
            up_sample = transform.resize(sub_sample, (sub_sample.shape[0] + 1, sub_sample.shape[1] + 1), mode='reflect')
            sub_sample = np.repeat(sub_sample[np.newaxis, :, :, :], numOfImages, axis=0)
            up_sample = np.repeat(up_sample[np.newaxis, :, :, :], numOfImages, axis=0)
            epoches = int( up_sample.shape[1] / 2.0 )
            
            imgNext, lossTemp = SuperResCNNs[i].forward(up_sample, imgOriginal, mode)
            imgNext = (imgNext + imgOriginal) / 2
            
            outImg = imgNext.reshape(up_sample.shape[1],up_sample.shape[2], up_sample.shape[3])
            imgOriginal = transform.resize(outImg, (outImg.shape[0] - 1, outImg.shape[1] - 1), mode='reflect')
            
            imgOriginal = np.repeat(imgOriginal[np.newaxis, :, :, :], numOfImages, axis=0)
            
            imgplot= plt.imshow(outImg)
            plt.show()
            
            sub_sample.reshape(sub_sample.shape[1],sub_sample.shape[2], sub_sample.shape[3])
            up_sample.reshape(up_sample.shape[1],up_sample.shape[2], up_sample.shape[3])
            
            SuperResCNNs[i].backward(momentum, weight_decay, method)
        mode = 'test'
        with open('model_data.pkl', 'wb') as output:
               pickle.dump(SuperResCNNs, output, pickle.HIGHEST_PROTOCOL)
    
    if mode == 'test':
        if len(SuperResCNNs) == 0:
            with open('model_data.pkl', 'rb') as input_:
                SuperResCNNs = pickle.load(input_)
        
        imgOriginal = transform.resize(img, (img.shape[0] + 1, img.shape[1] + 1), mode='reflect')        
        imgOriginal = np.repeat(imgOriginal[np.newaxis, :, :, :], numOfImages, axis=0)
    
        for i in range(numberOfIterations):
            imgResize = transform.resize(img, (img.shape[0] + numberOfIterations * 2 - ( i + 1), img.shape[1] + numberOfIterations * 2 - ( i + 1)), mode='reflect')
            imgResize = np.repeat(imgResize[np.newaxis, :, :, :], numOfImages, axis=0)
            index = len(SuperResCNNs) - 1 if len(SuperResCNNs) <= i else i
            imgNext = SuperResCNNs[index].forward(imgResize, label_image=imgResize, mode=mode)
            imgNext = (.25 * imgNext + 2 * imgResize)
            imgNext = ( imgNext - imgNext.min()) / ( imgNext.max() - imgNext.min())
        
            imgNext = imgNext.reshape(imgResize.shape[1],imgResize.shape[2], imgResize.shape[3])
            imgResize = imgNext.reshape(imgResize.shape[1],imgResize.shape[2], imgResize.shape[3])
        
        finalImage = imgResize
        normalResize = './ass3images/' +'normal-resize-' + imgName
        save_image(transform.resize(img, (img.shape[0] + 4, img.shape[0] + 4), mode='reflect'), normalResize)
        save_image(finalImage, outfilename)
        imgplot= plt.imshow(finalImage)
        plt.show()
        
    return


if __name__ == "__main__":
   main(sys.argv[1:])


