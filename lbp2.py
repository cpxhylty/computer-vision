import cv2
import numpy as np
from matplotlib import pyplot as plt
import load_dataset as ld

def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

def lbp_calculated_pixel(img, x, y):
    '''

     64 | 128 |   1
    ----------------
     32 |   0 |   2
    ----------------
     16 |   8 |   4    

    '''    
    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x-1, y+1))     # top_right
    val_ar.append(get_pixel(img, center, x, y+1))       # right
    val_ar.append(get_pixel(img, center, x+1, y+1))     # bottom_right
    val_ar.append(get_pixel(img, center, x+1, y))       # bottom
    val_ar.append(get_pixel(img, center, x+1, y-1))     # bottom_left
    val_ar.append(get_pixel(img, center, x, y-1))       # left
    val_ar.append(get_pixel(img, center, x-1, y-1))     # top_left
    val_ar.append(get_pixel(img, center, x-1, y))       # top
    
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val    

def show_output(output_list):
    output_list_len = len(output_list)
    figure = plt.figure()
    for i in range(output_list_len):
        current_dict = output_list[i]
        current_img = current_dict["img"]
        current_xlabel = current_dict["xlabel"]
        current_ylabel = current_dict["ylabel"]
        current_xtick = current_dict["xtick"]
        current_ytick = current_dict["ytick"]
        current_title = current_dict["title"]
        current_type = current_dict["type"]
        current_plot = figure.add_subplot(1, output_list_len, i+1)
        if current_type == "gray":
            current_plot.imshow(current_img, cmap = plt.get_cmap('gray'))
            current_plot.set_title(current_title)
            current_plot.set_xticks(current_xtick)
            current_plot.set_yticks(current_ytick)
            current_plot.set_xlabel(current_xlabel)
            current_plot.set_ylabel(current_ylabel)
        elif current_type == "histogram":
            current_plot.plot(current_img, color = "black")
            current_plot.set_xlim([0,260])
            current_plot.set_title(current_title)
            current_plot.set_xlabel(current_xlabel)
            current_plot.set_ylabel(current_ylabel)            
            ytick_list = [int(i) for i in current_plot.get_yticks()]
            current_plot.set_yticklabels(ytick_list,rotation = 90)

    plt.show()
    
def calculate_lbp(img_bgr):
    #image_file = 'lenna.jpg'
    #img_bgr = cv2.imread(image_file)
    img_r = img_bgr[0].reshape(32, 32, 1)
    img_g = img_bgr[1].reshape(32, 32, 1)
    img_b = img_bgr[2].reshape(32, 32, 1)
    img_bgr = np.concatenate((np.concatenate((img_r, img_g), axis=2), img_b), axis=2)
    #print(img_bgr.shape)
    #plt.imshow(img_bgr)
    #plt.show()显示图片
    height, width, channel = img_bgr.shape
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    #img_lbp = np.zeros((height, width,3), np.uint8)
    img_lbp_vector = np.zeros((1, 256), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
             #img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
            img_lbp_vector[0][lbp_calculated_pixel(img_gray, i, j)] += 1
    return img_lbp_vector
    #print("LBP Program is finished")

examples_processed = 1
x_train, y_train, file_names = ld.load_CIFAR_100("D:/picture-data/cifar-100-python/train")
lbp_matrix = np.zeros((examples_processed, 256))
for i in range(examples_processed):
    image = x_train[i]
    lpb_vector = calculate_lbp(image)
    lbp_matrix[i][:] = lpb_vector
print(lbp_matrix)
#with open("D:/picture-data/cifar-100-python/lbp-data.txt", 'ab')as abc:
    #np.savetxt(abc, lbp_matrix, delimiter=",")
