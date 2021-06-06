'''
predict.py有几个注意点
1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
2、如果想要保存，利用r_image.save("img.jpg")即可保存。
3、如果想要获得框的坐标，可以进入detect_image函数，读取top,left,bottom,right这四个值。
4、如果想要截取下目标，可以利用获取到的top,left,bottom,right这四个值在原图上利用矩阵的方式进行截取。
'''
import tensorflow as tf
from PIL import Image

from frcnn import FRCNN

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

frcnn = FRCNN()

while True:
    img = input('Input image filename:')
    # img = 'img/street.jpg'
    try:
        image = Image.open(img)  # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x480 at 0x1C61B0CA308>
        # print(image)
        # -------------------------------------#
        #   转换成RGB图片，可以用于灰度图预测。
        # -------------------------------------#
        image = image.convert("RGB")  # <PIL.Image.Image image mode=RGB size=640x480 at 0x18179EEC848>
        # print(image)
        # image.show()
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = frcnn.detect_image(image)

        r_image.save('output/'+img)
        r_image.show()
