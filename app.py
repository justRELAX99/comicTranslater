import cv2
import base64
import io
import numpy
import os
import csv
from Image import Image
from autocorrect import Speller
spell = Speller(lang='en')

def looper(root_dir):
    file_name_list = []
    file_path_list = []
    for sub_Dir, dirs, files in os.walk(root_dir):
        for file in files:
            file_info = file.split('.')
            file_name, file_exten = file_info[0], file_info[-1]
            file_path = os.path.join(sub_Dir, file)
            if file_exten == 'jpg' or file_exten == 'png' or file_exten == '.bmp':
            # if file_exten != 'zip':
                if file_name not in file_name_list:
                    file_name_list.append(file_name)
                    file_path_list.append(file_path)

    return file_path_list

# append image path and script to the output csv file
def write_script_to_csv(image_path, script, output_file_path):
    with open(output_file_path, 'a', encoding = "utf-8", newline = "") as f:
        writer = csv.writer(f)
        new_row = [image_path, script]
        writer.writerow(new_row)

# Отрегулируйте гамму изображения каким-либо фактором
def adjust_gamma(image, gamma=1.0):
   invGamma = 1.0 / gamma
   table = numpy.array([((i / 255.0) ** invGamma) * 255
      for i in numpy.arange(0, 256)]).astype("uint8")
   return cv2.LUT(image, table)

def app(output_file_path,root_dir):
    # инициализируем выходной файл
    with open(output_file_path, 'w',newline = "") as f:
        writer = csv.writer(f)
        writer.writerow(['filePath', 'script'])
    count=0
    # для каждого изображения в заданном каталоге обработать каждый найденный речевой пузырь и передать его в движок ocr
    for image_path in looper(root_dir):
        print(image_path)
        #path_to_image = 'sample_images/02-2.jpg'
        image=Image(image_path)
        image.segment_image()
        image.parse_bubbles()
        image.paint_over_bubbles((0,255,0))
        image.save_image('test'+str(count))
        write_script_to_csv(image_path, image.text_in_bubbles, output_file_path)
        count+=1

def main():
    app(r'test_data\test.csv','sample_images')

if __name__ == '__main__':
    main()
    