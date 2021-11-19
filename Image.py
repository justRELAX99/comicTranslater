import cv2
import numpy
import os
import pytesseract
import re
from Bubble import Bubble

class Image:
    path_for_save = r'C:\Users\etozh\Desktop\comics translator\comic-book-reader-master\test_data'
    text_in_bubbles=[]
    __bubble_list=[]#лист с пузырями текста

    def __init__(self,path_to_image):
        self.__path_to_image=path_to_image
        self.__image=cv2.imread(path_to_image,1)

    @property
    def path_to_image(self):
        return self.__path_to_image
    
    @property
    def image(self):
        return self.__image
    
    @property
    def bubble_list(self):
        return self.__bubble_list

    @bubble_list.setter
    def bubble_list(self,bubble_list):
        self.__bubble_list=bubble_list
    
    def segment_image(self,should_show_image = False):
        image=self.__image
        bubble_contours = self.__find_speech_bubbles(image)#получаем отсортированные контуры по иерархии
        for bubble_contour in bubble_contours:
            cropped_image = self.__crop_speech_bubbles(bubble_contour)#обрезаем изображение по контурам
            self.__bubble_list.append(Bubble(bubble_contour,cropped_image))
        
        if should_show_image:
            cv2.drawContours(image, bubble_contours, -1, (0, 255, 0), 2)
            cv2.imshow('Speech Bubble Identification', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def parse_bubbles(self):
        text_in_bubbles=[]
        empty_bubbles=[]
        for bubble in self.__bubble_list:
            bubble.parse_image()
            if bubble.text_in_bubble !="" and bubble.text_in_bubble not in text_in_bubbles:
                text_in_bubbles.append(bubble.text_in_bubble)
            else:
                empty_bubbles.append(bubble)

        for empty_bubble in empty_bubbles:
            self.__bubble_list.remove(empty_bubble)
        self.text_in_bubbles=text_in_bubbles
    
    #закрашивает все найденные пузыри изнутри
    def paint_over_bubbles(self,color=(0,0,0),should_show_image = False):
        for bubble in self.__bubble_list:  
            try:
                cv2.fillPoly(self.__image, pts =[bubble.contour], color=color)
            except:
                print('Error while painting')
                
        if should_show_image:
            cv2.imshow('Speech Bubble Identification', self.__image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def save_image(self,file_name):
        cv2.imwrite(os.path.join(self.path_for_save , file_name+'.jpg'), self.__image)

    # Найдет все пузыри речи на данной странице комиксов и вернет список их контуров
    def __find_speech_bubbles(self,image):
        # Преобразовать изображение в оттенки серого
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Распознает прямоугольные / круглые пузыри, борется с темными пузырями 
        binary = cv2.threshold(imageGray,235,255,cv2.THRESH_BINARY)[1]

        # Найдите контуры и задокументируйте их иерархию на будущее
        contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contour_map = {}
        final_contour_list = []

        contour_map = self.__filter_contours_by_size(contours)
        contour_map = self.__filter_containing_contours(contour_map, hierarchy)

        # Сортировать окончательный список контуров
        final_contour_list = list(contour_map.values())
        final_contour_list.sort(key=lambda x:self.__get_contour_precedence(x, binary.shape[1]))

        return final_contour_list
    
    def __filter_contours_by_size(self,contours):
    # Мы могли бы передать это и обновить по ссылке, но я предпочитаю такую ​​«неизменяемую» обработку.
        contour_map = {}

        for i in range(len(contours)):
            # Отфильтровать кандидатов речевого пузыря с необоснованным размером
            if cv2.contourArea(contours[i]) < 120000 and cv2.contourArea(contours[i]) > 4000:
                # Сгладить найденные контуры
                epsilon = 0.0025*cv2.arcLength(contours[i], True)
                approximated_contour = cv2.approxPolyDP(contours[i], epsilon, True)
                contour_map[i] = approximated_contour

        return contour_map
    
    # Иногда контурный алгоритм идентифицирует целые панели, которые уже могут содержать пузыри речи
    # идентифицировано, что заставило нас дважды проанализировать их с помощью OCR. Этот метод пытается удалить контуры, которые
    # содержат контуры других кандидатов речевого пузыря полностью внутри них.
    def __filter_containing_contours(self,contour_map, hierarchy):
        # Мне очень жаль, что не было лучшего способа сделать это, чем удаление всех родителей за O (n ^ 2) в
        # иерархия контура, но с количеством найденных контуров это единственный способ
        # подумайте о том, чтобы сделать это.
        for i in list(contour_map.keys()):
            current_index = i
            while hierarchy[0][current_index][3] > 0:
                if hierarchy[0][current_index][3] in contour_map.keys():
                    contour_map.pop(hierarchy[0][current_index][3])
                current_index = hierarchy[0][current_index][3]

        # Я бы предпочел обрабатывать это "неизменно", как указано выше, но я бы предпочел не делать ненужную копию dict.
        return contour_map

    # Функция сравнения для сортировки контуров
    def __get_contour_precedence(self,contour, cols):
        tolerance_factor = 200
        origin = cv2.boundingRect(contour)
        return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]
    
    # Учитывая список контуров, вернуть список обрезанных изображений на основе ограничивающих прямоугольников контуров
    def __crop_speech_bubbles(self, contour ,padding = 0):
        
        rect = cv2.boundingRect(contour)
        [x, y, w, h] = rect
        cropped_image = self.image[y-padding:y+h+padding, x-padding:x+w+padding]

        return cropped_image
