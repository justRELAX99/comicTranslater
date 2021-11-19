import cv2
import numpy
import os
import re
import pytesseract

class Bubble:
    __text_in_bubble=''
    def __init__(self,contour,image):
        self.__contour=contour
        self.__image=image

    @property
    def contour(self):
        return self.__contour

    @property
    def image(self):
        return self.__image

    @property
    def text_in_bubble(self):
        return self.__text_in_bubble
    
    @text_in_bubble.setter
    def text_in_bubble(self, text_in_bubble):
        self.__text_in_bubble=text_in_bubble
    
    def parse_image(self,should_show_image = False):
        image=self.__image
        # Увеличить обрезанное изображение
        image = cv2.resize(image, (0,0), fx = 2, fy = 2)
        # # Denoise
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)

        if should_show_image:
            cv2.imshow('Cropped Speech Bubble', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Передать обрезанное изображение в движок ocr
        text = self.__tesseract(image)

        # Если мы не находим никаких символов, попробуйте уменьшить обрезанную область.
        # Это иногда помогает tesseract распознавать отдельные строки, но увеличивает время обработки.
        count = 0
        while (text == '' and count < 3):
            count+=1
            image = self.__shrink_by_pixels(image,5)
            text = self.__tesseract(image)

        self.__text_in_bubble=text

    # Применит движок ocr к данному изображению и вернет распознанный текст, где лишние символы отфильтрованы
    def __tesseract(self,image,language='eng'):
        # Мы могли бы рассмотреть возможность использования tessedit_char_whitelist, чтобы ограничить распознавание Tesseract.
        # Это ухудшило производительность OCR на практике
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        text = pytesseract.image_to_string(image, lang = language)
        return self.__process_script(text)
    
    # Обработка строки текста на основе некоторых "бизнес-правил"
    def __process_script(self,text):
        # Некоторые современные комиксы имеют эту строку на обложке
        if "COMICS.COM" in text:
            return ''

        # Tesseract иногда воспринимает символы "I" как "|"
        text = text.replace('|','I')
        # Мы хотим, чтобы новые строки были пробелами, чтобы мы могли рассматривать каждый речевой пузырь как одну строку текста
        text = text.replace('\n',' ')
        # Удаляем несколько пробелов из нашей строки
        words = text.split()
        text = ' '.join(words)

        for char in text:
            # Комиксы обычно пишутся заглавными буквами, поэтому мы удаляем все, кроме заглавных букв.
            if char not in ' -QWERTYUIOPASDFGHJKLZXCVBNM,.?!""\'’1234567890':
                text = text.replace(char,'')

        # Эта строка удаляет "-" и объединяет слова, разделенные на две строки
        # Один примечательный крайний случай, который мы здесь не обрабатываем, слова с переносом разделены на две строки
        text = re.sub(r"(?<!-)- ", "", text)
        words = text.split()
        for i in range(0, len(words)):
            #! Добавить проверку правописания всех слов
            
            # Удалите отдельные символы, кроме "I" и "A"
            if len(words[i]) == 1:
                if (words[i] != 'I' and words[i] != 'A'):
                    words[i] = ''

        # Удаляем все повторяющиеся пробелы
        text = ' '.join(words)
        words = text.split()
        final = ' '.join(words)

        # Удаляем все две строки символов, кроме "NO" и "OK"
        if len(final) == 2 and text != "NO" and text != "OK":
            return ''

        return final
    
    # Обрезать изображение, удалив несколько пикселей
    def __shrink_by_pixels(self,image,pixels):
        h = image.shape[0]
        w = image.shape[1] 
        return image[pixels:h-pixels, pixels:w-pixels]