# import numpy as np
# import cv2
#
# cap = cv2.VideoCapture("./video1.mp4")
#
# v_c = cv2.CascadeClassifier('./haarcascade_car.xml')
#
# while cap.isOpened():
#     ret, frame = cap.read()
#
#     if not ret:
#         break
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     detect = v_c.detectMultiScale(gray, 1.1, 5)
#     for (x, y, w, h) in detect:
#         cv2.rectangle(frame, (x, y), (x + w, h + y), (0, 255, 0), 2)
#         print(x)
#     cv2.imshow('car', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
import cv2
# <---------EX -1 -------------->


# from PIL import Image
# from IPython.display import display
#
# img = Image.open('./download.jpg').convert('RGBA')
# img1 = Image.open('./sun.jpg').convert('RGBA')
#
# img1_r = img1.resize((80, 90))
#
# data = img1_r.getdata()
# new_arr = []
#
# for i in data:
#     if i[:3] == (255, 255, 255):
#         new_arr.append((255, 255, 255, 0))
#     else:
#         new_arr.append(i)
#
# img1_r.putdata(new_arr)
#
# img.paste(img1_r, (60, 50), img1_r)
# img.save('output.png')


# <------------EX - 2 -------------->

# import cv2
# import numpy as np
#
# img= cv2.imread('./skinn.jpg')
#
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# blur = cv2.GaussianBlur(gray, (5,5), 0)
#
# thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,11,10)
#
# result = cv2.inpaint(img,thresh, 3,cv2.INPAINT_TELEA)
#
# cv2.imshow('result', result)
#
# cv2.waitKey(100000)

# <-----------EX - 4 --------------->

# import cv2
# from cvzone.SelfiSegmentationModule import SelfiSegmentation
#
# fg = cv2.imread('./fg.jpg')
# bg = cv2.imread('./bg.jpg')
#
# h=480
# w=640
#
# fg= cv2.resize(fg, (w, h))
# bg= cv2.resize(bg, (w, h))
#
# s=SelfiSegmentation()
# result=s.removeBG(fg,bg,0.3)
#
# cv2.imshow('result', result)
# cv2.waitKey(100000)

# <-------- EX - 5 ----------->

# import numpy as np
#
# img= cv2.imread('./sun.jpg')
#
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# blur = cv2.GaussianBlur(gray, (5, 5), 0)
#
# edge = cv2.Canny(blur, 150, 50)
#
# result,_ = cv2.findContours(edge, cv2.RETR_TREE,  cv2.CHAIN_APPROX_SIMPLE)
#
# f_result= cv2.drawContours(img, result,-1, (0,255,0),2)
#
# cv2.imshow('you', f_result)
# cv2.waitKey(100000)


#<----------------------EX - 7 ---------------->

# import numpy as np
#
# img = cv2.imread('skinn.jpg')
#
# gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
#
# blur = cv2.GaussianBlur(gray, (5,5) ,0)
#
# ret, thresh = cv2.threshold(blur,0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#
# kernel = np.zeros((5,5),np.uint8)
#
# morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#
# contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# result = cv2.drawContours(img, contours , -1, (0,255,0), 2)
#
# cv2.imshow(' ', thresh)
#
# cv2.waitKey(1000000)


# <-----------------EX - 6 ---------------->

# import numpy as np
# import cv2
# img1= cv2.imread('./img1.jpg')
# img2=cv2.imread('./img2.jpg')
#
# orb = cv2.ORB_create()
#
# tkey , tdesc =  orb.detectAndCompute(img1,None)
# qkey, qdesc = orb.detectAndCompute(img2,None)
#
# matcher = cv2.BFMatcher()
# match = matcher.match(tdesc, qdesc)
#
# f_img = cv2.drawMatches(img1, tkey, img2, qkey, match[:20], None)
#
# cv2.imshow('img', f_img)
# cv2.waitKey(10000000)
#

# <-------- EX -  9----------->

# import numpy as np
#
# img = cv2.imread('skinn.jpg')
#
# gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
#
# blur = cv2.GaussianBlur(gray, (5,5) ,0)
#
# ret, thresh = cv2.threshold(blur,0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#
# kernel = np.zeros((5,5),np.uint8)
#
# morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#
# contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# result = cv2.drawContours(img, contours , -1, (0,255,0), 2)
#
# cv2.imshow(' ', result)
#
# cv2.waitKey(1000000)
# <------------------last ex ----------------->

# import cv2
#
# cap = cv2.VideoCapture("video1.mp4")
# v=cv2.CascadeClassifier('haarcascade_car.xml')
#
# while cap.isOpened():
#     ret, frame = cap.read()
#
#     if not ret:
#         break
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     result = v.detectMultiScale(gray, 1.1 , 5)
#
#     for (x,y,w,h) in result:
#         cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
#
#     cv2.imshow('car', frame)
#     cv2.waitKey(100)
#
# cap.release()
#
#
#
