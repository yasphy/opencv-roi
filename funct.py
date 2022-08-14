import cv2 as cv
import numpy as np
import streamlit as st
def roi(img):
    img=cv.resize(img,(900,700))
    r=cv.selectROI(img)
    roi_cp=img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    return roi_cp
    #cv.imshow("ROI", roi_cp)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
@st.cache()
def threshold(img,option=None,adv1=None,adv2=None,adv3=None):
    img=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if ((option=="Global") and (adv1==adv2==adv3==None)):
        ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
        ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
        ret,thresh3 = cv.threshold(img,125,255,cv.THRESH_TRUNC)
        ret,thresh4 = cv.threshold(img,124,225,cv.THRESH_TOZERO)
        ret,thresh5 = cv.threshold(img,125,255,cv.THRESH_TOZERO_INV)
        #cv.imshow("Binary thresh", thresh1)
        #cv.imshow("Inverse binary",thresh2)
        #cv.imshow("Truncated",thresh3)
        #cv.imshow("To zero",thresh4)
        #cv.imshow("Inverse zero",thresh5)
        #cv.waitKey(0)
        #cv.destroyAllWindows()
        return thresh1,thresh2,thresh3,thresh4,thresh5
    elif (option=="Global") and (adv1!=None) and adv2!=None and adv3!=None:
        ret,thresh1 = cv.threshold(img,adv1,adv2,cv.THRESH_BINARY)
        ret,thresh2 = cv.threshold(img,adv1,adv2,cv.THRESH_BINARY_INV)
        ret,thresh3 = cv.threshold(img,adv1,adv2,cv.THRESH_TRUNC)
        ret,thresh4 = cv.threshold(img,adv1,adv2,cv.THRESH_TOZERO)
        ret,thresh5 = cv.threshold(img,adv1,adv2,cv.THRESH_TOZERO_INV)
        #cv.imshow("Binary thresh", thresh1)
        #cv.imshow("Inverse binary",thresh2)
        #cv.imshow("Truncated",thresh3)
        #cv.imshow("To zero",thresh4)
        #cv.imshow("Inverse zero",thresh5)
        #cv.waitKey(0)
        #cv.destroyAllWindows()
        return thresh1,thresh2,thresh3,thresh4,thresh5
    elif (adv1!=None) and adv2!=None and adv3!=None:
        th2 = cv.adaptiveThreshold(img,adv1,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,adv2,adv3)
        th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,adv2,adv3)
        return th2,th3
    else:
        th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)
        th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)
        #cv.imshow("Mean Adaptive thresh",th2)
        #cv.imshow("Gaussian adaptive thresh",th3)
        return th2,th3
@st.cache()
def gradient(img,option=None,c=6):
    #img=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if (option=="Laplacian"):
        l = cv.Laplacian(img,cv.CV_64F)
        #cv.imshow(option,l)
        #cv.waitKey(0)
        #cv.destroyAllWindows()
        return l
    elif (option=="Sobelx"):
        l = cv.Sobel(img,cv.CV_64F,1,0,ksize=c)
        #cv.imshow(option,l)
        #cv.waitKey(0)
        #cv.destroyAllWindows()
        return l
    elif (option=="sobely"):
       l = cv.Sobel(img,cv.CV_64F,0,1,ksize=c)
       #cv.imshow(option,l)
       #cv.waitKey(0)
       #cv.destroyAllWindows()
       return l
    elif (option=="sobelx8u"):
        l = cv.Sobel(img,cv.CV_8U,1,0,ksize=c)
        #cv.imshow(option,l)
        #cv.waitKey(0)
        #cv.destroyAllWindows()
        return l
    else:
    # Output dtype = cv.CV_64F. Then take its absolute and convert to cv.CV_8U
        l = cv.Sobel(img,cv.CV_64F,1,0,ksize=c)
        abs_sobel64f = np.absolute(l)
        sobel_8u = np.uint8(abs_sobel64f)
        #cv.imshow("SOME SObel",l)
        #cv.imshow("S1",abs_sobel64f)
        #cv.imshow("S2",sobel_8u)
        #cv.waitKey(0)
        #cv.destroyAllWindows()
        return l,abs_sobel64f,sobel_8u
@st.cache()
def blur(img,option=None,a=6,b=7,c=8):
    if (option=="Median Blur"):
        m=cv.medianBlur(img, (a))
        #cv.imshow(option,m)
        return m
        #cv.waitKey(0)
        #cv.destroyAllWindows()
    elif (option=="Gaussian blur"):
        m=cv.GaussianBlur(img,(a,b),c)
        #cv.imshow(option,m)
        #cv.waitKey(0)
        #cv.destroyAllWindows()
        return m
    elif option=="Bilateral":
        blur = cv.bilateralFilter(img,a,b,c)
        #cv.imshow(option,blur)
        return blur
    else:
        m=cv.blur(img,(a,b))
        #cv.imshow("Blur",m)
        return m
        #cv.waitKey(0)
        #cv.destroyAllWindows()
@st.cache()
def edges(img,dim1=None,dim2=None):
    if (dim1==None) or dim2==None:
        c=cv.Canny(img,100,2.0)
        #cv.imshow("Canny trf",c)
        return c
        #cv.waitKey(0)
        #cv.destroyAllWindows()
    else:
        c=cv.Canny(img,dim1,dim2)
        return c
@st.cache()
def morph(img,dim,kt,do=None,option=None):
    kernel = np.ones((dim,dim),np.float32)/kt
    if (option=="2D FIlt") and (do!=None):
        dst = cv.filter2D(img,-do,kernel)
        #cv.imshow(option,dst)
        return dst
    elif (option=="2D FIlt"):
        dst = cv.filter2D(img,-1,kernel)
        #cv.imshow(option,dst)
        return dst
    elif (option=="erosion"):
        kernel = np.ones((dim,dim),np.uint8)
        dst = cv.erode(img,kernel,iterations = 1)
        #cv.imshow(option+str(np.random.random()),dst)
        return dst
    elif (option=="dilation"):
        dst = cv.dilate(img,kernel,iterations = 1)
        #cv.imshow(option+str(np.random.random()),dst)
        return dst
    elif (option=="open morph"):
        dst= cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
        #cv.imshow(option+str(np.random.random()),dst)
        return dst
    elif (option=="close morph"):
        dst= cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
        #cv.imshow(option,dst)
        return dst
    elif (option=="morph grad"):
        dst = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
        #cv.imshow(option,dst)
        return dst
    elif (option=="topht"):
        dst = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
        #cv.imshow(option,dst)
        return dst
    else:
        dst = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)
        #cv.imshow("Black",dst)
        return dst
@st.cache()
def stackImages(imgArray,scale,lables=[]):
    sizeW= imgArray[0][0].shape[1]
    sizeH = imgArray[0][0].shape[0]
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv.resize(imgArray[x][y], (sizeW, sizeH), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv.cvtColor( imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv.resize(imgArray[x], (sizeW, sizeH), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv.FILLED)
                cv.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver