import streamlit as st
import funct
import cv2 as cv
file1 = st.file_uploader("Select the files in  such a way that second file is watermark file", type="jpg", 'png'], accept_multiple_files=True)
for uploaded_file in file1:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        def ROI():
            opencv_image = cv.imdecode(file_bytes, 1)
            st.image(opencv_image,channels="BGR")
            opencv_image=cv.resize(opencv_image,(800,600))
            r = cv.selectROI(opencv_image)
            imCrop = opencv_image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
            opt = st.selectbox("Select the process option", ("Thresholding",
                               "Gradient", "Morphological transform", "Blur", "Edges"))
            if (opt == "Edges"):
                j1 = st.slider("Select dim1", min_value=0, max_value=1000)
                j2 = st.slider("Select dim2", min_value=0, max_value=1000)
                m = funct.edges(imCrop, j1, j2)
                opencv_image=cv.cvtColor(opencv_image,cv.COLOR_BGR2GRAY)
                opencv_image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]=m
                m=opencv_image
                st.image(m)
            elif (opt == "Thresholding"):
                k2 = st.selectbox("Select threshold object",
                                  ("Global", "Adaptive"))
                k3 = st.slider("Select Ad1", min_value=1,
                               max_value=127, step=2)
                k4 = st.slider("Select Ad2", min_value=1,
                               max_value=255, step=2)
                k5 = st.slider("Select Ad3", min_value=1, max_value=10)
                if (k2 == "Global"):
                    m, m1, m2, m3, m4 = funct.threshold(
                        imCrop, option=k2, adv1=k3, adv2=(k4), adv3=(k5))
                    op=cv.cvtColor(opencv_image, cv.COLOR_BGR2GRAY)
                    op1=cv.cvtColor(opencv_image, cv.COLOR_BGR2GRAY)
                    op2=cv.cvtColor(opencv_image, cv.COLOR_BGR2GRAY)
                    op3=cv.cvtColor(opencv_image, cv.COLOR_BGR2GRAY)
                    op4=cv.cvtColor(opencv_image, cv.COLOR_BGR2GRAY)
                    op[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]=m
                    m=op
                    st.image(cv.resize(m,(3000,4000)), caption="Binary thresh")
                    op1[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]=m1
                    m1=op1
                    st.image(cv.resize(m1,(3000,4000)), caption="Inverse binary")
                    op2[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]=m2
                    m2=op2
                    st.image(cv.resize(m2,(3000,4000)), caption="Truncated")
                    op3[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]=m3
                    m3=op3
                    st.image(cv.resize(m3,(3000,4000)), caption="To zero")
                    op4[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]=m4
                    m4=op4
                    st.image(cv.resize(m4,(3000,4000)), caption="Inverse zero")
                else:
                    m, m1 = funct.threshold(
                        imCrop, adv1=k3, adv2=(k4), adv3=(k5))
                    op=cv.cvtColor(opencv_image, cv.COLOR_BGR2GRAY)
                    op1=cv.cvtColor(opencv_image, cv.COLOR_BGR2GRAY)
                    op[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]=m
                    m=op
                    st.image(cv.resize(m,(3000,4000)), caption="Mean Adaptive thresh")
                    op1[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]=m1
                    m1=op1
                    st.image(cv.resize(m1,(3000,4000)), caption="Gaussian Adaptive thresh")
            elif (opt == "Morphological transform"):
                j1 = st.slider("Select dimension of kernel",
                               min_value=1, max_value=500)
                j2 = st.slider("Select denominator",
                               min_value=1, max_value=256)
                j3 = st.slider("Select value", min_value=0, max_value=80)
                j4 = st.selectbox("Select the option", ("2D FIlt", "erosion", "dilation",
                                  "open morph", "close morph", "morph grad", "topht", "Black"))
                j8 = funct.morph(imCrop, j1, j2, j3, j4)
                opencv_image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]=j8
                j8=opencv_image
                st.image(cv.resize(j8,(3000,4000)), caption=j4, channels="BGR")
            elif (opt == "Gradient"):
                opt89 = st.selectbox(
                    "Select the option", ("Laplacian", "Sobelx", "sobely", "combo"))
                c = st.slider("Select ksize", min_value=1,
                              max_value=31, step=2)
                if (opt89 == "combo"):
                    s1, s2, s3 = funct.gradient(imCrop, c=c)
                    op1=opencv_image
                    op2=opencv_image
                    op3=opencv_image
                    op1[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]=s1
                    st.image(cv.resize(op1,(3000,4000)), clamp=True)
                    op2[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]=s2
                    st.image(cv.resize(op2,(3000,4000)), clamp=True)
                    op3[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]=s3
                    st.image(cv.resize(op3,(3000,4000)), clamp=True)
                    m = st.selectbox(
                        "Select Color", ("HSV", "RGB CIE", "XYZ", "YUV", "YIQ", "YPbPr", "YCbCr", "YDbDr"))
                    l = convert_colorspace(op1, "RGB", m)
                    st.image(cv.resize(l,(3000,4000)), clamp=True)
                    l1 = convert_colorspace(op2, "RGB", m)
                    st.image(cv.resize(l1,(3000,4000)), clamp=True)
                    l2 = convert_colorspace(op3, "RGB", m)
                    st.image(cv.resize(l2,(3000,4000)), clamp=True)
                else:
                    s12 = funct.gradient(imCrop, option=opt89, c=c)
                    opencv_image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]=s12
                    st.image(cv.resize(opencv_image,(3000,4000)), caption=opt89, clamp=True)
                    m = st.selectbox(
                        "Select Color", ("HSV", "RGB CIE", "XYZ", "YUV", "YIQ", "YPbPr", "YCbCr", "YDbDr"))

                    l = convert_colorspace(opencv_image, "RGB", m)
                    st.image(cv.resize(l,(3000,4000)), clamp=True)
            else:
                s1 = st.selectbox(
                    'Option choose', ("Median Blur", "Gaussian blur", "Bilateral", 'blur'))
                s2 = st.slider("Select A", min_value=1, max_value=301, step=2)
                s3 = st.slider("select B", min_value=1, max_value=301, step=2)
                s4 = st.slider("select C", min_value=1, max_value=301, step=2)
                yt = funct.blur(imCrop, s1, s2, s3, s4)
                opencv_image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]=yt
                st.image(cv.resize(opencv_image,(3000,4000)), caption=s1, channels="BGR")
                m = st.selectbox("Select Color", ("HSV", "RGB CIE",
                                 "XYZ", "YUV", "YIQ", "YPbPr", "YCbCr", "YDbDr"))
                l = convert_colorspace(yt, "RGB", m)
                st.image(l, clamp=True)
            cv.waitKey(0)
            cv.destroyAllWindows()
        ROI()
