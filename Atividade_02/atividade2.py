#!/usr/bin/python

# -*- coding: utf-8 -*-

# __author__      = "Matheus Dib, Fabio de Miranda" ==> Modificado
__author__ = "Carlos Dip"


# Imports
import cv2
import numpy as np
from matplotlib import pyplot as plt
# import time
# import auxiliar as aux

# STATE MACHINE TO CONTROL OUTPUT:
# ------------------------------------------------
# ------------------------------------------------
SHOW_BRISK = 0                                #--- Mostra a captura analisada com BRISK
SHOW_BASE = 0                                 #--- Mostra somente a captura direta
SHOW_MAG_MASK = 0                             #--- Mostra a captura da cor magente
SHOW_BLU_MASK = 0                             #--- Mostra a captura da cor azul
SHOW_LINES_DIST = 0                           #--- Mostra os círculos, assim como a linha entre os dois, angulo entre eles e a horizontal(graus), e distância entre eles e a câmera(cm)
SHOW_BITWISE_MAGBLU = 1                       #--- Mostra a junção das máscaras azul e magenta. Não funciona muito bem, mas é interessante.
# ------------------------------------------------
# ------------------------------------------------

# Setup webcam video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Auxiliar variables
lower = 0
upper = 1


focus = 280.0071427660373 * 30 / 14 # Calibração do foco (F = h*d/H) 
# O Foco é calculado através da equação acima, onde "h" é a distância entre os circulo em pixels, 
# medidos numa distância física da câmera conhecida "d", enquanto "H" é a distância real entre os círculos.

magenta = "#F08020"
blue = "#586B7D"
m1,m2 = np.array([172, 50, 50]), np.array([180, 255, 255])
b1,b2 = np.array([80, 50, 50]), np.array([110, 255, 255])

# BRISK
img_logo = cv2.imread("insper_logo.png")
img_logo = cv2.cvtColor(img_logo, cv2.COLOR_BGR2GRAY)
brisk = cv2.BRISK_create()
font = cv2.FONT_HERSHEY_SIMPLEX
fontsize = 2

kplogo, deslogo = brisk.detectAndCompute(img_logo, None) 

MIN_MATCH_COUNT = 10

# Functions

# Canny edge detection
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

# Draw box around BRISK image matches
def find_homography_draw_box(kp1, kp2, img_cena, img_original):
    
    out = img_cena.copy()
    
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)


    # Tenta achar uma trasformacao composta de rotacao, translacao e escala que situe uma imagem na outra
    # Esta transformação é chamada de homografia 
    # Para saber mais veja 
    # https://docs.opencv.org/3.4/d9/dab/tutorial_homography.html
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()


    
    h,w = img_original.shape
    # Um retângulo com as dimensões da imagem original
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

    # Transforma os pontos do retângulo para onde estao na imagem destino usando a homografia encontrada
    dst = cv2.perspectiveTransform(pts,M)


    # Desenha um contorno em vermelho ao redor de onde o objeto foi encontrado
    img2b = cv2.polylines(out,[np.int32(dst)],True,(255,0,0),3, cv2.LINE_AA)
    
    return img2b

# Determines quality of matches
def find_good_matches(descriptor_image1, frame_gray):
    """
        Recebe o descritor da imagem a procurar e um frame da cena, e devolve os keypoints e os good matches
    """
    des1 = descriptor_image1
    kp2, des2 = brisk.detectAndCompute(frame_gray,None)

    # Tenta fazer a melhor comparacao usando o algoritmo
    matches = bf.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    return kp2, good

# Main loop
running = True
while running:
    ret, frame = cap.read()
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # Shape detection using color (cv2.inRange masks are applied over orginal image)
    if SHOW_BLU_MASK or SHOW_MAG_MASK or SHOW_BITWISE_MAGBLU:
        frame_mag = cv2.inRange(frame_hsv,m1,m2)
        frame_blu = cv2.inRange(frame_hsv,b1,b2)

        frame_mag = cv2.morphologyEx(frame_mag,cv2.MORPH_CLOSE,np.ones((4, 4)))
        frame_blu = cv2.morphologyEx(frame_blu,cv2.MORPH_CLOSE,np.ones((4, 4)))
        
        contornos_mag, arvore = cv2.findContours(frame_mag.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
        contornos_blu, arvore = cv2.findContours(frame_blu.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        frame_mag_out = frame.copy()
        frame_blu_out = frame.copy()
        
        cv2.drawContours(frame_mag_out, contornos_mag, -1, [0, 0, 255], 3)
        cv2.drawContours(frame_blu_out, contornos_blu, -1, [0, 0, 255], 3)

        if SHOW_BITWISE_MAGBLU:
            frame_magblu_out = cv2.bitwise_or(frame_blu, frame_mag)
    else:
        frame_mag_out = frame.copy()
        frame_blu_out = frame.copy() 

    # BRISK
    if SHOW_BRISK:
        # Gets descriptor for frame
        kpframe, desframe = brisk.detectAndCompute(frame_gray,None)
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(deslogo,desframe,k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)


        if len(good)>MIN_MATCH_COUNT:   
            framed = find_homography_draw_box(kplogo, kpframe, frame, img_logo)
    else:
        framed = frame.copy()

    if SHOW_LINES_DIST:

        # Canny edge detection
        blur = cv2.GaussianBlur(frame_gray,(7,7),0)
        bordas = auto_canny(blur)
        bordas_color = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR) # Changed to RGB to be able to draw with color on top.

        # Deteccção de círculos (Hough Circles)
        circles = None
        circles = cv2.HoughCircles(bordas,cv2.HOUGH_GRADIENT,2,40,param1=50,param2=100,minRadius=5,maxRadius=60)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            circ_list = []

            for i in circles[0,:]:
                
                # Variáveis auxiliares
                r,g,b = frame[i[1]][i[0]]
                h,s,v = frame_hsv[i[1]][i[0]]
                # print((r,g,b))
                
                # Guarda o centro do círculo atual, e desenha seu centro e contorno
                circ_list.append((i[0],i[1]))
                cv2.circle(bordas_color,(i[0],i[1]),i[2]-3,(int(r),int(g),int(b)),6)
                cv2.circle(bordas_color,(i[0],i[1]),2,(int(r),int(g),int(b)),i[2])


            # Desenha a reta e calcula a distância entre os círculos
            if len(circ_list) >= 2:
                cv2.line(bordas_color, circ_list[0], circ_list[1], (255,255,255), 2) # Linha entre círculos
                dx = np.abs(int(circ_list[0][0]) - int(circ_list[1][0]))
                dy = np.abs(int(circ_list[0][1]) - int(circ_list[1][1]))
                dist = float(np.sqrt(dx**2 + dy**2)) # Distância euclideana em pixels (h)
                real_dist = focus*14/dist # Distância real (D = F*H(14cm)/ h)
                cv2.putText(bordas_color, "Distance: %.2f" %real_dist, (5,fontsize*30), font, fontsize, (255,255,255))
                # print(real_dist)
                angle = np.arctan(dy/dx)*180/np.pi
                cv2.putText(bordas_color, "Angle: %.4f" %angle, (5,fontsize*60), font, fontsize, (255,255,255))
                # print(angle)

    # Display the resulting frame
    if SHOW_BASE:
        cv2.imshow('Detector de circulos',frame)
    elif SHOW_LINES_DIST:
        cv2.imshow('Detector de circulos',bordas_color)
    elif SHOW_MAG_MASK:
        cv2.imshow('Detector de circulos',frame_mag_out)
    elif SHOW_BLU_MASK:
        cv2.imshow('Detector de circulos',frame_blu_out)
    elif SHOW_BRISK:
        cv2.imshow('Detector de circulos',framed)
    elif SHOW_BITWISE_MAGBLU:
        cv2.imshow('Detector de circulos',frame_magblu_out)  
    else:
        cv2.imshow('Detector de circulos',frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
