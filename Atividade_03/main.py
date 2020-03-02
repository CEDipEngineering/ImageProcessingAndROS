#!/usr/bin/python

# -*- coding: utf-8 -*-

# __author__      = "Matheus Dib, Fabio de Miranda" ==> Modificado
__author__ = "Carlos Dip"


# Imports
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

# STATE MACHINE TO CONTROL OUTPUT:
# ------------------------------------------------
# ------------------------------------------------
SHOW_BRISK = 0                                #--- Mostra a captura analisada com BRISK
SHOW_BASE = 1                                 #--- Mostra somente a captura direta
SHOW_MAG_MASK = 0                             #--- Mostra a captura da cor magente
SHOW_BLU_MASK = 0                             #--- Mostra a captura da cor azul
SHOW_LINES_DIST = 0                           #--- Mostra os círculos, assim como a linha entre os dois, angulo entre eles e a horizontal(graus), e distância entre eles e a câmera(cm)
SHOW_BITWISE_MAGBLU = 0                       #--- Mostra a junção das máscaras azul e magenta. Não funciona muito bem, mas é interessante.
# ------------------------------------------------
# ------------------------------------------------

# Setup webcam video capture
cap = cv2.VideoCapture("/home/borg/Repos/LearningImageProcessing/Atividade_03")
time.sleep(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def treatForLines(frame):
    # Shape detection using color (cv2.inRange masks are applied over orginal image)
    mask = cv2.inRange(frame,np.array([0,0,0]),np.array([0,0,255]))
    morphMask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((4, 4)))
    contornos, arvore = cv2.findContours(morphMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    frame_out = frame.copy()
    cv2.drawContours(frame_out, contornos, -1, [0, 0, 255], 3)
    return frame_out


running = True
while running:
    ret, frame = cap.read()
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
   
    if SHOW_LINES_DIST:

        # Canny edge detection
        blur = cv2.GaussianBlur(frame_gray,(7,7),0)
        bordas = auto_canny(blur)
        bordas_color = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR) # Changed to RGB to be able to draw with color on top.

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
    else:
        cv2.imshow('Detector de circulos',treatForLines(frame))

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
