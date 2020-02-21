#!/usr/bin/python

# -*- coding: utf-8 -*-

# __author__      = "Matheus Dib, Fabio de Miranda" Modificado
__author__ = "Carlos Dip"


import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import auxiliar as aux

# If you want to open a video, just change this path
#cap = cv2.VideoCapture('hall_box_battery.mp4')

# Parameters to use when opening the webcam.
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
img_logo = cv2.imread("insper.jpg")
img_logo = cv2.cvtColor(img_logo, cv2.COLOR_BGR2RGB)

lower = 0
upper = 1
focus = 280.0071427660373 * 30 / 14 # Calibração do foco (F = h*d/H)
magenta = "#F08020"
blue = "#586B7D"
m1,m2 = np.array([172, 50, 50]), np.array([180, 255, 255])
b1,b2 = np.array([80, 50, 50]), np.array([110, 255, 255])

# BRISK
brisk = cv2.BRISK_create()
font = cv2.FONT_HERSHEY_SIMPLEX
fontsize = 2

# BRISK no logo INSPER
kplogo, deslogo = brisk.detectAndCompute(img_logo, None) 
MIN_MATCH_COUNT = 10
            

# Returns an image containing the borders of the image
# sigma is how far from the median we are setting the thresholds
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def find_homography_draw_box(kp1, kp2, img_cena):
    
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
    img2b = cv2.polylines(out,[np.int32(dst)],True,(255,255,0),5, cv2.LINE_AA)
    
    return img2b



while(True):
    # Captura da imagem
    ret, frame = cap.read()
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Tratamento da imagem (Detecção de bordas)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    bordas = auto_canny(blur)
    bordas_color = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)

    # BRISK

    kpframe, desframe = brisk.detectAndCompute(bordas_color,None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = bf.knnMatch(deslogo,desframe,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)


    if len(good)>MIN_MATCH_COUNT:
        # Separa os bons matches na origem e no destino
        print("Matches found")    
        framed = find_homography_draw_box(kplogo, kpframe, bordas_color)
    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    # Seleção das imagens por bordas

    # frame_mag = cv2.inRange(frame_hsv,m1,m2)
    # frame_blu = cv2.inRange(frame_hsv,b1,b2)
    # frame_mag = cv2.morphologyEx(frame_mag,cv2.MORPH_CLOSE,np.ones((4, 4)))
    # frame_blu = cv2.morphologyEx(frame_blu,cv2.MORPH_CLOSE,np.ones((4, 4)))
    # contornos_mag, arvore = cv2.findContours(frame_mag.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    # contornos_blu, arvore = cv2.findContours(frame_blu.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # frame_mag_out = frame.copy()
    # frame_blu_out = frame.copy()
    # cv2.drawContours(frame_mag_out, contornos_mag, -1, [0, 0, 255], 3)
    # cv2.drawContours(frame_blu_out, contornos_blu, -1, [0, 0, 255], 3) 


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
            cv2.putText(bordas_color, str(real_dist), (5,fontsize*30), font, fontsize, (255,255,255))
            # print(real_dist)
            angle = np.arctan(dy/dx)*180/np.pi
            cv2.putText(bordas_color, str(angle), (5,fontsize*60), font, fontsize, (255,255,255))
            # print(angle)

    # Display the resulting frame
    cv2.imshow('Detector de circulos',bordas_color)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
