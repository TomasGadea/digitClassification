# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23rd 2020

@author: TomasGadea
"""

import pygame
import numpy as np
import time
import random

print(10*"\n")

def outputStats(total, nBlue, nRed, iterations):

    # print("\riteration:", iterations, end="", flush=True)

    t = (4*total//10)
    b = (4*nBlue//10)
    r = (4*nRed//10)
    print("\rB: [", b*"*", (t-b)*".", "]", nBlue, '%', sep='', end='      ')
    print("R: [", r*"*", (t-r)*".", "]", nRed, '%', sep='', end='', flush=True)




WIDTH, HEIGHT = 500, 500
nX, nY = 28, 28
xSize = WIDTH / nX
ySize = HEIGHT / nY

pygame.init()  # Initialize PyGame

screen = pygame.display.set_mode([WIDTH, HEIGHT])  # Set size of screen

red = (255, 160, 122)
blue = (65, 105, 225)

status = np.zeros((nX, nY))

pauseRun = True
running = True

iterations = 0

while running:
    iterations += 1

    for x in range(0, nX):
        for y in range(0, nY):
            poly = [(x*xSize, y*ySize),
                    ((x+1)*xSize, y*ySize),
                    ((x+1)*xSize, (y+1)*ySize),
                    (x*xSize, (y+1)*ySize)]

            if status[x, y] == 1:
                pygame.draw.polygon(screen, blue, poly, 0)
            else:
                pygame.draw.polygon(screen, red, poly, 0)






    newStatus = np.copy(status)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            pauseRun = not pauseRun

        mouseClick = pygame.mouse.get_pressed()
        if sum(mouseClick) > 0:
            posX, posY = pygame.mouse.get_pos()
            x, y = int(np.floor(posX/xSize)), int(np.floor(posY/ySize))
            newStatus[x, y] = not mouseClick[2]


    if not pauseRun:
        randX = random.randint(0, nX-1)
        randY = random.randint(0, nX-1)

        neighbours = [((randX-1)%(nX), randY),
                        ((randX+1)%(nX), randY),
                        (randX, (randY-1)%(nY)),
                        (randX, (randY+1)%(nY))]

        pos = random.randint(0, 3)

        newStatus[randX, randY] = status[neighbours[pos]]


        nBlue = int(np.sum(status))
        nRed = status.size - nBlue

        outputStats(status.size, nBlue, nRed, iterations)

        if nBlue == status.size:
            running = False
            print()
            print(10*' ', "BLUE-DOMINATION in", iterations, "iterations", flush=True)
        elif nRed == status.size:
            running = False
            print()
            print(60*' ', "RED-DOMINATION in", iterations, "iterations", flush=True)



    status = np.copy(newStatus)
    time.sleep(0.01)
    pygame.display.flip()


pygame.quit()
