# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23rd 2020

@author: TomasGadea
"""

import pygame
import numpy as np
import time
import random
import torch


def predict(X):
	print('We are currently working on the neural network')


WIDTH, HEIGHT = 500, 500
nX, nY = 28, 28
xSize = WIDTH / nX
ySize = HEIGHT / nY

pygame.init()  # Initialize PyGame

screen = pygame.display.set_mode([WIDTH, HEIGHT])  # Set size of screen

black = (0, 0, 0)
gray = (100, 100, 100)
white = (255, 255, 225)

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
				pygame.draw.polygon(screen, white, poly, 0)
			elif status[x, y] == 0:
				pygame.draw.polygon(screen, black, poly, 0)
			else:
				pygame.draw.polygon(screen, gray, poly, 0)


	newStatus = np.copy(status)

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False

		if event.type == pygame.KEYDOWN:
			#pauseRun = not pauseRun
			if event.key == pygame.K_r:
				print('reset')
				newStatus = np.zeros((nX, nY))
			elif event.key == pygame.K_RETURN:
				predict(newStatus)

		mouseClick = pygame.mouse.get_pressed()
		if sum(mouseClick) > 0:
			posX, posY = pygame.mouse.get_pos()
			x, y = int(np.floor(posX/xSize)), int(np.floor(posY/ySize))
			newStatus[x, y] = not mouseClick[2]

			neighbours = [((x-1)%(nX), y),
						  ((x+1)%(nX), y),
						  (x, (y-1)%(nY)),
						  (x, (y+1)%(nY))]
			for coords in neighbours:
				x2, y2 = coords[0], coords[1]
				if newStatus[x2, y2] == 1:
					pass
				else:
					newStatus[x2, y2] = 2




	status = np.copy(newStatus)
	time.sleep(0.01)
	pygame.display.flip()


pygame.quit()
