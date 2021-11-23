# -*- coding: utf-8 -*-
"""
@author: TomasGadea
"""

from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame 
import numpy as np
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange


class BobNet(nn.Module):
	def __init__(self):
		super(BobNet, self).__init__()
		self.l1 = nn.Linear(784, 128)
		self.act = nn.ReLU()
		self.l2 = nn.Linear(128, 10)
		self.sm = nn.LogSoftmax(dim=1)
	def forward(self, x):
		x = self.l1(x)
		x = self.act(x)
		x = self.l2(x)
		x = self.sm(x)
		return x




# load the mnist dataset
def fetch(url):
	import requests, gzip, os, hashlib, numpy
	fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
	if os.path.isfile(fp):
		with open(fp, "rb") as f:
			dat = f.read()
	else:
		with open(fp, "wb") as f:
			dat = requests.get(url).content
			f.write(dat)
	return numpy.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()


def train(model):
	print()

	X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
	Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
	X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
	Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]


	loss_function = nn.NLLLoss()
	opt = optim.SGD(model.parameters(), lr=0.001)
	BS = 32

	losses, accuracies = [], []

	for i in trange(1000):
		samp = np.random.randint(0, X_train.shape[0], size=BS)
		X = torch.tensor(X_train[samp].reshape((-1, 28*28))).float()
		Y = torch.tensor(Y_train[samp]).long()

		opt.zero_grad()

		out = model.forward(X)
		cat = torch.argmax(out, dim=1)
		accuracy = (cat == Y).float().mean()
		loss = loss_function(out, Y)
		loss = loss.mean()
		loss.backward()
		opt.step()
		loss, accuracy = loss.item(), accuracy.item()
		losses.append(loss)
		accuracies.append(accuracy)

	print("Finished training")



def predict(X, model):
	pred = torch.argmax(model.forward(torch.tensor(X).reshape((-1, 28*28)).float()), dim=1).numpy()[0]
	print("Your number is a", pred)

def diaplay_instructions():
	print()
	print('WELCOME TO THE DIGIT PREDICTOR!')
	print()
	print('Please, draw any digit number from 0 to 9 in the screen that popped up')
	print('To draw, hold your left button mouse and move it around the screen')
	print("If you want to erase your drawing press 'r' and try again")
	print("Once you have finished press 'return' and the system will predict the digit (see terminal)")
	print()
	print()
	print()




def main():
	Bob = BobNet()
	train(Bob)

	diaplay_instructions()

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
				
				if status[x, y] == 255:
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
					#print(newStatus.T)
					predict(newStatus.T, Bob)

			mouseClick = pygame.mouse.get_pressed()
			if sum(mouseClick) > 0:
				posX, posY = pygame.mouse.get_pos()
				x, y = int(np.floor(posX/xSize)), int(np.floor(posY/ySize))
				newStatus[x, y] = 255#not mouseClick[2]

				neighbours = [((x-1)%(nX), y),
							  ((x+1)%(nX), y),
							  (x, (y-1)%(nY)),
							  (x, (y+1)%(nY))]
				for coords in neighbours:
					x2, y2 = coords[0], coords[1]
					if newStatus[x2, y2] == 255:
						pass
					else:
						newStatus[x2, y2] = 200




		status = np.copy(newStatus)
		time.sleep(0.01)
		pygame.display.flip()


	pygame.quit()



if __name__ == '__main__':
	main()