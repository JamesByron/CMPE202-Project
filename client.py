from __future__ import unicode_literals, print_function, division
import socket
import time
import random
from io import open
import glob
import unicodedata
import string
import torch.nn as nn
from torch.autograd import Variable
import math
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#from functools import reduce

all_letters = ",1234567890" + string.ascii_letters + " .;'_-"
n_letters = len(all_letters)
category_hit = {}
category_miss = {}
all_categories = [i for i in range(16)]
n_categories = len(all_categories)
n_hidden = 128
criterion = nn.NLLLoss()
learning_rate = 0.005
n_iters = 100000
print_every = 10000
plot_every = 1000
# Keep track of losses for plotting
current_loss_hit = 0
all_losses_hit = []
current_loss_miss = 0
all_losses_miss = []
startTime = time.time()


def readLines_Unused(inputList, index):
	theList = inputList[index]
	output = []
	for i in range(len(theList)):
		thisString = ""
		for j in range(len(theList[i])):
			thisString += theList[i][j] + ("," if j+1 < len(theList[i]) else "")
		output.append(thisString)
	return output

def letterToIndex(letter):
	return all_letters.find(letter)

def lineToTensor(line):
	tensor = torch.zeros(len(line), 1, n_letters)
	for li, letter in enumerate(line):
		tensor[li][0][letterToIndex(letter)] = 1
	return tensor

class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(RNN, self).__init__()
		self.hidden_size = hidden_size
		self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
		self.i2o = nn.Linear(input_size + hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, input, hidden):
		combined = torch.cat((input, hidden), 1)
		hidden = self.i2h(combined)
		output = self.i2o(combined)
		output = self.softmax(output)
		return output, hidden

	def initHidden(self):
		return Variable(torch.zeros(1, self.hidden_size))

rnn_hit = RNN(n_letters, n_hidden, n_categories)
rnn_miss = RNN(n_letters, n_hidden, n_categories)

# Training
def categoryFromOutput(output):
	top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
	category_i = top_i[0][0]
	return all_categories[category_i], category_i

def randomChoice(l):
	return l[random.randint(0, len(l) - 1)]

def randomTrainingExample(lines_dict):
	category = randomChoice(all_categories)
	line = randomChoice(lines_dict[category])
	category_tensor = Variable(torch.LongTensor([all_categories.index(category)])) # The Number of the category in the list
	line_tensor = Variable(lineToTensor(line))
	return category, line, category_tensor, line_tensor

def trainHit(category_tensor, line_tensor):
	global rnn_hit
	hidden = rnn_hit.initHidden()
	rnn_hit.zero_grad()
	for i in range(line_tensor.size()[0]):
		output, hidden = rnn_hit(line_tensor[i], hidden)
	loss = criterion(output, category_tensor)
	loss.backward()
	# Add parameters' gradients to their values, multiplied by learning rate
	for p in rnn_hit.parameters():
		p.data.add_(-learning_rate, p.grad.data)
	return output, loss.data[0]

def trainMiss(category_tensor, line_tensor):
	global rnn_miss
	hidden = rnn_miss.initHidden()
	rnn_miss.zero_grad()
	for i in range(line_tensor.size()[0]):
		output, hidden = rnn_miss(line_tensor[i], hidden)
	loss = criterion(output, category_tensor)
	loss.backward()
	# Add parameters' gradients to their values, multiplied by learning rate
	for p in rnn_miss.parameters():
		p.data.add_(-learning_rate, p.grad.data)
	return output, loss.data[0]

def timeSinceStart():
	now = time.time()
	s = now - startTime
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)

def trainLoop():
	global current_loss_hit
	global all_losses_hit
	for iter in range(1, n_iters + 1):
		category, line, category_tensor, line_tensor = randomTrainingExample(category_hit)
		output, loss = trainHit(category_tensor, line_tensor)
		current_loss_hit += loss
		# Print iter number, loss, name and guess
		if iter % print_every == 0:
			guess, guess_i = categoryFromOutput(output)
			correct = '✓' if guess == category else '✗ (%s)' % category
			print("Hit",'%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSinceStart(), loss, line, guess, correct))
		# Add current loss avg to list of losses
		if iter % plot_every == 0:
			all_losses_hit.append(current_loss_hit / plot_every)
			current_loss_hit = 0
	# Train for misses
	global current_loss_miss
	global all_losses_miss
	for iter in range(1, n_iters + 1):
		category, line, category_tensor, line_tensor = randomTrainingExample(category_miss)
		output, loss = trainMiss(category_tensor, line_tensor)
		current_loss_miss += loss
		# Print iter number, loss, name and guess
		if iter % print_every == 0:
			guess, guess_i = categoryFromOutput(output)
			correct = '✓' if guess == category else '✗ (%s)' % category
			print("Miss",'%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSinceStart(), loss, line, guess, correct))
		# Add current loss avg to list of losses
		if iter % plot_every == 0:
			all_losses_miss.append(current_loss_miss / plot_every)
			current_loss_miss = 0

def evaluateHit(line_tensor):
	global rnn_hit
	hidden = rnn_hit.initHidden()
	for i in range(line_tensor.size()[0]):
		output, hidden = rnn_hit(line_tensor[i], hidden)
	return output

def evaluateMiss(line_tensor):
	global rnn_miss
	hidden = rnn_miss.initHidden()
	for i in range(line_tensor.size()[0]):
		output, hidden = rnn_miss(line_tensor[i], hidden)
	return output

def predict(input_line, n_predictions=5):
	output = evaluateHit(Variable(lineToTensor(input_line)))
	outputMiss = evaluateMiss(Variable(lineToTensor(input_line)))
	# Get top N categories
	topv, topi = output.data.topk(n_predictions, 1, True)
	predictions = []
	topvM, topiM = outputMiss.data.topk(n_predictions, 1, True)
	predictionsM = []
	for i in range(n_predictions):
		value = topv[0][i]
		category_index = topi[0][i]
		#print('(%.2f) %s' % (value, all_categories[category_index]))
		#predictions.append([value, all_categories[category_index]])
		predictions.append(all_categories[category_index])
		value = topvM[0][i]
		category_index = topiM[0][i]
		#print('(%.2f) %s' % (value, all_categories[category_index]))
		#predictionsM.append([value, all_categories[category_index]])
		predictionsM.append(all_categories[category_index])
	best = -1
	for i in range(len(predictions)):
		if predictions[i] not in predictionsM:
			best = predictions[i]
			break
	best = best if best >= 0 else predictions[0]
	return best

def start():
	dataListsHit = [[] for i in range(16)]
	dataListsMiss = [[] for i in range(16)]
	while(True):
		print("Type 't' to train.\nType 'l' to learn.\nType 'e' to test.\nType 'w' to write.\nType 'p' to plot.\nType 'q' to quit.")
		inputString = input("Type a selection: ")
		if (inputString == "t") or (inputString == "e"):
			print("Acquiriog Connection")
			s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
			while (True):
				try:
					s.connect("/tmp/soc")
					break
				except:
					print("Sleeping for 5 seconds...")
					time.sleep(5)
			print("Connection Acquired")
			hit, miss, numHit, numMiss, numSent = getOneSequence(s, inputString)
			print("Total numHit:",numHit)
			print("Total numMiss:",numMiss)
			print("Total numSent:",numSent)
			s.close()
			for i in range(16):
				dataListsHit[i] += hit[i]
				dataListsMiss[i] += miss[i]
			print("Done")
		elif (inputString == "q"):
			break
		elif (inputString == "p"):
			makePlots()
		elif ((inputString == "l") and ((len(dataListsHit) > 0) or (len(dataListsMiss) > 0))):
			for i in range(16):
				category_hit[i] = dataListsHit[i]
				dataListsHit[i] = []
				category_miss[i] = dataListsMiss[i]
				dataListsMiss[i] = []
			trainLoop()
		if (inputString == "w"):
			outputFile = open("outputFile", "w")
			outputString = ""
			for i in range(len(dataListsHit)):
				outputString += str(dataListsHit[i]) + "\n"
				if (i % 2000 == 0) and (i > 0):
					outputFile.write(outputString)
					outputString = ""
			for i in range(len(dataListsMiss)):
				outputString += str(dataListsMiss[i]) + "\n"
				if (i % 2000 == 0) and (i > 0):
					outputFile.write(outputString)
					outputString = ""
			outputFile.write(outputString)
			outputFile.close()
	if (False):
		print(dataListsHit[0])
		print(dataListsHit[1])
		print(dataListsMiss[0])
		print(dataListsMiss[1])
	print("Exit")

def getOneValue(soc):
	data = soc.recv(4096)
	data = data.decode()
	data = data.split("*")
	output = []
	for i in range(len(data)):
		if len(data[i])>0:
			output.append(data[i].split(","))
	return output

def listToString(data):
	output = ""
	for i in range(len(data)):
		output += (data[i] + ("," if i+1 < len(data[i]) else ""))
	return output

def getOneSequence(soc, inputString):
	dataListHit = [[] for i in range(16)]
	dataListMiss = [[] for i in range(16)]
	numSent = 0
	numHit = 0
	numMiss = 0
	classifyList = [0 for i in range(16)]
	while (True):
		data = getOneValue(soc)
		for i in range(len(data)):
			if (data[i][0] == "Exit"):
				print(data)
				print("numHit:",numHit,", numMiss:",numMiss,", numSent:",numSent, classifyList)
				return dataListHit, dataListMiss, numHit, numMiss, numSent
			elif (data[i][0] == "Get"):
				#This is where we return a guess
				data[i].pop(4)
				data[i].pop(2)
				data[i].pop(1)
				data[i].pop(0)
				guess = random.randint(0,15) if inputString == "t" else predict(listToString(data[i]))
				classifyList[guess] += 1
				soc.send(str(guess).encode())
				numSent += 1
			elif (len(data[i]) == 22):
				way = int(data[i].pop(4))
				data[i].pop(2)
				hm = data[i].pop(1)
				data[i].pop(0)
				if hm == "Hit":
					numHit += 1
					if inputString == "t":
						dataListHit[way].append(listToString(data[i]))
				else:
					numMiss += 1
					if inputString == "t":
						dataListMiss[way].append(listToString(data[i]))
		if (numHit+numMiss) % 10000 == 0:
			print("numHit:",numHit,", numMiss:",numMiss,", numSent:",numSent, classifyList)
	return dataListHit, dataListMiss, numHit, numMiss, numSent

def makePlots():
	confusion = torch.zeros(n_categories, n_categories)
	n_confusion = 10000
	# Go through a bunch of examples and record which are correctly guessed
	for i in range(n_confusion):
		category, line, category_tensor, line_tensor = randomTrainingExample(category_hit)
		output = evaluateHit(line_tensor)
		guess, guess_i = categoryFromOutput(output)
		category_i = all_categories.index(category)
		confusion[category_i][guess_i] += 1
	for i in range(n_categories):
		confusion[i] = confusion[i] / confusion[i].sum()
	plt.figure()
	plt.plot(all_losses_hit, label='Hit Prediction Error')
	plt.plot(all_losses_miss, label='Miss Prediction Error')
	plt.legend()
	#plt.show()
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(confusion.numpy())
	fig.colorbar(cax)

	# Set up axes
	ax.set_xticklabels([''] + all_categories, rotation=90)
	ax.set_yticklabels([''] + all_categories)

	# Force label at every tick
	ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
	ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

	# sphinx_gallery_thumbnail_number = 2
	plt.show()

start()