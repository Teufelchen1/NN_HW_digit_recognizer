#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 20:28:13 2019

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt

def loadimg(name):
    return (plt.imread(name)).flatten()

def showimg(array):
    img = np.reshape(array,(28,28))
    plt.imshow(img)

def randomimg():
    return np.random.rand(28*28)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

class Network:    
    # Input layer
    IL  = np.zeros(28*28)
    # First hidden layer
    HL1 = np.zeros(32)
    # Second hidden layer
    HL2 = np.zeros(32)
    # Output layer
    OL  = np.zeros(11)
    
    # Weight for first hidden layer
    W1  = (np.random.rand(32,28*28))-0.5
    # Bias for first hidden layer
    B1  = (np.random.rand(32))-0.5
    # Weight for second hidden layer
    W2  = (np.random.rand(32,32))-0.5
    # Bias for second hidden layer
    B2  = (np.random.rand(32))-0.5
    # Weight for output layer
    W3  = (np.random.rand(11,32))-0.5
    # Bias for output layer
    B3  = (np.random.rand(11))-0.5

    # Calculates output layer from input layer
    def calc_network(self):
        self.__calc_hl1()
        self.__calc_hl2()
        self.__calc_ol()

    # calculates first hidden layer from input layer
    def __calc_hl1(self):
        self.HL1 = sigmoid (np.dot(self.W1,self.IL) + self.B1)
    
    # calculates second hidden layer  from first hidden layer    
    def __calc_hl2(self):
        self.HL2 = sigmoid (np.dot(self.W2,self.HL1) + self.B2)
    
    # calculates output layer from second hidden layer
    def __calc_ol(self):
        self.OL = sigmoid (np.dot(self.W3,self.HL2) + self.B3)
    
    expected_value = 0
    
    def set_expected_value(self,number):
        self.expected_value = number
    
    def expec_vector(self):
        vec = np.zeros(11)
        vec[self.expected_value] = 1
        return vec
    
    grad_W3 = (np.zeros((11,32)))
    grad_B3 = (np.zeros(11))
    
    def __backprop_ol(self):
        for j in range(0,11):
            z = np.dot(self.W3[j],self.HL2) + self.B3[j]
            self.grad_B3[j] = 2*dsigmoid(z)*(self.OL[j]-self.expec_vector()[j])
            
            self.grad_W3[j,:] = self.HL2 * self.grad_B3[j]
    
    grad_W2 = np.zeros((32,32))
    grad_B2 = np.zeros(32)
    
    def __backprop_hl2(self):
        for k in range(0,32):
            prev = np.dot(self.W3[:,k],self.grad_B3)
            z = np.dot(self.W2[k],self.HL1) + self.B2[k]
            self.grad_B2[k] = dsigmoid(z)*prev
            
            self.grad_W2[k,:] = self.HL1 * self.grad_B2[k]
                
    grad_W1 = np.zeros((32,28*28))
    grad_B1 = np.zeros(32)
     
    def __backprop_hl1(self):
        for i in range(0,32):
            prev = np.dot(self.W2[:,i],self.grad_B2)
            z = np.dot(self.W1[i],self.IL) + self.B1[i]
            self.grad_B1[i] = dsigmoid(z)*prev
            
            self.grad_W1[i,:] = self.IL * self.grad_B1[i]
        
    def backprop(self):
        self.__backprop_ol()
        self.__backprop_hl2()
        self.__backprop_hl1()
    
    def feed(self,data):
        self.IL = data
        self.calc_network()
        return np.argmax(self.OL)

    def feedraw(self,file):
        self.feed((plt.imread(file)).flatten())

class Trainer:
    data = []
    package_size = 10
    
    def load_data(self):
        for num in [1,2,3,4,5,6,7,8,9,0]:
            for i in range(0,5000):
                img = (plt.imread("Training data/"+str(num)+"/"+str(i)+".png")).flatten()
                self.data.append((num,img))
        
        for letter in ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]:
            for i in range(0,200):
                img = (plt.imread("Training data/letters/"+str(letter)+"/"+str(letter)+"-"+str(i)+".png")).flatten()
                self.data.append((10,img))
        
        for i in range(0,5000):
            self.data.append((10,randomimg()))
        
        self.data = np.array(self.data)
        np.random.shuffle(self.data)
    
    def randomize(self):
        np.random.shuffle(self.data)
        
    @staticmethod
    def __training_step(network,package):
        avr_grad_W1 = np.zeros((32,28*28))
        avr_grad_W2 = np.zeros((32,32))
        avr_grad_W3 = np.zeros((11,32))
        
        avr_grad_B1 = np.zeros(32)
        avr_grad_B2 = np.zeros(32)
        avr_grad_B3 = np.zeros(11)
        
        for (num,data) in package:
            network.set_expected_value(num)
            network.feed(data)
            network.backprop()
            
            avr_grad_W1 += network.grad_W1
            avr_grad_W2 += network.grad_W2
            avr_grad_W3 += network.grad_W3
            
            
            avr_grad_B1 += network.grad_B1
            avr_grad_B2 += network.grad_B2
            avr_grad_B3 += network.grad_B3
    
        length = len(package)
        
        network.W1 -= avr_grad_W1 / length
        network.W2 -= avr_grad_W2 / length
        network.W3 -= avr_grad_W3 / length
        
        network.B1 -= avr_grad_B1 / length
        network.B2 -= avr_grad_B2 / length
        network.B3 -= avr_grad_B3 / length
        
    def training(self,network):
        for i in range(0,int(len(self.data)/self.package_size)):
            package = self.data[self.package_size*i:self.package_size*(i+1)]
            self.__training_step(network,package)
            print ("\r"+str(i)+"/"+str(int(len(self.data)/self.package_size)))

class Tester:
    data = []
    
    def load_data(self):
        for num in [1,2,3,4,5,6,7,8,9,0]:
            for i in range(0,300):
                img = (plt.imread("Test data/"+str(num)+"/"+str(5001+i)+".png")).flatten()
                self.data.append((num,img))
        
        for letter in ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]:
            for i in range(0,200):
                img = (plt.imread("Test data/letters/"+str(letter)+"/"+str(letter)+"-"+str(201+i)+".png")).flatten()
                self.data.append((10,img))
        
        for i in range(0,200):
            self.data.append((10,randomimg()))
        
        self.data = np.array(self.data)

    correct = 0
    wrong_cases = []
    
    def test(self):
        self.correct = 0
        self.wrong_cases = []
        for (num,img) in self.data:
            erg = n.feed(img)
            if erg == num:
                self.correct += 1
            else:
                self.wrong_cases.append((erg,num,img))
        
        return (self.correct/len(self.data))

    def show_wrong_case(self,n):
        return (self.wrong_cases[n][0],self.wrong_cases[n][1],showimg(self.wrong_cases[n][2]))

n = Network()   

tr = Trainer()
tr.load_data()
for i in range(0,10):
    tr.training(n)
    tr.randomize()

te = Tester()
te.load_data()
print(te.test())