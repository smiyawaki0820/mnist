##########################################################################
### Prepare the trainloader/testloader for training/validating network ###
##########################################################################

import os
import sys
import logging
import argparse
from typing import List

import numpy as np
import pandas as pd

import tensorboard

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from   torchvision import datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from src.dataloaders.dataloader import *
from src.models.lenet import LeNet
from src.models.simple_network import SimpleNetwork


logging.basicConfig(
    format='%(asctime)s #%(lineno)s %(levelname)s %(name)s :::  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)

class ModelTrainer(object):
  def __init__(self,
               net: nn.Module = None,
               criterion = None,
               optimizer = None,
               mnist_train = None,
               mnist_test = None,
              ):
    if any(map(lambda x: x is None, (net, criterion, optimizer))):
      logger.warning('Apply Default Settings for {net, criterion, optimizer}')
      self.set_default_modules()
    else:
      self.net = net
      self.criterion = criterion
      self.optimizer = optimizer

    if torch.cuda.is_available():
      self.net = self.net.cuda()
    logger.info('\033[31m' + str(self.net) + '\033[0m')

    # The 「transform」 is used to 
    # i)  convert PIL.Image to torch.FloatTensor (batch, dim, H, W), and change the 
    #     inputs' range to [0, 1] (by inputs/= 255.0);
    # ii) standardize the input images by mean=0.1307, std=0.3081
    self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    if any(map(lambda x: x is None, (mnist_train, mnist_test))):
      logger.info('Download MNIST data')
      self.download_data()
    else:
      self.mnist_train = mnist_train
      self.mnist_test  = mnist_test

    self.trainloader = DataLoader(self.mnist_train, batch_size=50, shuffle=True,  num_workers=2)
    self.testloader  = DataLoader(self.mnist_test, batch_size=100, shuffle=False, num_workers=2)
    
  def set_default_modules(self,):
    self.net = SimpleNetwork()  # LeNet()
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)  # optim.Adam(net.parameters())

  def download_data(self, dest='./data'):
    isDownload = False if os.path.isdir(dest) else True
    self.mnist_train = MyMNIST('./data', train=True,  download=isDownload, transform=self.transform, limit_data=1000)
    self.mnist_test  = MyMNIST('./data', train=False, download=isDownload, transform=self.transform)

  def train(self, args:argparse.Namespace, trainloader, testloader) -> List[dict]:
    results = []
    for epoch in range(1, args.epoch_size+1):
      loss_train = .0
      ct_num = 0

      for iteration, data in enumerate(trainloader, start=1):
        # Take the inputs and the labels for 1 batch.
        inputs, labels = data
        bch = inputs.size(0)
        inputs = inputs.view(bch, -1)
        
        if torch.cuda.is_available():
          inputs = inputs.cuda()
          labels = labels.cuda()

        self.optimizer.zero_grad()
        outputs = self.net(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        
        loss_train += loss.item()
        ct_num+= 1
        if iteration%20 == 19:
          logger.info(
            "[Epoch: {ep}] --- Iteration: {iter}, Loss: {loss}.".format(ep=epoch, iter=iteration, loss=loss_train/ct_num)
          )

        # Test
        # if epoch%3 == 2 and iteration%100 == 99:
        #   _ = self.evaluate()
        
      acc = self.evaluate(testloader)
      results.append({
        'epoch': epoch,
        'loss': loss_train/ct_num,
        'acc': acc,
      })
    
    return results

  def evaluate(self, testloader) -> float:
    """ ------ Script for evaluate the network ------ """
    logger.info("Testing the network...")
    self.net.eval()
    total_num, correct_num = 0, 0
    for test_iter, test_data in enumerate(testloader):
      inputs, labels = test_data    
      bch = inputs.size(0)
      inputs = inputs.view(bch, -1).cuda()
      labels = torch.LongTensor(list(labels)).cuda()

      outputs = self.net(inputs)
      _, pred_cls = torch.max(outputs, 1)
      if total_num == 0:
         logger.info(f"True label: {labels}")
         logger.info(f"Prediction: {pred_cls}")

      correct_num += (pred_cls == labels).float().sum().item()
      total_num += bch
    
    acc = correct_num/float(total_num)
    logger.info("Accuracy: {:.2f}".format(acc))
    return acc

  @staticmethod
  def write_score(results:List[dict], fo:str=None):
    pd.DataFrame(results)\
      .astype({'epoch':int, 'loss':float, 'acc':float})\
      .set_index('epoch')\
      .to_csv(open(fo, 'w') if fo is not None else sys.stdout, index=True, sep='\t')

def create_arg_parser():
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--epoch_size', default=10, type=int)
  parser.set_defaults(no_thres=False)
  return parser

def run():
  parser = create_arg_parser()
  args = parser.parse_args()

  trainer = ModelTrainer(
    net = None,
    criterion = None,
    optimizer = None,
  )

  results = trainer.train(args, trainer.trainloader, trainer.testloader)
  trainer.write_score(results)


if __name__ == '__main__':
  run()