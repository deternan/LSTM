# coding=utf8

'''
version: December 05, 2019 05:31 PM
Last revision: December 05, 2019 06:54 PM

Author : Chao-Hsuan Ke

RNN LSTM
Reference:
https://www.itread01.com/content/1546400720.html
https://www.cnblogs.com/jclian91/p/9863848.html
https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

BeautifulSoup
https://blog.gtwang.org/programming/python-beautiful-soup-module-scrape-web-pages-tutorial/2/

'''

import requests
from bs4 import BeautifulSoup

url = 'https://mojim.com/twy100012x1x1.htm'
name = '瘋狂世界'
extension = '.txt'

#output
outputFolder = "D:\\Phelps\\GitHub\\Python\\LSTM\\data\\MayDay\\"               # output folder name
outputFile = name                                                       # output file name
# open file
fp = open(outputFolder + name + extension, "a")


r = requests.get(url)
html_str = r.text

# response state
#print(r.status_code)

# parsing data
soup = BeautifulSoup(html_str, 'html.parser')
#print(soup.prettify())

# = 'fsZx3'
link_tag = soup.find(id='fsZx3')
#print(link_tag)
#print(link_tag.text)

#ouptut
fp.writelines(link_tag.text)
fp.close()

print('finished')

