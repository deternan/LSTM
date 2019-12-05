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

r = requests.get(url)
html_str = r.text

# response state
#print(r.status_code)

#print(type(r))
#print(type(html_str))

# parsing data
#soup = BeautifulSoup(html_str)
soup = BeautifulSoup(html_str, 'html.parser')
#print(soup.prettify())

# = 'fsZx3'
link_tag = soup.find(id='fsZx3')
print(link_tag.text)
