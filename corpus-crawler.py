#!/usr/bin/python

import os
import urllib
import requests
from bs4 import BeautifulSoup

# Site URL.
URL = "http://freemusicarchive.org/genre/"

# Enumeration of all existing genres.
GENRES = [
    "Blues", "Classical", "Country", "Electronic",
    "Experimental", "Folk", "Hip-Hop", "Instrumental",
    "International", "Jazz", "Old-Time__Historic",
	"Pop", "Rock", "Soul-RB", "Spoken"
]

HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

# Crawl the given page for the given genre.
def crawlPage(target, page):
    document = BeautifulSoup(page, 'html.parser')
    for a in document.find_all('a', title='Download'):
        file = target + a['href'].split('/')[-1] + ".mp3"
        if not os.path.exists(file):
            with open(file, 'w') as handle:
                response = requests.get(a['href'], headers=HEADERS, stream=True)
                if response.ok:
                    for block in response.iter_content(1024):
                        handle.write(block)
                    print "\t\tDownloaded " + file
                else:
                    print("Error while downloading : " + a['href'])
    next = document.findAll('a', href=True, text=u'NEXT\xa0\xbb')
    if len(next) > 0:
        return next[0]['href']
    return None

# Crawl the given genre.
def crawlGenre(output, genre):
    target = output + genre + '/'
    if not os.path.exists(target):
        os.makedirs(target)
    url = URL + genre
    while url != None:
        print "\tCrawling page " + url
        response = requests.get(url, headers=HEADERS)
        url = crawlPage(target, response.text)
    return

# Crawl the website.
def crawl(output):
    for genre in GENRES:
        print "Crawling genre " + genre
        crawlGenre(output, genre)
    return

# Main method.
if __name__ == "__main__":
    corpus = "corpus/"
    if not os.path.exists(corpus):
        os.makedirs(corpus)
    crawl(corpus)
