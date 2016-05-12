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

# Crawl the given page for the given genre.
def crawlPage(target, page):
    document = BeautifulSoup(page, 'html.parser')
    for a in document.find_all('a', title='Download'):
        file = target + a['href'].split('/')[-1] + ".mp3"
        if not os.path.exists(file):
            urllib.urlretrieve(a['href'], file)
            print "\t\tDownloaded " + file
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
        r = requests.get(url)
        url = crawlPage(target, r.text)
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
