
import bayes_algorithm
import feedparser

ny = feedparser.parse('https://newyork.craigslist.org/search/stp?format=rss')
sf = feedparser.parse('https://sfbay.craigslist.org/search/stp?format=rss')

vocabList, pSF, pNY = bayes_algorithm.localWords(ny, sf)

vocabList, pSF, pNY = bayes_algorithm.localWords(ny, sf)

bayes_algorithm.getTopWords(ny, sf)

