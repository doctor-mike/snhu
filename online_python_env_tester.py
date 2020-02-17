#
# 2020-02-09  
# by:  Dr. Michael Moran
#  professor.moran@gmail.com, m.moran3@snhu.edu
#

"""
This is a Python test program originally written
to test these online Python coding environments:
1. Repl.it:
  https://repl.it/languages/python3
2. Coding ground:
  https://www.tutorialspoint.com/execute_python3_online.php
3. Online GDB: 
  https://www.onlinegdb.com/online_python_compiler
4. Python Anywhere:
  https://www.pythonanywhere.com/login/
5. Infosec "data science" lab environment:
  https://lab.infoseclearning.com
The goal was to see if the selected online Python
coding environments work with the following Python libraries:
  vaderSentiment, Numpy, Pandas,
  matplotlib, Seaborn, NLTK, 
  Scikit-learn, and WordCloud.
"""

# 1. Test to see if Vader works:
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(score)))
    return score['compound']
#Try a couple of review statements for Vader to score the sentiment:
review1 = "The phone is super cool."
result1 = sentiment_analyzer_scores(review1)
print ("Score: {}".format(result1) )
# now do the same, but with added emoticon; note the score increase:
review2 = "The phone is super cool. :-)"
result2 = sentiment_analyzer_scores(review2)
print ("Score: {}".format(result2) )
review3 = "No comment"
result3 = sentiment_analyzer_scores(review3)
print ("Score: {}".format(result3) )
review4 = "No negative"
result4 = sentiment_analyzer_scores(review4)
print ("Score: {}".format(result4) )
review5 = "No positive"
result5 = sentiment_analyzer_scores(review5)
print ("Score: {}".format(result5) )
review6 = "N/A"
result6 = sentiment_analyzer_scores(review6)
print ("Score: {}".format(result6) )
review7 = "null"
result7 = sentiment_analyzer_scores(review7)
print ("Score: {}".format(result7) )


# 2. Test to see if Numpy works:
import numpy as np
print ('\nThe numpy version is {}'.format(np.__version__) )
a = np.arange(15).reshape(3, 5)
print(a)
print( a.shape )
print( a.ndim )
print( a.dtype.name )
print( a.itemsize )
a = np.empty( (3,5) )  
print(a) 


# 3. Now test Numpy + matplotlib:
import matplotlib
print ('\nThe matplotlib version is {}'.format(matplotlib.__version__) )
from matplotlib import pyplot as plt 
#print(dir(plt))
x = np.arange(1,11) 
y = 2 * x + 5 
plt.title("Matplotlib demo") 
plt.xlabel("x axis caption") 
plt.ylabel("y axis caption") 
plt.plot(x,y) 
plt.show() 
plt.savefig("1_np_chart.png") # Needed for repl.it
plt.clf() # clear out the figure/plot


# 4. Another test of Numpy + matplotlib but with Sine wave:
# Get x-values of the sine wave
time        = np.arange(0, 10, 0.1)
# Amplitude of sine wave will be sine of a variable like time
amplitude   = np.sin(time)
# Plot sine wave using time and amplitude
plt.plot(time, amplitude)
# Add a title for the sine wave plot
plt.title('Sine wave')
# Give x axis label for the sine wave plot
plt.xlabel('Time')
# Give y axis label for the sine wave plot
plt.ylabel('Amplitude = sin(time)')
plt.grid(True, which='both')
plt.axhline(y=0, color='k')
plt.show()
plt.savefig("2_sinewave_chart.png") # Needed for repl.it
plt.clf() # clear out the figure/plot


# 5. Test to see if Pandas works:
import pandas as pd
print ('\nThe pandas version is {}'.format(pd.__version__) )
Index= ['aaa', 'bbb', 'ccc', 'ddd', 'eee']
Cols = ['A', 'B', 'C', 'D']
df = pd.DataFrame(abs(np.random.randn(5, 4)), index=Index, columns=Cols)
print(df) # display Pandas dataframe contents


# 6. Test to see if Pandas + matplotlib works:
#from matplotlib import pyplot as plt 
plt.pcolor(df)
# Create a heatmap:
plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
plt.show()
plt.savefig("3_pd_matplotlib_heatmap.png") # Needed for repl.it
# don't clear out the plot this time as we use it next.


# 7. Test to see if Pandas + seaborn works:
import seaborn as sb
print ('\nThe seaborn version is {}'.format(sb.__version__) )
heat_map = sb.heatmap(df)
plt.show()
plt.savefig("4_pd_seaborn_heatmap.png")
# Compare the heatmaps created by seaborn and matplotlib.
# Which heatmap looks better to you?
plt.clf() # clear out the figure/plot


# 8. Same as above, but change heatmap appearance.
# See this page for details: 
#  https://likegeeks.com/seaborn-heatmap-tutorial/
plt.xlabel("Values on X axis")
plt.ylabel('Values on Y axis')
plt.title("Pandas + Seaborn Example Heatmap")
heat_map = sb.heatmap(df, annot=True, cmap="YlGnBu_r")
plt.show()
plt.savefig("5_pd_seaborn_heatmap2.png")
plt.clf() # clear out the figure/plot


#9. Import NLTK library and stopwords:
import nltk
print('\nThe nltk version is {}.'.format(nltk.__version__))
#nltk.download('stopwords')
#from nltk.corpus import stopwords


#10. Import Scikit learn library:
import sklearn
print('\nThe scikit-learn version is {}.'.format(sklearn.__version__))
from sklearn.feature_extraction.text import TfidfVectorizer 
vectorizer = TfidfVectorizer()
#print(vectorizer)


#11. Import Wordcloud library:
import wordcloud
print('\nThe wordcloud version is {}.'.format(wordcloud.__version__))
import pandas as pd
words=['Python is a general purpose, high-level language. Python is a favourite of up and coming coders. Python was designed to be fun to use (after all, it is named for Monty Python, how could it not be?) Python is highly recommended language for beginning coding to learn – and has become the top introductory coding language in American university programs. Python is used mostly for web apps and information security, though Python is also popular in the academic community for data science. Python is used by tech giants like Google, Dropbox, Pinterest and Spotify. Python popularity is growing each year.' ]
df=pd.DataFrame(words,columns=['text'])
df['text'] = df['text'].str.lower()
#print( df['text'].str.lower().str.split() )
from wordcloud import WordCloud, STOPWORDS
wordcloud = WordCloud(
    width = 300,
    height = 200,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(df['text']))
fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
plt.savefig("6_wordcloud.png")
plt.clf()


print("\nEnd of tests.")

