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
words=['C is a general-purpose coding language, originally created for Unix systems. It’s commonly used in cross-platform systems, Unix coding and game coding. It’s often selected because it’s more compact than C++ and runs faster.  It’s the second most common programming language following Java. C is the grandfather of many other coding languages including C#, Java, JavaScript, Perl, PHP and Python. C++ is an intermediate-level coding language that is object-oriented. It derives from C, however, it has add-ons and enhancements to make it a more multifaceted coding language. It’s well suited to large projects, as it can be broken up into parts enabling easy collaboration. It’s used by some of the world’s best-known tech companies including Adobe, Google, Mozilla, and Microsoft. Objective-C, like most of the coding languages on this list, it derives from C. It’s a general purpose, high-level code that has an added message-passing function. It’s known for being the coding language of choice for Apple’s OS X and iOS apps until it was replaced by Swift. Java is currently the most popular and widely used language in the world. Though it was originally created for interactive TV, it’s become known as the language of choice for Android devices. It’s also the coding language of choice enterprise-level software. It’s a good multi-purpose coding language because it can be used cross-platform (meaning it’s just as easily used on smartphone apps as on desktop apps). It resembles C++ in syntax and structure making it easy to pick up if you know C languages already. JavaScript was created as an add-on code to extend the functionality of web pages. It adds dynamic features such as submission forms, interactivity, animations, user-tracking, etc. It’s mostly used for front-end development, or for coding solutions that customers and clients interact with. It’s compatible with all browsers, making it a good general-purpose web development code, though it’s also known to be difficult to debug. Swift is hailed as the replacement for Objective-C when it comes to Apple programs, it’s been gaining popularity in recent years, because it’s easy-to-read, easy to maintain and faster than Objective-C. If you’re looking to be an Apple coder or write programs for iOS, this is the language you want to learn. Though Objective-C is still used, Swift is quickly becoming the programming language of choice for coders creating programs for Apple devices. A general purpose, object-oriented code, C# (pronounced C Sharp) was created by Microsoft in 2001. Though it’s named after the C family of coding languages, it has more in common with Java than other C languages. C# is mostly used for internal/enterprise solutions and is less frequent in commercial software. PHP is an open source code, primarily used for web development (a.k.a. creating web pages.) PHP was created to streamline web page creation. It’s a fairly simple programming language that can be picked up quickly. It’s used by many web-based companies including Facebook, Wikipedia, and WordPress. Another general purpose, high-level code, Python is a favourite of up and coming coders. It was designed to be fun to use (after all, it’s named for Monty Python, how could it not be?) It’s another highly recommended language for coding beginners to learn – and has become the top introductory coding language in American university programs. It’s used mostly for web apps and information security, though it’s also popular in the academic community for data analysis. It’s used by tech giants like Google, Dropbox, Pinterest and Spotify. Ruby is an object-oriented, general purpose programming language developed in the mid-90s in Japan. It’s one of the simpler programming languages to learn and is often used as a stepping stone to Ruby on Rails. It was developed to be both fun to code, and to increase productivity. It’s known for being easy to read, and as a result many programmers recommend learning Ruby as your first coding language. It’s used by sites like Hulu, Shopify, Airbnb and many others.' ]
df=pd.DataFrame(words,columns=['text'])
df['text'] = df['text'].str.lower().str.split()
print(df['text'])
from wordcloud import WordCloud, STOPWORDS
wordcloud = WordCloud(
    width = 500,
    height = 400,
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
