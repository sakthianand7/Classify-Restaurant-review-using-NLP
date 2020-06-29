import re
import nltk
import pandas as pd
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

nltk.download('stopwords')
ps=PorterStemmer()
corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-z]',' ',dataset['Review'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english')) ]
    review=' '.join(review)
    corpus.append(str(review))
cv=CountVectorizer()
X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,-1].values
classifier=GaussianNB()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
classifier.fit(X_train,y_train)
ypred=classifier.predict(X_test)
cm=confusion_matrix(y_test,ypred)
print(cm)
print(classifier.score(X_test,y_test))