
# coding: utf-8

# # Ⅰ.Import Libraries

# In[1]:


# data analysis and wrangling
import os
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve


# # Ⅱ.Data Collection

# In[2]:


# import data
base = "dataset"
files = {"test":"test.csv","train":"train.csv"
              
        }
def file_path(string):
    return os.path.join(base,files[string])

train_df = pd.read_csv(file_path("train"))
test_df = pd.read_csv(file_path("test"))


# # Ⅲ.Basic Exploratory Analysis

# #### make  clear explanations for serveral important features(variables):
# #### Pclass: A proxy for socio-economic status (SES), 1st = Upper,2nd = Middle,3rd = Lower
# #### SibSp: The dataset defines family relations in this way: Sibling = brother, sister, stepbrother, stepsister. Spouse = husband, wife (mistresses and fiancés were ignored) 
# #### parch: The dataset defines family relations in this way: Parent = mother, father Child = daughter, son, stepdaughter, stepson. Some children travelled only with a nanny, therefore parch=0 for them.
# #### embarked: Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton
# #### And the dependent variable in train dataset is Survival: 1 means survive, 0 means dead.
# 

# In[3]:


# firstly, have a glance of data

train_df.head()


# In[4]:


# all variables' name
train_df.columns.values


# #### From above, we could recognize our variables' type: 
# ####  numerical and categorical
# ####  Categorical: Survived, Sex, and Embarked. Ordinal: Pclass.
# #### Numerical : Continous variables: Age, Fare. Discrete variables: SibSp, Parch.

# In[5]:


# Now we could view how many missing data and data types in our dataset
train_df.info()


# In[6]:


test_df.info()


# In[7]:


# Make sure our predict target
IDtest = test_df["PassengerId"]


# In[8]:


# And a better idea is to have a big picture of whole dataset
# Join train and test datasets in order to obtain the same number of features during categorical conversion
dataset =  pd.concat(objs=[train_df, test_df], axis=0).reset_index(drop=True)
dataset.info()


# #### From above, we find there are many missing data in Age and Cabin variables, which means we need to solve this in following analysis.

# In[9]:


### Summarize data
# Summarie and statistics
train_df.describe()


# # Ⅳ.Feature Analysis

# ### 1. Numerical features(variables)

# In[10]:


# Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived 
g = sns.heatmap(train_df[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")


# #### Only Fare feature seems to have a significative correlation with the survival probability.
# #### It doesn't mean that the other features are not usefull. Subpopulations in these features can be correlated with the survival. To determine this, we need to explore in detail these features

# ### SibSP

# In[11]:


# Explore SibSp feature vs Survived
g = sns.factorplot(x="SibSp",y="Survived",data=train_df,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# ### Parch

# In[12]:


# Explore Parch feature vs Survived
f  = sns.factorplot(x="Parch",y="Survived",data=train_df,kind="bar", size = 6 , 
palette = "muted")
f.despine(left=True)
f = f.set_ylabels("survival probability")


# ### Age

# In[13]:


# Explore Age vs Survived
h = sns.FacetGrid(train_df, col='Survived')
h = h.map(sns.distplot, "Age")


# #### Age distribution seems to be a tailed distribution, maybe a Normal distribution.
# 
# #### So, even if "Age" is not correlated with "Survived", we can see that there is age categories of passengers that of have more or less chance to survive.
# 
# #### It seems that very young passengers have more chance to survive.

# In[14]:


# Explore Age distibution 
j = sns.kdeplot(train_df["Age"][(train_df["Survived"] == 0) & (train_df["Age"].notnull())], color="Red", shade = True)
j = sns.kdeplot(train_df["Age"][(train_df["Survived"] == 1) & (train_df["Age"].notnull())], ax =j, color="Blue", shade= True)
j.set_xlabel("Age")
j.set_ylabel("Frequency")
j = j.legend(["Not Survived","Survived"])


# #### Now, we cleary see a peak correponsing (between 0 and 5) to babies and very young childrens.

# ### Fare

# #### From the basic explantory, we could find there is one misssing data in fare features. In machine learning, we always use imputation to solve missing data problem. And for this variable, imputing median is a plausible idea.

# In[15]:


#Fill Fare missing values with the median value
dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())


# In[16]:


# Explore Fare distribution 
q = sns.distplot(dataset["Fare"], color="m", label="Skewness : %.2f"%(dataset["Fare"].skew()))
q = q.legend(loc="best")


# #### As we  see, Fare distribution is very skewed. This can lead to overweigth very high values in the model, even if it is scaled.
# 
# #### In this case, it is better to transform it with the log function to reduce this skew.

# In[17]:


# Apply log to Fare to reduce skewness distribution
dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)


# In[18]:


# plot the new distribution
p = sns.distplot(dataset["Fare"], color="b", label="Skewness : %.2f"%(dataset["Fare"].skew()))
p = p.legend(loc="best")


# #### Wow, see, log transformation does make bell shape!

# ### 2. categorical features(variables)

# ### Sex

# In[19]:


# make a bar plot
g = sns.barplot(x="Sex",y="Survived",data=train_df)
g = g.set_ylabel("Survival Probability")


# #### From this plot, It is clearly obvious that Male have less chance to survive than Female.
# 
# #### So Sex, might play an important role in the prediction of the survival.

# ### Pclass

# In[20]:


# Explore Pclass vs Survived
p = sns.factorplot(x="Pclass",y="Survived",data=train_df,kind="bar", size = 6 , 
palette = "muted")
p.despine(left=True)
p = p.set_ylabels("survival probability")


# In[21]:


# Explore Pclass vs Survived by Sex
g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=train_df,
                   size=6, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# ####  First class passengers have more chance to survive than second class and third class passengers.
# 
# #### This trend is conserved when we look at both male and female passengers. And again, female has larger possibilities to survive.

# ### Embarked

# In[22]:


dataset["Embarked"].isnull().sum()


# In[23]:


dataset["Embarked"].value_counts()


# #### From above, we find 2 missing value in Embark features, and "S" is the most frequent value in Emabark, thus impute "S" for missing value.

# In[24]:


dataset["Embarked"] = dataset["Embarked"].fillna("S")


# In[25]:


# Explore Embarked vs Survived 
k = sns.factorplot(x="Embarked", y="Survived",  data=train_df,
                   size=6, kind="bar", palette="muted")
k.despine(left=True)
k = k.set_ylabels("survival probability")


# #### We find passenger coming from Cherbourg (C) have more chance to survive.
# 

# ## Ⅴ.Feature Engineering

# #### Feature engineering is an very important step in machine learning that creates more high-corralated variables for models.

# In[26]:


# Firstly, We need to fill missing values in age variables by imputing median

dataset["Age"] = dataset["Age"].fillna(dataset["Age"].median())


# ### Name/Title

# In[27]:


# take a look at 
dataset["Name"].head()


# #### The Name feature contains information on passenger's title.
# #### Since some passenger with distingused title may be preferred during the evacuation, it is interesting to add them to the model.

# In[28]:


# Get Title from Name
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset["Title"] = pd.Series(dataset_title)
dataset["Title"].head()


# In[29]:


g = sns.countplot(x="Title",data=dataset)
g = plt.setp(g.get_xticklabels(), rotation=45) 


# #### There is 17 titles in the dataset, most of them are very rare and we can group them in 4 categories.

# In[30]:


# Convert to categorical values Title 
dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
dataset["Title"] = dataset["Title"].astype(int)


# In[31]:


# Make a nice bar plot of four group count
g = sns.countplot(dataset["Title"])
g = g.set_xticklabels(["Master","Miss/Ms/Mme/Mlle/Mrs","Mr","Rare"])


# In[32]:


# And it is interesting to explore the relation between survival and title
g = sns.factorplot(x="Title",y="Survived",data=dataset,kind="bar")
g = g.set_xticklabels(["Master","Miss-Mrs","Mr","Rare"])
g = g.set_ylabels("survival probability")


# #### Wow, another amazing finding is people with rare names have more chances to survive than men with title Mr!

# In[33]:


# And now, we coulf drop variable "name" since we already made good use of it.
dataset.drop(labels = ["Name"], axis = 1, inplace = True)


# In[34]:


# convert to indicator values Title and Embarked 
dataset = pd.get_dummies(dataset, columns = ["Title"])
dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix="Em")
dataset.head()


# In[35]:


dataset.columns.values


# #### Now, we have 17 features in dataset.

# ### Cabin

# In[36]:


# At first have a look at "cabin"
dataset["Cabin"].head()


# In[37]:


dataset["Cabin"].isnull().sum()


# In[38]:


dataset.count()


# In[39]:


#### The Cabin feature column contains 292 values and 1007 missing values.
#### We supposed that passengers without a cabin have a missing value displayed instead of the cabin number.
dataset["Cabin"][dataset["Cabin"].notnull()].head()


# In[40]:


# Replace the Cabin number by the type of cabin 'X' if not
dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])


# In[41]:


# Similiar as last feature, we make a bar plot for encoded feature
g = sns.countplot(dataset["Cabin"],order=['A','B','C','D','E','F','G','T','X'])


# In[42]:


# And do not forget survial vs cabin
g = sns.factorplot(y="Survived",x="Cabin",data=dataset,kind="bar",order=['A','B','C','D','E','F','G','T','X'])
g = g.set_ylabels("Survival Probability")


# In[43]:


# encoding our "Cabin" feature to numerical values
dataset = pd.get_dummies(dataset, columns = ["Cabin"],prefix="Cabin")


# ### Ticket

# In[44]:


# Repeat what we did before
dataset["Ticket"].head()


# In[45]:


## Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X. 

Ticket = []
for i in list(dataset.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
    else:
        Ticket.append("X")
        
dataset["Ticket"] = Ticket
dataset["Ticket"].head()


# In[46]:


# encoding our "Ticket" feature to numerical values
dataset = pd.get_dummies(dataset, columns = ["Ticket"], prefix="T")


# In[47]:


# Create categorical values for Pclass
dataset["Pclass"] = dataset["Pclass"].astype("category")
dataset = pd.get_dummies(dataset, columns = ["Pclass"],prefix="Pc")
# convert Sex into categorical value 0 for male and 1 for female
dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})
dataset.head()


# # Ⅵ.Modeling

# In[48]:


# Drop useless variables 
dataset.drop(labels = ["PassengerId"], axis = 1, inplace = True)


# In[49]:


## Separate train dataset and test dataset
train_len = len(train_df)
train_df = dataset[:train_len]
test_df = dataset[-(len(dataset)-train_len):]
test_df = test_df.drop(["Survived"],axis = 1)


# In[50]:


## Separate train features and label(dependent variables) 

train_df["Survived"] = train_df["Survived"].astype(int)
Y_train = train_df["Survived"]
X_train = train_df.drop(labels = ["Survived"],axis = 1)


# ## 1.Cross Validation model

# #### We compared several classifiers that learnt from my machine learning class and evaluate the mean accuracy of each of them by a stratified kfold cross validation procedure.
# #### SVC
# #### Decision Tree
# #### Random Forest
# #### Gradient Boosting
# #### KNN
# #### Logistic regression
# #### Linear Discriminant Analysis

# In[51]:


# Cross validate model with 10-fold stratified cross val (note: 10-folds is the most common validation method for us to train model)
kfold = StratifiedKFold(n_splits=10)


# In[52]:


# Modeling step Test differents algorithms 
random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree",
"RandomForest","GradientBoosting","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")


# ## 2.Model Selection (Hyperparameter Tunning )

# In[53]:


### SVC classifier
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 2, verbose = 1)

gsSVMC.fit(X_train,Y_train)

SVMC_best = gsSVMC.best_estimator_

# Best score
gsSVMC.best_score_


# In[54]:


# Random forest Parameters tunning 
RFC = RandomForestClassifier()


## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 2, verbose = 1)

gsRFC.fit(X_train,Y_train)

RFC_best = gsRFC.best_estimator_

# Best score
gsRFC.best_score_


# In[55]:


# Gradient boosting tunning

GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 2, verbose = 1)

gsGBC.fit(X_train,Y_train)

GBC_best = gsGBC.best_estimator_

# Best score
gsGBC.best_score_


# ### Learning curve

# #### Learning curve is a good way to see the overfitting effect on the training set and the effect of the training size on the accuracy.

# In[56]:


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="g")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="b")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="g",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="b",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

g = plot_learning_curve(gsRFC.best_estimator_,"RF learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsSVMC.best_estimator_,"SVC learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsGBC.best_estimator_,"GradientBoosting learning curves",X_train,Y_train,cv=kfold)


# ### Ensemble Model

# #### Ensemble Model in machine learning means using specific weighted method to combine several models together in order to make a better prediction.

# In[57]:


# Here, we choose one weighted method called voting classifier to weight our three models.
votingC = VotingClassifier(estimators=[('rfc', RFC_best), 
('svc', SVMC_best),('gbc',GBC_best)], voting='soft', n_jobs=2)

votingC = votingC.fit(X_train, Y_train)


# # Production

# ### Prediction

# In[58]:


# This is our last step: make the prediction based on trained model
test_Survived = pd.Series(votingC.predict(test_df), name="Survived")
result = pd.concat([IDtest,test_Survived],axis=1)
writer = pd.ExcelWriter('Suvival prediction.xlsx')
result.to_excel(writer,'Sheet1')
writer.save()

