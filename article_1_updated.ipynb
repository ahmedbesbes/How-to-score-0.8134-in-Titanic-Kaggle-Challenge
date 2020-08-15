{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The <a href=\"https://www.kaggle.com/c/titanic/\"> Titanic challenge</a>  hosted by Kaggle is a competition in which the goal is to predict the survival or the death of a given passenger based on a set of variables describing him such as his age, his sex, or his passenger class on the boat.\n",
    "\n",
    "I have been playing with the Titanic dataset for a while, and I have recently achieved an accuracy score of 0.8134 on the public leaderboard. As I'm writing this post, I am ranked among the top 4% of all Kagglers. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This post is the opportunity to share my solution with you.\n",
    "\n",
    "To make this tutorial more \"academic\" so that anyone could benefit, I will first start with an exploratory data analysis (EDA) then I'll follow with feature engineering and finally present the predictive model I set up.\n",
    "\n",
    "Throughout this jupyter notebook, I will be using Python at each level of the pipeline.\n",
    "\n",
    "The main libraries involved in this tutorial are: \n",
    "\n",
    "* <b>Pandas</b> for data manipulation and ingestion\n",
    "* <b>Matplotlib</b> and <b> seaborn</b> for data visualization\n",
    "* <b>Numpy</b> for multidimensional array computing\n",
    "* <b>sklearn</b> for machine learning and predictive modeling\n",
    "\n",
    "### Installation procedure \n",
    "\n",
    "A very easy way to install these packages is to download and install the <a href=\"http://conda.pydata.org/docs/install/quick.html#os-x-miniconda-install\">Conda</a> distribution that encapsulates them all. This distribution is available on all platforms (Windows, Linux and Mac OSX).\n",
    "\n",
    "### Nota Bene\n",
    "\n",
    "This is my first attempt as a blogger and as a machine learning practitioner. \n",
    "\n",
    "If you have a question about the code or the hypotheses I made, do not hesitate to post a comment in the comment section below.\n",
    "\n",
    "If you also have a suggestion on how this notebook could be improved, please reach out to me.\n",
    "\n",
    "This tutorial is available on my <a href=\"https://github.com/ahmedbesbes/How-to-score-0.8134-in-Titanic-Kaggle-Challenge\"> github </a> account.\n",
    "\n",
    "Hope you've got everything set on your computer. Let's get started."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# I -  Exploratory data analysis\n",
    "\n",
    "As in different data projects, we'll first start diving into the data and build up our first intuitions.\n",
    "\n",
    "In this section, we'll be doing four things. \n",
    "\n",
    "- Data extraction : we'll load the dataset and have a first look at it. \n",
    "- Cleaning : we'll fill in missing values.\n",
    "- Plotting : we'll create some interesting charts that'll (hopefully) spot correlations and hidden insights out of the data.\n",
    "- Assumptions : we'll formulate hypotheses from the charts."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We tweak the style of this notebook a little bit to have centered plots."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from IPython.core.display import HTML\n",
    "HTML(\"\"\"\n",
    "<style>\n",
    ".output_png {\n",
    "    display: table-cell;\n",
    "    text-align: center;\n",
    "    vertical-align: middle;\n",
    "}\n",
    "</style>\n",
    "\"\"\");"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We import the useful libraries."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "%matplotlib inline\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "warnings.filterwarnings('ignore', category=DeprecationWarning)\n",
    "\n",
    "import pandas as pd\n",
    "pd.options.display.max_columns = 100\n",
    "\n",
    "from matplotlib import pyplot as plt\n",
    "import numpy as np\n",
    "\n",
    "import seaborn as sns\n",
    "\n",
    "import pylab as plot\n",
    "params = { \n",
    "    'axes.labelsize': \"large\",\n",
    "    'xtick.labelsize': 'x-large',\n",
    "    'legend.fontsize': 20,\n",
    "    'figure.dpi': 150,\n",
    "    'figure.figsize': [25, 7]\n",
    "}\n",
    "plot.rcParams.update(params)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Two datasets are available: a training set and a test set.\n",
    "We'll be using the training set to build our predictive model and the testing set to score it and generate an output file to submit on the Kaggle evaluation system.\n",
    "\n",
    "We'll see how this procedure is done at the end of this post.\n",
    "\n",
    "Now let's start by loading the training set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "data = pd.read_csv('./data/train.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(891, 12)\n"
     ]
    }
   ],
   "source": [
    "print data.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We have:\n",
    "\n",
    "- 891 rows\n",
    "- 12 columns"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Pandas allows you to have a sneak peak at your data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style>\n",
       "    .dataframe thead tr:only-child th {\n",
       "        text-align: right;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "4            5         0       3   \n",
       "\n",
       "                                                Name     Sex   Age  SibSp  \\\n",
       "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                           Allen, Mr. William Henry    male  35.0      0   \n",
       "\n",
       "   Parch            Ticket     Fare Cabin Embarked  \n",
       "0      0         A/5 21171   7.2500   NaN        S  \n",
       "1      0          PC 17599  71.2833   C85        C  \n",
       "2      0  STON/O2. 3101282   7.9250   NaN        S  \n",
       "3      0            113803  53.1000  C123        S  \n",
       "4      0            373450   8.0500   NaN        S  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The Survived column is the ** target variable**. If Suvival = 1 the passenger survived, otherwise he's dead. The is the variable we're going to predict.\n",
    "\n",
    "The other variables describe the passengers. They are the **features**.\n",
    "\n",
    "- PassengerId: and id given to each traveler on the boat\n",
    "- Pclass: the passenger class. It has three possible values: 1,2,3 (first, second and third class)\n",
    "- The Name of the passeger\n",
    "- The Sex\n",
    "- The Age\n",
    "- SibSp: number of siblings and spouses traveling with the passenger \n",
    "- Parch: number of parents and children traveling with the passenger\n",
    "- The ticket number\n",
    "- The ticket Fare\n",
    "- The cabin number \n",
    "- The embarkation. This describe three possible areas of the Titanic from which the people embark. Three possible values S,C,Q"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Pandas allows you to a have a high-level simple statistical description of the numerical features.\n",
    "This can be done using the describe method."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style>\n",
       "    .dataframe thead tr:only-child th {\n",
       "        text-align: right;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>714.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>446.000000</td>\n",
       "      <td>0.383838</td>\n",
       "      <td>2.308642</td>\n",
       "      <td>29.699118</td>\n",
       "      <td>0.523008</td>\n",
       "      <td>0.381594</td>\n",
       "      <td>32.204208</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>257.353842</td>\n",
       "      <td>0.486592</td>\n",
       "      <td>0.836071</td>\n",
       "      <td>14.526497</td>\n",
       "      <td>1.102743</td>\n",
       "      <td>0.806057</td>\n",
       "      <td>49.693429</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.420000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>223.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>20.125000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>7.910400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>446.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>28.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>14.454200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>668.500000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>38.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>31.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>80.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>512.329200</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       PassengerId    Survived      Pclass         Age       SibSp  \\\n",
       "count   891.000000  891.000000  891.000000  714.000000  891.000000   \n",
       "mean    446.000000    0.383838    2.308642   29.699118    0.523008   \n",
       "std     257.353842    0.486592    0.836071   14.526497    1.102743   \n",
       "min       1.000000    0.000000    1.000000    0.420000    0.000000   \n",
       "25%     223.500000    0.000000    2.000000   20.125000    0.000000   \n",
       "50%     446.000000    0.000000    3.000000   28.000000    0.000000   \n",
       "75%     668.500000    1.000000    3.000000   38.000000    1.000000   \n",
       "max     891.000000    1.000000    3.000000   80.000000    8.000000   \n",
       "\n",
       "            Parch        Fare  \n",
       "count  891.000000  891.000000  \n",
       "mean     0.381594   32.204208  \n",
       "std      0.806057   49.693429  \n",
       "min      0.000000    0.000000  \n",
       "25%      0.000000    7.910400  \n",
       "50%      0.000000   14.454200  \n",
       "75%      0.000000   31.000000  \n",
       "max      6.000000  512.329200  "
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The count variable shows that 177 values are missing in the Age column.\n",
    "\n",
    "One solution is to fill in the null values with the median age. We could also impute with the mean age but the median is more robust to outliers."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "data['Age'] = data['Age'].fillna(data['Age'].median())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's check the result."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style>\n",
       "    .dataframe thead tr:only-child th {\n",
       "        text-align: right;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>446.000000</td>\n",
       "      <td>0.383838</td>\n",
       "      <td>2.308642</td>\n",
       "      <td>29.361582</td>\n",
       "      <td>0.523008</td>\n",
       "      <td>0.381594</td>\n",
       "      <td>32.204208</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>257.353842</td>\n",
       "      <td>0.486592</td>\n",
       "      <td>0.836071</td>\n",
       "      <td>13.019697</td>\n",
       "      <td>1.102743</td>\n",
       "      <td>0.806057</td>\n",
       "      <td>49.693429</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.420000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>223.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>22.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>7.910400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>446.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>28.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>14.454200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>668.500000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>35.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>31.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>80.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>512.329200</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       PassengerId    Survived      Pclass         Age       SibSp  \\\n",
       "count   891.000000  891.000000  891.000000  891.000000  891.000000   \n",
       "mean    446.000000    0.383838    2.308642   29.361582    0.523008   \n",
       "std     257.353842    0.486592    0.836071   13.019697    1.102743   \n",
       "min       1.000000    0.000000    1.000000    0.420000    0.000000   \n",
       "25%     223.500000    0.000000    2.000000   22.000000    0.000000   \n",
       "50%     446.000000    0.000000    3.000000   28.000000    0.000000   \n",
       "75%     668.500000    1.000000    3.000000   35.000000    1.000000   \n",
       "max     891.000000    1.000000    3.000000   80.000000    8.000000   \n",
       "\n",
       "            Parch        Fare  \n",
       "count  891.000000  891.000000  \n",
       "mean     0.381594   32.204208  \n",
       "std      0.806057   49.693429  \n",
       "min      0.000000    0.000000  \n",
       "25%      0.000000    7.910400  \n",
       "50%      0.000000   14.454200  \n",
       "75%      0.000000   31.000000  \n",
       "max      6.000000  512.329200  "
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Perfect.\n",
    "\n",
    "Let's now make some charts.\n",
    "\n",
    "Let's visualize survival based on the gender."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "data['Died'] = 1 - data['Survived']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "data.groupby('Sex').agg('sum')[['Survived', 'Died']].plot(kind='bar', figsize=(25, 7),\n",
    "                                                          stacked=True, colors=['g', 'r']);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![energy](./images/article_1/1.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "It looks like male passengers are more likely to succumb.\n",
    "\n",
    "Let's plot the same graph but with ratio instead."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "data.groupby('Sex').agg('mean')[['Survived', 'Died']].plot(kind='bar', figsize=(25, 7), \n",
    "                                                           stacked=True, colors=['g', 'r']);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![energy](./images/article_1/2.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The Sex variable seems to be a discriminative feature. Women are more likely to survive."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's now correlate the survival with the age variable."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig = plt.figure(figsize=(25, 7))\n",
    "sns.violinplot(x='Sex', y='Age', \n",
    "               hue='Survived', data=data, \n",
    "               split=True,\n",
    "               palette={0: \"r\", 1: \"g\"}\n",
    "              );"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![energy](./images/article_1/3.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "As we saw in the chart above and validate by the following:\n",
    "\n",
    "- Women survive more than men, as depicted by the larger female green histogram \n",
    "\n",
    "Now, we see that:\n",
    "- The age conditions the survival for male passengers:\n",
    "    - Younger male tend to survive\n",
    "    - A large number of passengers between 20 and 40 succumb\n",
    "    \n",
    "- The age doesn't seem to have a direct impact on the female survival"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "These violin plots confirm that one old code of conduct that sailors and captains follow in case of threatening situations: <b>\"Women and children first !\"</b>."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![energy](./images/article_1/titanic.jpg)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Right?"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's now focus on the Fare ticket of each passenger and see how it could impact the survival. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "figure = plt.figure(figsize=(25, 7))\n",
    "plt.hist([data[data['Survived'] == 1]['Fare'], data[data['Survived'] == 0]['Fare']], \n",
    "         stacked=True, color = ['g','r'],\n",
    "         bins = 50, label = ['Survived','Dead'])\n",
    "plt.xlabel('Fare')\n",
    "plt.ylabel('Number of passengers')\n",
    "plt.legend();"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![energy](./images/article_1/4.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Passengers with cheaper ticket fares are more likely to die. \n",
    "Put differently, passengers with more expensive tickets, and therefore a more important social status, seem to be rescued first."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Ok this is nice. Let's now combine the age, the fare and the survival on a single chart."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(25, 7))\n",
    "ax = plt.subplot()\n",
    "\n",
    "ax.scatter(data[data['Survived'] == 1]['Age'], data[data['Survived'] == 1]['Fare'], \n",
    "           c='green', s=data[data['Survived'] == 1]['Fare'])\n",
    "ax.scatter(data[data['Survived'] == 0]['Age'], data[data['Survived'] == 0]['Fare'], \n",
    "           c='red', s=data[data['Survived'] == 0]['Fare']);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![energy](./images/article_1/5.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The size of the circles is proportional to the ticket fare.\n",
    "\n",
    "On the x-axis, we have the ages and the y-axis, we consider the ticket fare.\n",
    "\n",
    "We can observe different clusters:\n",
    "\n",
    "1. Large green dots between x=20 and x=45: adults with the largest ticket fares\n",
    "2. Small red dots between x=10 and x=45, adults from lower classes on the boat\n",
    "3. Small greed dots between x=0 and x=7: these are the children that were saved"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "As a matter of fact, the ticket fare correlates with the class as we see it in the chart below. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "ax = plt.subplot()\n",
    "ax.set_ylabel('Average fare')\n",
    "data.groupby('Pclass').mean()['Fare'].plot(kind='bar', figsize=(25, 7), ax = ax);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![energy](./images/article_1/6.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's now see how the embarkation site affects the survival."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig = plt.figure(figsize=(25, 7))\n",
    "sns.violinplot(x='Embarked', y='Fare', hue='Survived', data=data, split=True, palette={0: \"r\", 1: \"g\"});"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![energy](./images/article_1/7.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "It seems that the embarkation C have a wider range of fare tickets and therefore the passengers who pay the highest prices are those who survive.\n",
    "\n",
    "We also see this happening in embarkation S and less in embarkation Q.\n",
    "\n",
    "Let's now stop with data exploration and switch to the next part."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# II - Feature engineering"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In the previous part, we flirted with the data and spotted some interesting correlations.\n",
    "\n",
    "In this part, we'll see how to process and transform these variables in such a way the data becomes manageable by a machine learning algorithm.\n",
    "\n",
    "We'll also create, or \"engineer\" additional features that will be useful in building the model.\n",
    "\n",
    "We'll see along the way how to process text variables like the passenger names and integrate this information in our model.\n",
    "\n",
    "We will break our code in separate functions for more clarity."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "But first, let's define a print function that asserts whether or not a feature has been processed. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def status(feature):\n",
    "    print 'Processing', feature, ': ok'"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "###  Loading the data\n",
    "\n",
    "One trick when starting a machine learning problem is to append the training set to the test set together.\n",
    "\n",
    "We'll engineer new features using the train set to prevent information leakage. Then we'll add these variables to the test set.\n",
    "\n",
    "Let's load the train and test sets and append them together."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def get_combined_data():\n",
    "    # reading train data\n",
    "    train = pd.read_csv('./data/train.csv')\n",
    "    \n",
    "    # reading test data\n",
    "    test = pd.read_csv('./data/test.csv')\n",
    "\n",
    "    # extracting and then removing the targets from the training data \n",
    "    targets = train.Survived\n",
    "    train.drop(['Survived'], 1, inplace=True)\n",
    "    \n",
    "\n",
    "    # merging train data and test data for future feature engineering\n",
    "    # we'll also remove the PassengerID since this is not an informative feature\n",
    "    combined = train.append(test)\n",
    "    combined.reset_index(inplace=True)\n",
    "    combined.drop(['index', 'PassengerId'], inplace=True, axis=1)\n",
    "    \n",
    "    return combined"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "combined = get_combined_data()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's have a look at the shape :"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(1309, 10)\n"
     ]
    }
   ],
   "source": [
    "print combined.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "train and test sets are combined.\n",
    "\n",
    "You may notice that the total number of rows (1309) is the exact summation of the number of rows in the train set and the test set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style>\n",
       "    .dataframe thead tr:only-child th {\n",
       "        text-align: right;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Pclass                                               Name     Sex   Age  \\\n",
       "0       3                            Braund, Mr. Owen Harris    male  22.0   \n",
       "1       1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0   \n",
       "2       3                             Heikkinen, Miss. Laina  female  26.0   \n",
       "3       1       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0   \n",
       "4       3                           Allen, Mr. William Henry    male  35.0   \n",
       "\n",
       "   SibSp  Parch            Ticket     Fare Cabin Embarked  \n",
       "0      1      0         A/5 21171   7.2500   NaN        S  \n",
       "1      1      0          PC 17599  71.2833   C85        C  \n",
       "2      0      0  STON/O2. 3101282   7.9250   NaN        S  \n",
       "3      1      0            113803  53.1000  C123        S  \n",
       "4      0      0            373450   8.0500   NaN        S  "
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "combined.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Extracting the passenger titles\n",
    "\n",
    "When looking at the passenger names one could wonder how to process them to extract a useful information.\n",
    "\n",
    "If you look closely at these first examples: \n",
    "\n",
    "- Braund, <b> Mr.</b> Owen Harris\t\n",
    "- Heikkinen, <b>Miss.</b> Laina\n",
    "- Oliva y Ocana, <b>Dona.</b> Fermina\n",
    "- Peter, <b>Master.</b> Michael J\n",
    "\n",
    "You will notice that each name has a title in it ! This can be a simple Miss. or Mrs. but it can be sometimes something more sophisticated like Master, Sir or Dona. In that case, we might introduce an additional information about the social status by simply parsing the name and extracting the title and converting to a binary variable.\n",
    "\n",
    "Let's see how we'll do that in the function below."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's first see what the different titles are in the train set "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "titles = set()\n",
    "for name in data['Name']:\n",
    "    titles.add(name.split(',')[1].split('.')[0].strip())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "set(['Sir', 'Major', 'the Countess', 'Don', 'Mlle', 'Capt', 'Dr', 'Lady', 'Rev', 'Mrs', 'Jonkheer', 'Master', 'Ms', 'Mr', 'Mme', 'Miss', 'Col'])\n"
     ]
    }
   ],
   "source": [
    "print titles"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "Title_Dictionary = {\n",
    "    \"Capt\": \"Officer\",\n",
    "    \"Col\": \"Officer\",\n",
    "    \"Major\": \"Officer\",\n",
    "    \"Jonkheer\": \"Royalty\",\n",
    "    \"Don\": \"Royalty\",\n",
    "    \"Sir\" : \"Royalty\",\n",
    "    \"Dr\": \"Officer\",\n",
    "    \"Rev\": \"Officer\",\n",
    "    \"the Countess\":\"Royalty\",\n",
    "    \"Mme\": \"Mrs\",\n",
    "    \"Mlle\": \"Miss\",\n",
    "    \"Ms\": \"Mrs\",\n",
    "    \"Mr\" : \"Mr\",\n",
    "    \"Mrs\" : \"Mrs\",\n",
    "    \"Miss\" : \"Miss\",\n",
    "    \"Master\" : \"Master\",\n",
    "    \"Lady\" : \"Royalty\"\n",
    "}\n",
    "\n",
    "def get_titles():\n",
    "    # we extract the title from each name\n",
    "    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())\n",
    "    \n",
    "    # a map of more aggregated title\n",
    "    # we map each title\n",
    "    combined['Title'] = combined.Title.map(Title_Dictionary)\n",
    "    status('Title')\n",
    "    return combined"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This function parses the names and extract the titles. Then, it maps the titles to categories of titles. \n",
    "We selected : \n",
    "\n",
    "- Officer\n",
    "- Royalty \n",
    "- Mr\n",
    "- Mrs\n",
    "- Miss\n",
    "- Master\n",
    "\n",
    "Let's run it !"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing Title : ok\n"
     ]
    }
   ],
   "source": [
    "combined = get_titles()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style>\n",
       "    .dataframe thead tr:only-child th {\n",
       "        text-align: right;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>Title</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>Mr</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "      <td>Mrs</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>Miss</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "      <td>Mrs</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>Mr</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Pclass                                               Name     Sex   Age  \\\n",
       "0       3                            Braund, Mr. Owen Harris    male  22.0   \n",
       "1       1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0   \n",
       "2       3                             Heikkinen, Miss. Laina  female  26.0   \n",
       "3       1       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0   \n",
       "4       3                           Allen, Mr. William Henry    male  35.0   \n",
       "\n",
       "   SibSp  Parch            Ticket     Fare Cabin Embarked Title  \n",
       "0      1      0         A/5 21171   7.2500   NaN        S    Mr  \n",
       "1      1      0          PC 17599  71.2833   C85        C   Mrs  \n",
       "2      0      0  STON/O2. 3101282   7.9250   NaN        S  Miss  \n",
       "3      1      0            113803  53.1000  C123        S   Mrs  \n",
       "4      0      0            373450   8.0500   NaN        S    Mr  "
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "combined.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's check if the titles have been filled correctly."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style>\n",
       "    .dataframe thead tr:only-child th {\n",
       "        text-align: right;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>Title</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1305</th>\n",
       "      <td>1</td>\n",
       "      <td>Oliva y Ocana, Dona. Fermina</td>\n",
       "      <td>female</td>\n",
       "      <td>39.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17758</td>\n",
       "      <td>108.9</td>\n",
       "      <td>C105</td>\n",
       "      <td>C</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      Pclass                          Name     Sex   Age  SibSp  Parch  \\\n",
       "1305       1  Oliva y Ocana, Dona. Fermina  female  39.0      0      0   \n",
       "\n",
       "        Ticket   Fare Cabin Embarked Title  \n",
       "1305  PC 17758  108.9  C105        C   NaN  "
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "combined[combined['Title'].isnull()]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "There is indeed a NaN value in the line 1305. In fact the corresponding name is Oliva y Ocana, **Dona**. Fermina.\n",
    "\n",
    "This title was not encoutered in the train dataset."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Perfect. Now we have an additional column called <b>Title</b> that contains the information."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "### Processing the ages\n",
    "\n",
    "We have seen in the first part that the Age variable was missing 177 values. This is a large number ( ~ 13% of the dataset). Simply replacing them with the mean or the median age might not be the best solution since the age may differ by groups and categories of passengers. \n",
    "\n",
    "To understand why, let's group our dataset by sex, Title and passenger class and for each subset compute the median age.\n",
    "\n",
    "To avoid data leakage from the test set, we fill in missing ages in the train using the train set and we fill in ages in the test set using values calculated from the train set as well."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Number of missing ages in train set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "177\n"
     ]
    }
   ],
   "source": [
    "print combined.iloc[:891].Age.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Number of missing ages in test set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "86\n"
     ]
    }
   ],
   "source": [
    "print combined.iloc[891:].Age.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "grouped_train = combined.iloc[:891].groupby(['Sex','Pclass','Title'])\n",
    "grouped_median_train = grouped_train.median()\n",
    "grouped_median_train = grouped_median_train.reset_index()[['Sex', 'Pclass', 'Title', 'Age']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style>\n",
       "    .dataframe thead tr:only-child th {\n",
       "        text-align: right;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Sex</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Title</th>\n",
       "      <th>Age</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>female</td>\n",
       "      <td>1</td>\n",
       "      <td>Miss</td>\n",
       "      <td>30.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>female</td>\n",
       "      <td>1</td>\n",
       "      <td>Mrs</td>\n",
       "      <td>40.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>female</td>\n",
       "      <td>1</td>\n",
       "      <td>Officer</td>\n",
       "      <td>49.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>female</td>\n",
       "      <td>1</td>\n",
       "      <td>Royalty</td>\n",
       "      <td>40.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>female</td>\n",
       "      <td>2</td>\n",
       "      <td>Miss</td>\n",
       "      <td>24.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      Sex  Pclass    Title   Age\n",
       "0  female       1     Miss  30.0\n",
       "1  female       1      Mrs  40.0\n",
       "2  female       1  Officer  49.0\n",
       "3  female       1  Royalty  40.5\n",
       "4  female       2     Miss  24.0"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "grouped_median_train.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This dataframe will help us impute missing age values based on different criteria."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Look at the median age column and see how this value can be different based on the Sex, Pclass and Title put together.\n",
    "\n",
    "For example: \n",
    "\n",
    "- If the passenger is female, from Pclass 1, and from royalty the median age is 40.5.\n",
    "- If the passenger is male, from Pclass 3, with a Mr title, the median age is 26.\n",
    "\n",
    "Let's create a function that fills in the missing age in <b>combined</b> based on these different attributes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def fill_age(row):\n",
    "    condition = (\n",
    "        (grouped_median_train['Sex'] == row['Sex']) & \n",
    "        (grouped_median_train['Title'] == row['Title']) & \n",
    "        (grouped_median_train['Pclass'] == row['Pclass'])\n",
    "    ) \n",
    "    return grouped_median_train[condition]['Age'].values[0]\n",
    "\n",
    "\n",
    "def process_age():\n",
    "    global combined\n",
    "    # a function that fills the missing values of the Age variable\n",
    "    combined['Age'] = combined.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)\n",
    "    status('age')\n",
    "    return combined"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing age : ok\n"
     ]
    }
   ],
   "source": [
    "combined = process_age()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Perfect. The missing ages have been replaced. \n",
    "\n",
    "However, we notice a missing value in Fare, two missing values in Embarked and a lot of missing values in Cabin. We'll come back to these variables later.\n",
    "\n",
    "Let's now process the names."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def process_names():\n",
    "    global combined\n",
    "    # we clean the Name variable\n",
    "    combined.drop('Name', axis=1, inplace=True)\n",
    "    \n",
    "    # encoding in dummy variable\n",
    "    titles_dummies = pd.get_dummies(combined['Title'], prefix='Title')\n",
    "    combined = pd.concat([combined, titles_dummies], axis=1)\n",
    "    \n",
    "    # removing the title variable\n",
    "    combined.drop('Title', axis=1, inplace=True)\n",
    "    \n",
    "    status('names')\n",
    "    return combined"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This function drops the Name column since we won't be using it anymore because we created a Title column.\n",
    "\n",
    "Then we encode the title values using a dummy encoding.\n",
    "\n",
    "You can learn about dummy coding and how to easily do it in Pandas <a href=\"http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html\">here</a>.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing names : ok\n"
     ]
    }
   ],
   "source": [
    "combined = process_names()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style>\n",
       "    .dataframe thead tr:only-child th {\n",
       "        text-align: right;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>Title_Master</th>\n",
       "      <th>Title_Miss</th>\n",
       "      <th>Title_Mr</th>\n",
       "      <th>Title_Mrs</th>\n",
       "      <th>Title_Officer</th>\n",
       "      <th>Title_Royalty</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Pclass     Sex   Age  SibSp  Parch            Ticket     Fare Cabin  \\\n",
       "0       3    male  22.0      1      0         A/5 21171   7.2500   NaN   \n",
       "1       1  female  38.0      1      0          PC 17599  71.2833   C85   \n",
       "2       3  female  26.0      0      0  STON/O2. 3101282   7.9250   NaN   \n",
       "3       1  female  35.0      1      0            113803  53.1000  C123   \n",
       "4       3    male  35.0      0      0            373450   8.0500   NaN   \n",
       "\n",
       "  Embarked  Title_Master  Title_Miss  Title_Mr  Title_Mrs  Title_Officer  \\\n",
       "0        S             0           0         1          0              0   \n",
       "1        C             0           0         0          1              0   \n",
       "2        S             0           1         0          0              0   \n",
       "3        S             0           0         0          1              0   \n",
       "4        S             0           0         1          0              0   \n",
       "\n",
       "   Title_Royalty  \n",
       "0              0  \n",
       "1              0  \n",
       "2              0  \n",
       "3              0  \n",
       "4              0  "
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "combined.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "As you can see : \n",
    "- there is no longer a name feature. \n",
    "- new variables (Title_X) appeared. These features are binary. \n",
    "    - For example, If Title_Mr = 1, the corresponding Title is Mr."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Processing Fare"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's imputed the missing fare value by the average fare computed on the train set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def process_fares():\n",
    "    global combined\n",
    "    # there's one missing fare value - replacing it with the mean.\n",
    "    combined.Fare.fillna(combined.iloc[:891].Fare.mean(), inplace=True)\n",
    "    status('fare')\n",
    "    return combined"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This function simply replaces one missing Fare value by the mean."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing fare : ok\n"
     ]
    }
   ],
   "source": [
    "combined = process_fares()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Processing Embarked"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def process_embarked():\n",
    "    global combined\n",
    "    # two missing embarked values - filling them with the most frequent one in the train  set(S)\n",
    "    combined.Embarked.fillna('S', inplace=True)\n",
    "    # dummy encoding \n",
    "    embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')\n",
    "    combined = pd.concat([combined, embarked_dummies], axis=1)\n",
    "    combined.drop('Embarked', axis=1, inplace=True)\n",
    "    status('embarked')\n",
    "    return combined"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This functions replaces the two missing values of Embarked with the most frequent Embarked value."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing embarked : ok\n"
     ]
    }
   ],
   "source": [
    "combined = process_embarked()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style>\n",
       "    .dataframe thead tr:only-child th {\n",
       "        text-align: right;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Title_Master</th>\n",
       "      <th>Title_Miss</th>\n",
       "      <th>Title_Mr</th>\n",
       "      <th>Title_Mrs</th>\n",
       "      <th>Title_Officer</th>\n",
       "      <th>Title_Royalty</th>\n",
       "      <th>Embarked_C</th>\n",
       "      <th>Embarked_Q</th>\n",
       "      <th>Embarked_S</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Pclass     Sex   Age  SibSp  Parch            Ticket     Fare Cabin  \\\n",
       "0       3    male  22.0      1      0         A/5 21171   7.2500   NaN   \n",
       "1       1  female  38.0      1      0          PC 17599  71.2833   C85   \n",
       "2       3  female  26.0      0      0  STON/O2. 3101282   7.9250   NaN   \n",
       "3       1  female  35.0      1      0            113803  53.1000  C123   \n",
       "4       3    male  35.0      0      0            373450   8.0500   NaN   \n",
       "\n",
       "   Title_Master  Title_Miss  Title_Mr  Title_Mrs  Title_Officer  \\\n",
       "0             0           0         1          0              0   \n",
       "1             0           0         0          1              0   \n",
       "2             0           1         0          0              0   \n",
       "3             0           0         0          1              0   \n",
       "4             0           0         1          0              0   \n",
       "\n",
       "   Title_Royalty  Embarked_C  Embarked_Q  Embarked_S  \n",
       "0              0           0           0           1  \n",
       "1              0           1           0           0  \n",
       "2              0           0           0           1  \n",
       "3              0           0           0           1  \n",
       "4              0           0           0           1  "
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "combined.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Processing Cabin"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "train_cabin, test_cabin = set(), set()\n",
    "\n",
    "for c in combined.iloc[:891]['Cabin']:\n",
    "    try:\n",
    "        train_cabin.add(c[0])\n",
    "    except:\n",
    "        train_cabin.add('U')\n",
    "        \n",
    "for c in combined.iloc[891:]['Cabin']:\n",
    "    try:\n",
    "        test_cabin.add(c[0])\n",
    "    except:\n",
    "        test_cabin.add('U')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "set(['A', 'C', 'B', 'E', 'D', 'G', 'F', 'U', 'T'])\n"
     ]
    }
   ],
   "source": [
    "print train_cabin"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "set(['A', 'C', 'B', 'E', 'D', 'G', 'F', 'U'])\n"
     ]
    }
   ],
   "source": [
    "print test_cabin"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We don't have any cabin letter in the test set that is not present in the train set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def process_cabin():\n",
    "    global combined    \n",
    "    # replacing missing cabins with U (for Uknown)\n",
    "    combined.Cabin.fillna('U', inplace=True)\n",
    "    \n",
    "    # mapping each Cabin value with the cabin letter\n",
    "    combined['Cabin'] = combined['Cabin'].map(lambda c: c[0])\n",
    "    \n",
    "    # dummy encoding ...\n",
    "    cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')    \n",
    "    combined = pd.concat([combined, cabin_dummies], axis=1)\n",
    "\n",
    "    combined.drop('Cabin', axis=1, inplace=True)\n",
    "    status('cabin')\n",
    "    return combined"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This function replaces NaN values with U (for <i>Unknow</i>). It then maps each Cabin value to the first letter.\n",
    "Then it encodes the cabin values using dummy encoding again."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing cabin : ok\n"
     ]
    }
   ],
   "source": [
    "combined = process_cabin()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Ok no missing values now."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style>\n",
       "    .dataframe thead tr:only-child th {\n",
       "        text-align: right;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Title_Master</th>\n",
       "      <th>Title_Miss</th>\n",
       "      <th>Title_Mr</th>\n",
       "      <th>Title_Mrs</th>\n",
       "      <th>Title_Officer</th>\n",
       "      <th>Title_Royalty</th>\n",
       "      <th>Embarked_C</th>\n",
       "      <th>Embarked_Q</th>\n",
       "      <th>Embarked_S</th>\n",
       "      <th>Cabin_A</th>\n",
       "      <th>Cabin_B</th>\n",
       "      <th>Cabin_C</th>\n",
       "      <th>Cabin_D</th>\n",
       "      <th>Cabin_E</th>\n",
       "      <th>Cabin_F</th>\n",
       "      <th>Cabin_G</th>\n",
       "      <th>Cabin_T</th>\n",
       "      <th>Cabin_U</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Pclass     Sex   Age  SibSp  Parch            Ticket     Fare  \\\n",
       "0       3    male  22.0      1      0         A/5 21171   7.2500   \n",
       "1       1  female  38.0      1      0          PC 17599  71.2833   \n",
       "2       3  female  26.0      0      0  STON/O2. 3101282   7.9250   \n",
       "3       1  female  35.0      1      0            113803  53.1000   \n",
       "4       3    male  35.0      0      0            373450   8.0500   \n",
       "\n",
       "   Title_Master  Title_Miss  Title_Mr  Title_Mrs  Title_Officer  \\\n",
       "0             0           0         1          0              0   \n",
       "1             0           0         0          1              0   \n",
       "2             0           1         0          0              0   \n",
       "3             0           0         0          1              0   \n",
       "4             0           0         1          0              0   \n",
       "\n",
       "   Title_Royalty  Embarked_C  Embarked_Q  Embarked_S  Cabin_A  Cabin_B  \\\n",
       "0              0           0           0           1        0        0   \n",
       "1              0           1           0           0        0        0   \n",
       "2              0           0           0           1        0        0   \n",
       "3              0           0           0           1        0        0   \n",
       "4              0           0           0           1        0        0   \n",
       "\n",
       "   Cabin_C  Cabin_D  Cabin_E  Cabin_F  Cabin_G  Cabin_T  Cabin_U  \n",
       "0        0        0        0        0        0        0        1  \n",
       "1        1        0        0        0        0        0        0  \n",
       "2        0        0        0        0        0        0        1  \n",
       "3        1        0        0        0        0        0        0  \n",
       "4        0        0        0        0        0        0        1  "
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "combined.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Processing Sex"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def process_sex():\n",
    "    global combined\n",
    "    # mapping string values to numerical one \n",
    "    combined['Sex'] = combined['Sex'].map({'male':1, 'female':0})\n",
    "    status('Sex')\n",
    "    return combined"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This function maps the string values male and female to 1 and 0 respectively. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing Sex : ok\n"
     ]
    }
   ],
   "source": [
    "combined = process_sex()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Processing Pclass"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def process_pclass():\n",
    "    \n",
    "    global combined\n",
    "    # encoding into 3 categories:\n",
    "    pclass_dummies = pd.get_dummies(combined['Pclass'], prefix=\"Pclass\")\n",
    "    \n",
    "    # adding dummy variable\n",
    "    combined = pd.concat([combined, pclass_dummies],axis=1)\n",
    "    \n",
    "    # removing \"Pclass\"\n",
    "    combined.drop('Pclass',axis=1,inplace=True)\n",
    "    \n",
    "    status('Pclass')\n",
    "    return combined"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This function encodes the values of Pclass (1,2,3) using a dummy encoding."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing Pclass : ok\n"
     ]
    }
   ],
   "source": [
    "combined = process_pclass()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Processing Ticket"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's first see how the different ticket prefixes we have in our dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def cleanTicket(ticket):\n",
    "    ticket = ticket.replace('.', '')\n",
    "    ticket = ticket.replace('/', '')\n",
    "    ticket = ticket.split()\n",
    "    ticket = map(lambda t : t.strip(), ticket)\n",
    "    ticket = list(filter(lambda t : not t.isdigit(), ticket))\n",
    "    if len(ticket) > 0:\n",
    "        return ticket[0]\n",
    "    else: \n",
    "        return 'XXX'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "tickets = set()\n",
    "for t in combined['Ticket']:\n",
    "    tickets.add(cleanTicket(t))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "37\n"
     ]
    }
   ],
   "source": [
    "print len(tickets)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def process_ticket():\n",
    "    \n",
    "    global combined\n",
    "    \n",
    "    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)\n",
    "    def cleanTicket(ticket):\n",
    "        ticket = ticket.replace('.','')\n",
    "        ticket = ticket.replace('/','')\n",
    "        ticket = ticket.split()\n",
    "        ticket = map(lambda t : t.strip(), ticket)\n",
    "        ticket = filter(lambda t : not t.isdigit(), ticket)\n",
    "        if len(ticket) > 0:\n",
    "            return ticket[0]\n",
    "        else: \n",
    "            return 'XXX'\n",
    "    \n",
    "\n",
    "    # Extracting dummy variables from tickets:\n",
    "\n",
    "    combined['Ticket'] = combined['Ticket'].map(cleanTicket)\n",
    "    tickets_dummies = pd.get_dummies(combined['Ticket'], prefix='Ticket')\n",
    "    combined = pd.concat([combined, tickets_dummies], axis=1)\n",
    "    combined.drop('Ticket', inplace=True, axis=1)\n",
    "\n",
    "    status('Ticket')\n",
    "    return combined"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing Ticket : ok\n"
     ]
    }
   ],
   "source": [
    "combined = process_ticket()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Processing Family"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This part includes creating new variables based on the size of the family (the size is by the way, another variable we create).\n",
    "\n",
    "This creation of new variables is done under a realistic assumption: Large families are grouped together, hence they are more likely to get rescued than people traveling alone."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def process_family():\n",
    "    \n",
    "    global combined\n",
    "    # introducing a new feature : the size of families (including the passenger)\n",
    "    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1\n",
    "    \n",
    "    # introducing other features based on the family size\n",
    "    combined['Singleton'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)\n",
    "    combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)\n",
    "    combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5 <= s else 0)\n",
    "    \n",
    "    status('family')\n",
    "    return combined"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This function introduces 4 new features: \n",
    "\n",
    "- FamilySize : the total number of relatives including the passenger (him/her)self.\n",
    "- Sigleton : a boolean variable that describes families of size = 1\n",
    "- SmallFamily : a boolean variable that describes families of 2 <= size <= 4\n",
    "- LargeFamily : a boolean variable that describes families of 5 < size"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing family : ok\n"
     ]
    }
   ],
   "source": [
    "combined = process_family()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(1309, 67)\n"
     ]
    }
   ],
   "source": [
    "print combined.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We end up with a total of 67 features. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style>\n",
       "    .dataframe thead tr:only-child th {\n",
       "        text-align: right;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Title_Master</th>\n",
       "      <th>Title_Miss</th>\n",
       "      <th>Title_Mr</th>\n",
       "      <th>Title_Mrs</th>\n",
       "      <th>Title_Officer</th>\n",
       "      <th>Title_Royalty</th>\n",
       "      <th>Embarked_C</th>\n",
       "      <th>Embarked_Q</th>\n",
       "      <th>Embarked_S</th>\n",
       "      <th>Cabin_A</th>\n",
       "      <th>Cabin_B</th>\n",
       "      <th>Cabin_C</th>\n",
       "      <th>Cabin_D</th>\n",
       "      <th>Cabin_E</th>\n",
       "      <th>Cabin_F</th>\n",
       "      <th>Cabin_G</th>\n",
       "      <th>Cabin_T</th>\n",
       "      <th>Cabin_U</th>\n",
       "      <th>Pclass_1</th>\n",
       "      <th>Pclass_2</th>\n",
       "      <th>Pclass_3</th>\n",
       "      <th>Ticket_A</th>\n",
       "      <th>Ticket_A4</th>\n",
       "      <th>Ticket_A5</th>\n",
       "      <th>Ticket_AQ3</th>\n",
       "      <th>Ticket_AQ4</th>\n",
       "      <th>Ticket_AS</th>\n",
       "      <th>Ticket_C</th>\n",
       "      <th>Ticket_CA</th>\n",
       "      <th>Ticket_CASOTON</th>\n",
       "      <th>Ticket_FC</th>\n",
       "      <th>Ticket_FCC</th>\n",
       "      <th>Ticket_Fa</th>\n",
       "      <th>Ticket_LINE</th>\n",
       "      <th>Ticket_LP</th>\n",
       "      <th>Ticket_PC</th>\n",
       "      <th>Ticket_PP</th>\n",
       "      <th>Ticket_PPP</th>\n",
       "      <th>Ticket_SC</th>\n",
       "      <th>Ticket_SCA3</th>\n",
       "      <th>Ticket_SCA4</th>\n",
       "      <th>Ticket_SCAH</th>\n",
       "      <th>Ticket_SCOW</th>\n",
       "      <th>Ticket_SCPARIS</th>\n",
       "      <th>Ticket_SCParis</th>\n",
       "      <th>Ticket_SOC</th>\n",
       "      <th>Ticket_SOP</th>\n",
       "      <th>Ticket_SOPP</th>\n",
       "      <th>Ticket_SOTONO2</th>\n",
       "      <th>Ticket_SOTONOQ</th>\n",
       "      <th>Ticket_SP</th>\n",
       "      <th>Ticket_STONO</th>\n",
       "      <th>Ticket_STONO2</th>\n",
       "      <th>Ticket_STONOQ</th>\n",
       "      <th>Ticket_SWPP</th>\n",
       "      <th>Ticket_WC</th>\n",
       "      <th>Ticket_WEP</th>\n",
       "      <th>Ticket_XXX</th>\n",
       "      <th>FamilySize</th>\n",
       "      <th>Singleton</th>\n",
       "      <th>SmallFamily</th>\n",
       "      <th>LargeFamily</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Sex   Age  SibSp  Parch     Fare  Title_Master  Title_Miss  Title_Mr  \\\n",
       "0    1  22.0      1      0   7.2500             0           0         1   \n",
       "1    0  38.0      1      0  71.2833             0           0         0   \n",
       "2    0  26.0      0      0   7.9250             0           1         0   \n",
       "3    0  35.0      1      0  53.1000             0           0         0   \n",
       "4    1  35.0      0      0   8.0500             0           0         1   \n",
       "\n",
       "   Title_Mrs  Title_Officer  Title_Royalty  Embarked_C  Embarked_Q  \\\n",
       "0          0              0              0           0           0   \n",
       "1          1              0              0           1           0   \n",
       "2          0              0              0           0           0   \n",
       "3          1              0              0           0           0   \n",
       "4          0              0              0           0           0   \n",
       "\n",
       "   Embarked_S  Cabin_A  Cabin_B  Cabin_C  Cabin_D  Cabin_E  Cabin_F  Cabin_G  \\\n",
       "0           1        0        0        0        0        0        0        0   \n",
       "1           0        0        0        1        0        0        0        0   \n",
       "2           1        0        0        0        0        0        0        0   \n",
       "3           1        0        0        1        0        0        0        0   \n",
       "4           1        0        0        0        0        0        0        0   \n",
       "\n",
       "   Cabin_T  Cabin_U  Pclass_1  Pclass_2  Pclass_3  Ticket_A  Ticket_A4  \\\n",
       "0        0        1         0         0         1         0          0   \n",
       "1        0        0         1         0         0         0          0   \n",
       "2        0        1         0         0         1         0          0   \n",
       "3        0        0         1         0         0         0          0   \n",
       "4        0        1         0         0         1         0          0   \n",
       "\n",
       "   Ticket_A5  Ticket_AQ3  Ticket_AQ4  Ticket_AS  Ticket_C  Ticket_CA  \\\n",
       "0          1           0           0          0         0          0   \n",
       "1          0           0           0          0         0          0   \n",
       "2          0           0           0          0         0          0   \n",
       "3          0           0           0          0         0          0   \n",
       "4          0           0           0          0         0          0   \n",
       "\n",
       "   Ticket_CASOTON  Ticket_FC  Ticket_FCC  Ticket_Fa  Ticket_LINE  Ticket_LP  \\\n",
       "0               0          0           0          0            0          0   \n",
       "1               0          0           0          0            0          0   \n",
       "2               0          0           0          0            0          0   \n",
       "3               0          0           0          0            0          0   \n",
       "4               0          0           0          0            0          0   \n",
       "\n",
       "   Ticket_PC  Ticket_PP  Ticket_PPP  Ticket_SC  Ticket_SCA3  Ticket_SCA4  \\\n",
       "0          0          0           0          0            0            0   \n",
       "1          1          0           0          0            0            0   \n",
       "2          0          0           0          0            0            0   \n",
       "3          0          0           0          0            0            0   \n",
       "4          0          0           0          0            0            0   \n",
       "\n",
       "   Ticket_SCAH  Ticket_SCOW  Ticket_SCPARIS  Ticket_SCParis  Ticket_SOC  \\\n",
       "0            0            0               0               0           0   \n",
       "1            0            0               0               0           0   \n",
       "2            0            0               0               0           0   \n",
       "3            0            0               0               0           0   \n",
       "4            0            0               0               0           0   \n",
       "\n",
       "   Ticket_SOP  Ticket_SOPP  Ticket_SOTONO2  Ticket_SOTONOQ  Ticket_SP  \\\n",
       "0           0            0               0               0          0   \n",
       "1           0            0               0               0          0   \n",
       "2           0            0               0               0          0   \n",
       "3           0            0               0               0          0   \n",
       "4           0            0               0               0          0   \n",
       "\n",
       "   Ticket_STONO  Ticket_STONO2  Ticket_STONOQ  Ticket_SWPP  Ticket_WC  \\\n",
       "0             0              0              0            0          0   \n",
       "1             0              0              0            0          0   \n",
       "2             0              1              0            0          0   \n",
       "3             0              0              0            0          0   \n",
       "4             0              0              0            0          0   \n",
       "\n",
       "   Ticket_WEP  Ticket_XXX  FamilySize  Singleton  SmallFamily  LargeFamily  \n",
       "0           0           0           2          0            1            0  \n",
       "1           0           0           2          0            1            0  \n",
       "2           0           0           1          1            0            0  \n",
       "3           0           1           2          0            1            0  \n",
       "4           0           1           1          1            0            0  "
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "combined.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "# III - Modeling"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this part, we use our knowledge of the passengers based on the features we created and then build a statistical model. You can think of this model as a box that crunches the information of any new passenger and decides whether or not he survives.\n",
    "\n",
    "There is a wide variety of models to use, from logistic regression to decision trees and more sophisticated ones such as random forests and gradient boosted trees.\n",
    "\n",
    "We'll be using Random Forests. Random Froests has proven a great efficiency in Kaggle competitions.\n",
    "\n",
    "For more details about why ensemble methods perform well, you can refer to these posts:\n",
    "\n",
    "- http://mlwave.com/kaggle-ensembling-guide/\n",
    "- http://www.overkillanalytics.net/more-is-always-better-the-power-of-simple-ensembles/\n",
    "\n",
    "Back to our problem, we now have to:\n",
    "\n",
    "1. Break the combined dataset in train set and test set.\n",
    "2. Use the train set to build a predictive model.\n",
    "3. Evaluate the model using the train set.\n",
    "4. Test the model using the test set and generate and output file for the submission.\n",
    "\n",
    "Keep in mind that we'll have to reiterate on 2. and 3. until an acceptable evaluation score is achieved."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's start by importing the useful libraries."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from sklearn.pipeline import make_pipeline\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier\n",
    "from sklearn.feature_selection import SelectKBest\n",
    "from sklearn.model_selection import StratifiedKFold\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "from sklearn.model_selection import cross_val_score\n",
    "from sklearn.feature_selection import SelectFromModel\n",
    "from sklearn.linear_model import LogisticRegression, LogisticRegressionCV"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To evaluate our model we'll be using a 5-fold cross validation with the accuracy since it's the metric that the competition uses in the leaderboard.\n",
    "\n",
    "To do that, we'll define a small scoring function. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def compute_score(clf, X, y, scoring='accuracy'):\n",
    "    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)\n",
    "    return np.mean(xval)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Recovering the train set and the test set from the combined dataset is an easy task."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def recover_train_test_target():\n",
    "    global combined\n",
    "    \n",
    "    targets = pd.read_csv('./data/train.csv', usecols=['Survived'])['Survived'].values\n",
    "    train = combined.iloc[:891]\n",
    "    test = combined.iloc[891:]\n",
    "    \n",
    "    return train, test, targets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "train, test, targets = recover_train_test_target()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Feature selection\n",
    "\n",
    "We've come up to more than 30 features so far. This number is quite large. \n",
    "\n",
    "When feature engineering is done, we usually tend to decrease the dimensionality by selecting the \"right\" number of features that capture the essential.\n",
    "\n",
    "In fact, feature selection comes with many benefits:\n",
    "\n",
    "- It decreases redundancy among the data\n",
    "- It speeds up the training process\n",
    "- It reduces overfitting"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Tree-based estimators can be used to compute feature importances, which in turn can be used to discard irrelevant features."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')\n",
    "clf = clf.fit(train, targets)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's have a look at the importance of each feature."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "features = pd.DataFrame()\n",
    "features['feature'] = train.columns\n",
    "features['importance'] = clf.feature_importances_\n",
    "features.sort_values(by=['importance'], ascending=True, inplace=True)\n",
    "features.set_index('feature', inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [],
   "source": [
    "features.plot(kind='barh', figsize=(25, 25))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![energy](./images/article_1/8.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "As you may notice, there is a great importance linked to Title_Mr, Age, Fare, and Sex. \n",
    "\n",
    "There is also an important correlation with the Passenger_Id.\n",
    "\n",
    "Let's now transform our train set and test set in a more compact datasets. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(891L, 14L)\n"
     ]
    }
   ],
   "source": [
    "model = SelectFromModel(clf, prefit=True)\n",
    "train_reduced = model.transform(train)\n",
    "print train_reduced.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(418L, 14L)\n"
     ]
    }
   ],
   "source": [
    "test_reduced = model.transform(test)\n",
    "print test_reduced.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Yay! Now we're down to a lot less features."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We'll see if we'll use the reduced or the full version of the train set."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Let's try different base models"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "logreg = LogisticRegression()\n",
    "logreg_cv = LogisticRegressionCV()\n",
    "rf = RandomForestClassifier()\n",
    "gboost = GradientBoostingClassifier()\n",
    "\n",
    "models = [logreg, logreg_cv, rf, gboost]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Cross-validation of : <class 'sklearn.linear_model.logistic.LogisticRegression'>\n",
      "CV score = 0.817071431282\n",
      "****\n",
      "Cross-validation of : <class 'sklearn.linear_model.logistic.LogisticRegressionCV'>\n",
      "CV score = 0.819318764148\n",
      "****\n",
      "Cross-validation of : <class 'sklearn.ensemble.forest.RandomForestClassifier'>\n",
      "CV score = 0.805891969854\n",
      "****\n",
      "Cross-validation of : <class 'sklearn.ensemble.gradient_boosting.GradientBoostingClassifier'>\n",
      "CV score = 0.830560996274\n",
      "****\n"
     ]
    }
   ],
   "source": [
    "for model in models:\n",
    "    print 'Cross-validation of : {0}'.format(model.__class__)\n",
    "    score = compute_score(clf=model, X=train_reduced, y=targets, scoring='accuracy')\n",
    "    print 'CV score = {0}'.format(score)\n",
    "    print '****'"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Hyperparameters tuning"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "As mentioned in the beginning of the Modeling part, we will be using a Random Forest model. It may not be the best model for this task but we'll show how to tune. This work can be applied to different models.\n",
    "\n",
    "Random Forest are quite handy. They do however come with some parameters to tweak in order to get an optimal model for the prediction task.\n",
    "\n",
    "To learn more about Random Forests, you can refer to this <a href=\"https://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/\">link </a>: \n",
    "\n",
    "Additionally, we'll use the full train set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {
    "collapsed": true,
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "# turn run_gs to True if you want to run the gridsearch again.\n",
    "run_gs = False\n",
    "\n",
    "if run_gs:\n",
    "    parameter_grid = {\n",
    "                 'max_depth' : [4, 6, 8],\n",
    "                 'n_estimators': [50, 10],\n",
    "                 'max_features': ['sqrt', 'auto', 'log2'],\n",
    "                 'min_samples_split': [2, 3, 10],\n",
    "                 'min_samples_leaf': [1, 3, 10],\n",
    "                 'bootstrap': [True, False],\n",
    "                 }\n",
    "    forest = RandomForestClassifier()\n",
    "    cross_validation = StratifiedKFold(n_splits=5)\n",
    "\n",
    "    grid_search = GridSearchCV(forest,\n",
    "                               scoring='accuracy',\n",
    "                               param_grid=parameter_grid,\n",
    "                               cv=cross_validation,\n",
    "                               verbose=1\n",
    "                              )\n",
    "\n",
    "    grid_search.fit(train, targets)\n",
    "    model = grid_search\n",
    "    parameters = grid_search.best_params_\n",
    "\n",
    "    print('Best score: {}'.format(grid_search.best_score_))\n",
    "    print('Best parameters: {}'.format(grid_search.best_params_))\n",
    "    \n",
    "else: \n",
    "    parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50, \n",
    "                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}\n",
    "    \n",
    "    model = RandomForestClassifier(**parameters)\n",
    "    model.fit(train, targets)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now that the model is built by scanning several combinations of the hyperparameters, we can generate an output file to submit on Kaggle."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "output = model.predict(test).astype(int)\n",
    "df_output = pd.DataFrame()\n",
    "aux = pd.read_csv('./data/test.csv')\n",
    "df_output['PassengerId'] = aux['PassengerId']\n",
    "df_output['Survived'] = output\n",
    "df_output[['PassengerId','Survived']].to_csv('./predictions/gridsearch_rf.csv', index=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### [BONUS]  Blending different models"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "I haven't personally uploaded a submission based on model blending but here's how you could do it"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "trained_models = []\n",
    "for model in models:\n",
    "    model.fit(train, targets)\n",
    "    trained_models.append(model)\n",
    "\n",
    "predictions = []\n",
    "for model in trained_models:\n",
    "    predictions.append(model.predict_proba(test)[:, 1])\n",
    "\n",
    "predictions_df = pd.DataFrame(predictions).T\n",
    "predictions_df['out'] = predictions_df.mean(axis=1)\n",
    "predictions_df['PassengerId'] = aux['PassengerId']\n",
    "predictions_df['out'] = predictions_df['out'].map(lambda s: 1 if s >= 0.5 else 0)\n",
    "\n",
    "predictions_df = predictions_df[['PassengerId', 'out']]\n",
    "predictions_df.columns = ['PassengerId', 'Survived']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "predictions_df.to_csv('./predictions/blending_base_models.csv', index=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To have a good blending submission, the base models should be different and their correlations uncorrelated."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "# IV - Conclusion"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "In this article, we explored an interesting dataset brought to us by <a href=\"http://kaggle.com\">Kaggle</a>.\n",
    "\n",
    "We went through the basic bricks of a data science pipeline:\n",
    "\n",
    "- Data exploration and visualization: an initial step to formulate hypotheses\n",
    "- Data cleaning \n",
    "- Feature engineering \n",
    "- Feature selection\n",
    "- Hyperparameters tuning\n",
    "- Submission\n",
    "- Blending\n",
    "\n",
    "This post can be downloaded as a notebook if you ever want to test and play with it : <a href=\"https://github.com/ahmedbesbes/post1/blob/master/titanic-article.ipynb\"> my github repo </a>\n",
    "\n",
    "Lots of articles have been written about this challenge, so obviously there is a room for improvement.\n",
    "\n",
    "Here is what I suggest for next steps:\n",
    "\n",
    "- Dig more in the data and eventually build new features.\n",
    "- Try different models : logistic regressions, Gradient Boosted trees, XGboost, ...\n",
    "- Try ensemble learning techniques (stacking)\n",
    "- Run auto-ML frameworks\n",
    "\n",
    "I would be more than happy if you could find out a way to improve my solution. This could make me update the article and definitely give you credit for that. So feel free to post a comment."
   ]
  }
 ],
 "metadata": {
  "anaconda-cloud": {},
  "kernelspec": {
   "display_name": "Python [conda root]",
   "language": "python",
   "name": "conda-root-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
