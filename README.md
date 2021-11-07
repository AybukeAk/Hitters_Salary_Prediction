# Salary Prediction with Machine Learning Algorithms

![pexels-tim-eiden-1374370 (1)](images/pexels-tim-eiden-1374370%20(1)-16362978468772.jpg)

## Business Problem

Can a machine learning project be implemented to estimate the salaries of baseball players whose salary information and career statistics for 1986 are shared?

---

## Dataset Story

This dataset was originally taken from the StatLib library at Carnegie Mellon University.

The dataset is part of the data used in the 1988 ASA Graphics Section Poster Session.

Salary data originally from Sports Illustrated, April 20, 1987.

1986 and career statistics are from the 1987 Baseball Encyclopedia Update published by Collier Books, Macmillan Publishing Company, New York.

---

## Variables

```
# AtBat: Number of hits with a baseball bat during the 1986-1987 season
# Hits: Number of hits in the 1986-1987 season
# HmRun: Most valuable hits in the 1986-1987 season
# Runs: The points he earned for his team in the 1986-1987 season
# RBI: Number of players jogged,  when a batsman hit
# Walks: Number of mistakes made by the opposing player
# Years: Player's playing time in major league (years)
# CAtBat: Number of hits during a player's career
# CHits: The number of hits the player has taken throughout his career
# CHmRun: The player's most valuable hit during his career
# CRuns: Points earned by the player during his career
# CRBI: The number of players the player has made during his career
# CWalks: Number of mistakes made by the opposing player during the player's career
# League: A factor with A and N levels showing the league in which the player played until the end of the season
# Division: A factor with levels E and W indicating the position played by the player at the end of 1986
# PutOuts: Helping your teammate in-game
# Assists: Number of assists made by the player in the 1986-1987 season
# Errors: Player's number of errors in the 1986-1987 season
# Salary: The salary of the player in the 1986-1987 season (over thousand)
# NewLeague: a factor with A and N levels indicating the player's league at the start of the 1987 seasonAuthor
```

---

### Libraries

```
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
import warnings
from pandas.core.common import SettingWithCopyWarning
from sklearn.exceptions import ConvergenceWarning
```

---

### Author

**Aybüke Hamide Ak** - [AybukeAk](https://github.com/AybukeAk)

---

## Reference

VBO - Data Science and Machine Learning Bootcamp
[www.veribilimiokulu.com](https://www.veribilimiokulu.com/)





