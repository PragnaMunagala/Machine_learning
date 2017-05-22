import json
import pandas as pd
import math

data = pd.read_csv("C:\\Users\\rohit\\Desktop\\Sem1CourseWork\\CSE575\\Project\\dataset\\ACT_DIR_MAPPING.csv", sep='\t', delimiter=';')
dataDict = dict([(i, [genres]) for i, genres in zip(data.movie_id, data.genres)])


ratingData = pd.read_csv("C:\\Users\\rohit\\Desktop\\Sem1CourseWork\\CSE575\\Project\\dataset\\MOVIE_RATING_MAP.csv", sep='\t', delimiter=',')
movieRatingDict = dict([(i,j) for i,j in zip(ratingData.id, ratingData.rtAllCriticsScore)])

with open("F4-CollaborationMatrix.json","r") as fp:
    f4Matrix = fp.read()

f4MatrixDict = json.loads(f4Matrix)

with open("Feature2-OverallQuality.json","r") as f2:
    f2Matrix = f2.read()

f2MatrixDict = json.loads(f2Matrix)


with open("Feature3-GenreWiseQuality.json","r") as f3:
    f3Matrix = f3.read()

f3MatrixDict = json.loads(f3Matrix)


with open('MovieMap.json','r') as md:
    movieData = md.read()

movieDataDict = json.loads(movieData)

a = ["SHORT", "HORROR", "CRIME", "ROMANCE", "DOCUMENTARY", "MUSICAL", "THRILLER", "SCI_FI", "CHILDREN", "FANTASY", "ADVENTURE", "FILM_NOIR", "MYSTERY", "DRAMA", "ACTION", "ANIMATION", "IMAX", "WAR", "COMEDY", "WESTERN"]
newDict = {}

for key,val in movieDataDict.items():
    movieActorList = (val[0].split(','))
    genreList = (dataDict[int(key)][0].split(','))
    sumOfCollaborations = 0
    MovieOverallQuality = 0
    MovieGenreWiseQuality = 0
    cosineDistPairwise = 0
    for i in range(0, movieActorList.__len__()):
        MovieOverallQuality = MovieOverallQuality + f2MatrixDict[movieActorList[i]]
        for genreIndex in range(0, genreList.__len__()):
            MovieGenreWiseQuality = MovieGenreWiseQuality + f3MatrixDict[movieActorList[i]][genreList[genreIndex].upper()] / genreList.__len__()
        for j in range(i+1, movieActorList.__len__()):
            sumOfCollaborations = sumOfCollaborations + f4MatrixDict[movieActorList[i]][movieActorList[j]]
            xy = 0
            xSquared = 0
            ySquared = 0
            for element in a:
                xy = xy + f3MatrixDict[movieActorList[i]][element] * f3MatrixDict[movieActorList[j]][element]
                xSquared = xSquared + math.pow(f3MatrixDict[movieActorList[i]][element],2)
                ySquared = ySquared + math.pow(f3MatrixDict[movieActorList[j]][element],2)
            if  xSquared == 0 or ySquared == 0:
                cosineDistPairwise = cosineDistPairwise + 1
            else:
                cosineDistPairwise = cosineDistPairwise + (1 - (float(xy)/math.sqrt(xSquared * ySquared)))
    newDict[int(key)] = {'F1':cosineDistPairwise/15, 'F2': MovieOverallQuality/6, 'F3': MovieGenreWiseQuality/6, 'F4': sumOfCollaborations/15, 'Rating': int(movieRatingDict[int(key)])}

print(newDict[1])


with open('ModelParameters-F1-F2-F3-F4-Rating.json','w') as mp:
    json.dump(newDict, mp)
