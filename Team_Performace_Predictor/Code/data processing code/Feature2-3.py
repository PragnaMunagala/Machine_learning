import pandas as pd
import json


actorMatrix = pd.read_csv("C:\\Users\\rohit\\Desktop\\Sem1CourseWork\\CSE575\\Project\\dataset\\ACTORS_SKILL_MATRIX.csv", sep='\t', delimiter=';')
directorMatrix = pd.read_csv("C:\\Users\\rohit\\Desktop\\Sem1CourseWork\\CSE575\\Project\\dataset\\DIRECTORS_SKILL_MATRIX.csv", sep='\t', delimiter=';')


personOverallQuality = dict([(i,j) for i,j in zip(actorMatrix.ACTOR_NAME,actorMatrix.cum_score)])
directorOverallQuality = dict([(i,j) for i,j in zip(directorMatrix.DIRECTOR_NAME,directorMatrix.cum_score)])
personOverallQuality.update(directorOverallQuality)


actorMatrix.set_index("ACTOR_NAME", drop=True, inplace=True)
personGenreWiseQuality = actorMatrix.to_dict(orient="index")

directorMatrix.set_index("DIRECTOR_NAME", drop=True, inplace=True)
directorGenreWiseQuality = directorMatrix.to_dict(orient="index")

personGenreWiseQuality.update(directorGenreWiseQuality)


# with open('Feature2.json', 'w') as f2:
#     json.dump(personOverallQuality, f2)
#
# with open('Feature3.json', 'w') as f3:
#     json.dump(personGenreWiseQuality, f3)

