import pandas as pd
import json
from collections import defaultdict

def jacquardDistance(countA, countB):
    return float(1)/float(countA + countB)

# df = pd.read_excel("C:\\Users\\rohit\\Desktop\\Sem1CourseWork\\CSE575\\Project\\dataset\\movieActors.xlsx", sheetname="movie_actors")
# df.as_matrix()
#
# #print df.keys()
# print(np.shape(df))

# data = np.genfromtxt("C:\\Users\\rohit\\Desktop\\Sem1CourseWork\\CSE575\\Project\\dataset\\sml.csv",dtype=None, delimiter=',')
# print (np.shape(data))

actorSet = set()
someDict = defaultdict(dict)
data = pd.read_csv("C:\\Users\\rohit\\Desktop\\Sem1CourseWork\\CSE575\\Project\\dataset\\ACT_DIR_MAPPING.csv", sep='\t', delimiter=';')
actorCount = pd.read_csv("C:\\Users\\rohit\\Desktop\\Sem1CourseWork\\CSE575\\Project\\dataset\\ACTOR_COUNT.csv", sep='\t', delimiter=';')
directorCount = pd.read_csv("C:\\Users\\rohit\\Desktop\\Sem1CourseWork\\CSE575\\Project\\dataset\\DIRECTOR_COUNT.csv", sep='\t', delimiter=';')
print(type(actorCount))
print(type(data))
print (data.keys())
actorCountDict = dict([(i,j) for i,j in zip(actorCount.ACTOR_NAME,actorCount.COUNT)])
directorCountDict = dict([(i,j) for i,j in zip(directorCount.DIRECTOR_NAME,directorCount.COUNT)])
dataDict = dict([(i, [actors]) for i, actors in zip(data.movie_id, data.act_dir_list)])

print(actorCountDict["Leonardo DiCaprio"])
print(actorCountDict["Paterson Joseph"])
print(directorCountDict["Christopher Nolan"])

print (type(dataDict))
print (dataDict.get(1))

actorCountDict.update(directorCountDict)

for actorNames in dataDict.values():
    movieActorList = (actorNames[0].split(','))
    for i in range(0, movieActorList.__len__()):
        for j in range(0, movieActorList.__len__()):
            actorSet.add(movieActorList[i])

#print (actorSet.__len__())
#print (actorSet)

individualMovieCountDict = dict.fromkeys(actorSet, 0)

# for actorKeys in individualMovieCountDict.keys():
#     count = 0
#     for vals in dataDict.values():
#
#         if actorKeys in vals[0]:
#             count += 1
#     individualMovieCountDict[actorKeys] = count


# count = 0
# for vals in dataDict.values():
#     if ("Leonardo DiCaprio" in vals[0]) and ("Paterson Joseph" in vals[0]):
#         count += 1



#collaborationDict = dict.fromkeys(actorSet, dict.fromkeys(actorSet, 0))

# print (collaborationDict.get("Don Rickles").get("Don Rickles"))
# collaborationDict["Don Rickles"]["Don Rickles"] = 1;
# print (collaborationDict.get("Don Rickles").get("Don Rickles"))

collaborationDict = {}

for actorNames in dataDict.values():
    movieActorList = (actorNames[0].split(','))
    for i in range(0, movieActorList.__len__()):
        if movieActorList[i] not in collaborationDict:
            collaborationDict[movieActorList[i]] = {}
        for j in range(0, movieActorList.__len__()):
            if i == j:
                collaborationDict[movieActorList[i]][movieActorList[j]] = 0
                continue
            if movieActorList[j] not in collaborationDict[movieActorList[i]]:
                collaborationDict[movieActorList[i]][movieActorList[j]] = 1 * jacquardDistance(int(actorCountDict[movieActorList[i]]), int(actorCountDict[movieActorList[j]]))
            else:
                collaborationDict[movieActorList[i]][movieActorList[j]] += 1 * jacquardDistance(int(actorCountDict[movieActorList[i]]), int(actorCountDict[movieActorList[j]]))

# for actor1 in individualMovieCountDict.keys():
#     for actor2 in individualMovieCountDict.keys():
#         count = 0
#         for vals in dataDict.values():
#             if(actor1 in vals[0] and actor2 in vals[0]):
#                 count +=1
#         collaborationDict[actor1][actor2] = count


print (collaborationDict.get("Leonardo DiCaprio").get("Paterson Joseph"))
print (collaborationDict.get("Don Rickles").get("Don Rickles"))


with open('F4-CollaborationMatrix.json','w') as fp:
    json.dump(collaborationDict, fp)





