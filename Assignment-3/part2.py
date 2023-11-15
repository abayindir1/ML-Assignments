import requests
import re
import numpy as np
import random

url = "https://raw.githubusercontent.com/abayindir1/ML-Assignments/main/Assignment-3/NBChealth.txt"
response = requests.get(url)
tweets = response.text.splitlines()

# print(tweets)
# print(type(tweets))

def preprocessing(tw):
    parts = tw.split('|')
    # remove id and time
    tweet_text = parts[-1]
    # Remove words starting with @
    tweet_text = re.sub(r'@\w+', '', tweet_text)
    # Remove #
    tweet_text = re.sub(r'#', '', tweet_text)  
    # Remove URLs
    tweet_text = re.sub(r'http\S+', '', tweet_text)  
    # lowercase
    tweet_text = tweet_text.lower()
    return tweet_text

# processing all the tweets
processedTweets = []
for tweet in tweets:
    pt = preprocessing(tweet)
    processedTweets.append(pt)
    # print(f"Original: {tweet}\nProcessed: {pt}\n") 



# calculate jDistance between two tweets
def jDistance(t1, t2):
    set1 = set(t1.split())
    set2 = set(t2.split())
    itrs = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return 1 - (itrs/union)

# perform kmeans clustering
def kmeans(data, k, maxIterations=100):
    # initialize centroids randomly
    centroids = random.sample(data, k)

    for _ in range(maxIterations):
        # assigning data points to centroids
        clusters = {}
        for i in range(k):
            clusters[i] = []

        # assign data points to the nearest centroid
        for tweet in data:
            distances = []
            # distances between the tweet and each centroid
            for centroid in centroids:
                distances.append(jDistance(tweet, centroid))
            # get index of the nearest centroid
            nearestCentroid = distances.index(min(distances))
            clusters[nearestCentroid].append(tweet)

        # update the centroids based on the clusters
        newCentroids = []
        for centroid, cluster in clusters.items():
            if len(cluster) == 0:
                # if a cluster is empty, assign a random data point as the new centroid
                newCentroids.append(random.choice(data))
            else:
                minTweet = min(cluster, key=lambda x:sum(jDistance(x, t) for t in cluster))
                newCentroids.append(minTweet)

        if newCentroids == centroids:
            break

        centroids = newCentroids
    
    return clusters, centroids


kValues = [3, 5, 7, 10, 15]

for k in kValues:
    # run kmeans for tweets
    clusters, centroids = kmeans(processedTweets, k)

    # salculate sum of squared errors
    sse = 0
    for centroid, cluster in zip(centroids, clusters.values()):
        clusterErr = 0
        for tweet in cluster:
            clusterErr += jDistance(tweet, centroid)
        sse += clusterErr
    
    # print results
    print(f"\n\nValue of K: {k} | SSE: {sse}")
    for i, (centroid, cluster) in enumerate(zip(centroids, clusters.values())):
        print(f"Cluster {i + 1} Size: {len(cluster)}")
        print(f"Centroid: {centroid}")
        print("Sample Tweets:")
        for tweet in cluster[:3]:
            print(f" - {tweet}")
