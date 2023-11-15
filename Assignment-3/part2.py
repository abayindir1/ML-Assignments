import requests
import re
import random
import pandas as pd

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
def kmeans(data, k):

    # initialize centroids randomly
    centroids = random.sample(data, k)


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

values_of_k_list = []
sse_error_list = []
size_of_cluster_list = []

for k in kValues:
    # run kmeans for tweets
    clusters, centroids = kmeans(processedTweets, k)

    # calculate sum of squared errors
    sse = 0
    sizes = []

    for centroid, cluster in zip(centroids, clusters.values()):
        cluster_err = 0
        for tweet in cluster:
            cluster_err += jDistance(tweet, centroid)
        sse += cluster_err
        sizes.append(len(cluster))

    # Append results to lists
    values_of_k_list.append(k)
    sse_error_list.append(sse)
    size_of_cluster_list.append(sizes)
# print results
results_table = pd.DataFrame()
results_table["Value_of_K"] = values_of_k_list
results_table["SSE_Error"] = sse_error_list
results_table["Size_of_Clusters"] = size_of_cluster_list
results_table.index = results_table.index + 1

# Print the DataFrame
print(results_table)
