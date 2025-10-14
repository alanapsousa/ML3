import numpy.random as rng 
import numpy as np
class Kmeans:

    def __init__(self, K, max_int = 200, tolerance=0.0001, random_state = 42):
        self.K =  K
        self.max_int = max_int
        self.tolerance = tolerance
        self.random_state = random_state

        

    def fit(self, X):
        self.X = X
    
        rng = np.random.default_rng(self.random_state)

        #inicializing the centroid
        centroids_ini_index = rng.choice(len(self.X), size= self.K, replace=False) # random index for each cluster K 
        centroids = self.X[centroids_ini_index] 


        #max interation control loop
        for i in range(self.max_int):
            distances = np.zeros((self.X.shape[0], self.K))

            #Loop to find the distances beteween each feature with all clusters
            for j in range(self.K): 
                distances[:, j] = np.linalg.norm(self.X - centroids[j], axis=1)

            # add to labels
            labels = np.argmin(distances, axis=1)

            #compute new centroids    
            new_centroids = np.array([self.X[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k]
                for k in range(self.K)])
            
            # Condition to stop loop
            shift = np.linalg.norm(new_centroids - centroids)
            if shift < self.tolerance:
                print(f" stop in the iteration {i}")
                break

            centroids = new_centroids
        
        # save the results
        self.centroids = centroids
        self.labels = labels   

        return self          

    def predict(self, X_new):
        #find the cluster of new data
        distances = np.zeros((X_new.shape[0], self.K))
        for n in range(self.K):
            #compute the distance of each X_new to each cluster
            distances[:, n] = np.linalg.norm(X_new - self.centroids[n], axis=1)
        return np.argmin(distances, axis=1)
        


     


