import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import statistics

np.random.seed(1)

# x1 = sym.Symbol('x1')
# x2 = sym.Symbol('x2')

# X = np.array([x1, x2])
# mean_class1 = np.array([4.09, 3.18]).T
# mean_class2 = np.array([2.97, 11.71]).T

# a = -1*np.matmul(mean_class1, X) + 0.5*np.matmul(mean_class1, mean_class1.T) 
# b = -1*np.matmul(mean_class2, X) + 0.5*np.matmul(mean_class2, mean_class2.T) 

# print(a-b)

# sol = sym.solve(a-b, x2)

# print(sol)
#############################################################################################3

mean_class1 = np.array([4, 7])
mean_class2 = np.array([5, 10])

covariance_class1 = np.array([[9,3],
                              [3,10]])

covariance_class2 = np.array([[7,0],
                              [0,16]])

data_class1_100_samples = np.random.multivariate_normal(mean_class1, covariance_class1, 100)
data_class2_100_samples = np.random.multivariate_normal(mean_class2, covariance_class2, 100)

class1_50_new_samples = np.random.multivariate_normal(mean_class1, covariance_class1, 50)
class2_50_new_samples = np.random.multivariate_normal(mean_class1, covariance_class1, 50)

white_noise = np.random.normal(0, 1, size=(50, 2))

class1_50_new_samples_noisy = class1_50_new_samples + white_noise
class2_50_new_samples_noisy = class2_50_new_samples + white_noise

data_class1_100_samples_labelled = np.c_[data_class1_100_samples, np.zeros(100)]
data_class2_100_samples_labelled = np.c_[data_class2_100_samples, np.ones(100)]

x_class1, y_class1 = np.meshgrid(np.linspace(-1, 10, 500), np.linspace(-1, 12, 500))

def euclidean_dist(a, b):
  return np.sqrt(np.sum((np.array(a) - np.array(b))**2))

# print(euclidean_dist((class1_50_new_samples_noisy[0,0], class1_50_new_samples_noisy[0,1]), (data_class1_100_samples_labelled[0,0], data_class1_100_samples_labelled[0,1])))
# print(data_class1_100_samples_labelled[0][2])
# print(euclidean_dist(class1_50_new_samples_noisy[0], data_class1_100_samples_labelled[0, [0, 1]]))

# print(data_class1_100_samples_labelled[0])
# print(data_class1_100_samples_labelled[:, [0, 1]])

def knn_classifier(k, x_train, y_train, x_test):
  my_dictionary = {}
  for j in range(0, len(x_train)-1):
    my_dictionary.update({euclidean_dist(x_test, x_train[j, [0, 1]]):0})
    my_dictionary.update({euclidean_dist(x_test, y_train[j, [0, 1]]):1})

  print(my_dictionary)
  myKeys = list(my_dictionary.keys())
  myKeys.sort()
  sorted_dict = {i: my_dictionary[i] for i in myKeys}
  print(sorted_dict)
  closest_k_neighbours = list(sorted_dict.items())[:k]
  closest_k_neighbours_arr = np.array(closest_k_neighbours)
  # print(closest_k_neighbours_arr)
  return statistics.mode(closest_k_neighbours_arr[:, -1])

print(class1_50_new_samples_noisy[0])
print(knn_classifier(3, data_class1_100_samples, data_class2_100_samples, class1_50_new_samples_noisy[0]))



# def knn_classifier(k):
#   my_dictionary = {}

#   for i in range(0, 49):
#     for j in range(0, 99):
#       my_dictionary.update({euclidean_dist(class1_50_new_samples_noisy[i], data_class1_100_samples_labelled[j, [0, 1]]):data_class1_100_samples_labelled[j][2]})

#   myKeys = list(my_dictionary.keys())
#   myKeys.sort()
#   sorted_dict = {i: my_dictionary[i] for i in myKeys}
#   closest_k_neighbours = list(sorted_dict.items())[:k]
#   closest_k_neighbours_arr = np.array(closest_k_neighbours)
#   return statistics.mode(closest_k_neighbours_arr[:, -1])

# class knn:
#     def __init__(self, x_train, y_train, k=3):
#         self.k = k
#         self.x_train = x_train
#         self.y_train = y_train
    
#     def euclidean_dist(self, a, b):
#       return np.sqrt(np.sum((np.array(a) - np.array(b))**2))
    
#     def predict(self, x_test):
#        predictions = []
#        for x_test_item in x_test:
#           distances = []
#           for x_train_item in self.x_train:
#              distances.append(self.euclidean_dist(x_train_item, x_test_item))
#           indices = np.argsort(distances)[self.k]
#           nearest_neighbours = self.y_train[indices]

#           count_dict = {}
#           for i in nearest_neighbours:
#              count_dict[i] = list(nearest_neighbours).count(i)
          
#           values = list(count_dict.values())
#           keys = list(count_dict.keys())
#           majority_vote = keys[values.index(max(values))]
          
#           predictions.append(majority_vote)
       
#        return predictions


# clf = knn(data_class1_100_samples, data_class2_100_samples)

# labels = clf.predict(sample_test)

# class knn:
#   def __init__(self, k=3):
#     self.k = k
  
#   def euclidean_dist(self, a, b):
#     return np.sqrt(np.sum((np.array(a) - np.array(b))**2))

# my_classifier = knn(3)

# print(my_classifier.euclidean_dist((class1_50_new_samples_noisy[0,0], class1_50_new_samples_noisy[0,1]), (data_class2_100_samples[0,0], data_class2_100_samples[0,1])))
# print(my_classifier.euclidean_dist((class1_50_new_samples_noisy[0,0], class1_50_new_samples_noisy[0,1]), (data_class1_100_samples[0,0], data_class1_100_samples[0,1])))

# distances = [my_classifier.euclidean_dist((class1_50_new_samples_noisy[0,0], class1_50_new_samples_noisy[0,1]), (data_class2_100_samples[0,0], data_class2_100_samples[0,1])),
#   my_classifier.euclidean_dist((class1_50_new_samples_noisy[0,0], class1_50_new_samples_noisy[0,1]), (data_class1_100_samples[0,0], data_class1_100_samples[0,1]))]

# # print(sorted(distances))

# dist_sorted = sorted(distances)

# dist_sorted_arr = np.array(dist_sorted)

# for i in range(0, 1):
#   print(dist_sorted_arr[i])

# print(data_class1_100_samples)

plt.figure(1)
plt.scatter(data_class1_100_samples_labelled[:,0], data_class1_100_samples_labelled[:,1])
plt.scatter(data_class2_100_samples_labelled[:,0], data_class2_100_samples_labelled[:,1])
# plt.scatter(class1_50_new_samples_noisy, class1_50_new_samples_noisy, s=10, c=knn_classifier(3, data_class1_100_samples, data_class2_100_samples, class1_50_new_samples_noisy), cmap='gray')
# plt.scatter(data_class1_100_samples[0,0], data_class1_100_samples[0,1])
# plt.scatter(data_class2_100_samples[0,0], data_class2_100_samples[0,1])
plt.scatter(class1_50_new_samples_noisy[0,0], class1_50_new_samples_noisy[0,1])
# plt.scatter(class1_50_new_samples_noisy[:,0], class2_50_new_samples_noisy[:,1])
# plt.scatter(class2_50_new_samples_noisy[:,0], class2_50_new_samples_noisy[:,1])
# plt.contourf(x_class1, y_class1, knn_classifier(data_class1_100_samples, data_class2_100_samples, [x_class1, y_class1]))
# plt.legend(["Class 1 Training Data", "Class 2 Training Data", "Class 1 Test Data", "Class 2 Test Data"])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid()
plt.show()

# plt.figure(2)
# for i in x_class1:
#   for j in y_class1:
#     if(knn_classifier(3, data_class1_100_samples, data_class2_100_samples, [i, j])) == 0:
#       plt.scatter(i, j, c='r')
#     else:
#       plt.scatter(i, j, c='b')

