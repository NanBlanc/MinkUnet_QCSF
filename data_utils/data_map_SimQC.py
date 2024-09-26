# This file is covered by the LICENSE file in the root of this project.
labels = {
  0 : "Ground",
  1 : "AB",
  2: "PM",
  3: "Low Veg"
}
# classes that are indistinguishable from single scan or inconsistent in
# ground truth are mapped to their closest equivalent
learning_map = {
  0: 0,      # "unlabeled"               mapped to "unlabeled" --------------------------mapped
  1: 1,      # "ego vehicle"             mapped to "unlabeled" --------------------------mapped
  2: 2,      # "rectification border"    mapped to "unlabeled" --------------------------mapped
  3: 3
}
learning_map_inv = { # inverse of previous map
  0: 0,      # "unlabeled"               mapped to "unlabeled" --------------------------mapped
  1: 1,      # "ego vehicle"             mapped to "unlabeled" --------------------------mapped
  2: 2,      # "rectification border"    mapped to "unlabeled" --------------------------mapped
  3: 3
}

learning_ignore = { # Ignore classes
  0: False,   
  1: False,   
  2: False,     
  3: False
}


poss_map = {
  0: 0,     
  1: 1,    
  2: 2,      
  3: 3
}

labels_weights = {
  0: 0.01,     
  1: 0.45,    
  2: 0.45,      
  3: 0.09
}


labels_poss = {
  0 : "Ground",
  1 : "AB",
  2: "PM",
  3: "Low Veg"
}
