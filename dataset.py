import numpy as np
import torch as torch
import matplotlib.pyplot as plt
import pdb 

class DataPoints2D(torch.utils.data.Dataset):
    def __init__(self, num_points, data_range):
        self.num_points = num_points
        self.data_range = data_range

        self.data_points, self.labels = self.sample_uniform_data(self.num_points, self.data_range)
    
    def __len__(self):
        return self.data_points.shape[0]

    def __getitem__(self, index):
        return torch.FloatTensor(self.data_points[index]), self.labels[index]

    def sample_uniform_data(self, num_points, data_range):
        x1x2_min = [data_range[0], data_range[0]]
        x1x2_max = [data_range[1], data_range[1]]
        data_points = np.random.uniform(low=x1x2_min, high=x1x2_max, size=(num_points, 2))

        labels = np.zeros(data_points.shape[0])

        # Label data_points following the rules: 0 if outside a circle, 1 if inside a circle with radius r = 5
        circle_center = [(x1x2_min[0]+x1x2_max[0])/2, (x1x2_min[1]+x1x2_max[1])/2]
        radius = abs(x1x2_min[1] - x1x2_max[0])/2

        for idx in range(0, num_points):
            distance_to_center = ((data_points[idx][0]-circle_center[0])**2 + (data_points[idx][1]-circle_center[1])**2)**(1/2)
            
            if distance_to_center < radius:
                labels[idx] = 1
            else:
                continue
        
        return data_points, labels
    
    def visualize_data_points(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        colors = []

        for i, label in enumerate(self.labels):
            if label == 1:
                colors.append("black")
            else:
                colors.append("pink")
        
        plt.scatter(self.data_points[:, 0], self.data_points[:, 1], c=colors)
        plt.savefig("visualized_data_points.png")

if __name__ == "__main__":
    dataset = DataPoints2D(10000, [-5,5])
    dataset.visualize_data_points()

    