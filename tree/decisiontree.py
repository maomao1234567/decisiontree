import numpy as np
import pandas as pd


class DecisionTree:
    tree = [0 for i in range(900)]
    attributes = ['a1', 'a2', 'a3', 'b1', 'b2', 'b3', 'c1', 'c2', 'c3', 'class']
    data_list = []
    attributes_dict = {}

    def __init__(self):
        with open('../data/training_set.csv', 'r') as f:
            lines = f.readlines()
            for i in range(1, len(lines)):
                line = lines[i]
                line = line.replace('\n', '')
                line_list = line.split(',')
                self.data_list.append(line_list)

    def get_each_attribute(self):
        data_array = np.array(self.data_list)
        for i in range(10):
            self.attributes_dict[self.attributes[i]] = data_array[:, i]

    def get_attribute_information_gain(self, attribute_array, class_array):
        attribute_martix = np.vstack((attribute_array, class_array))
        data_length = len(attribute_martix[0])
        x_sum = attribute_martix[0].tolist().count('x')
        o_sum = attribute_martix[0].tolist().count('o')
        b_sum = attribute_martix[0].tolist().count('b')
        x_true_sum = 0
        x_false_sum = 0
        b_true_sum = 0
        b_false_sum = 0
        o_true_sum = 0
        o_false_sum = 0

        for i in range(data_length):
            if attribute_martix[0][i] == 'x':
                if attribute_martix[1][i] == 'true':
                    x_true_sum = x_true_sum + 1
                else:
                    x_false_sum = x_false_sum + 1
            elif attribute_martix[0][i] == 'b':
                if attribute_martix[1][i] == 'true':
                    b_true_sum = b_true_sum + 1
                else:
                    b_false_sum = b_false_sum + 1
            elif attribute_martix[0][i] == 'o':
                if attribute_martix[1][i] == 'true':
                    o_true_sum = o_true_sum + 1
                else:
                    o_false_sum = o_false_sum + 1

        x_entropy = self.get_entropy(x_true_sum, x_false_sum)
        b_entropy = self.get_entropy(b_true_sum, b_false_sum)
        o_entropy = self.get_entropy(o_true_sum, o_false_sum)

        return x_entropy * (x_sum/data_length) + b_entropy * (b_sum/data_length) + o_entropy * (o_sum/data_length)

    def get_entropy(self, true_sum, false_sum):
        if true_sum == 0 or false_sum == 0:
            return 0

        true_sum_p = true_sum / true_sum + false_sum
        false_sum_p = false_sum / true_sum + false_sum

        return -1 * true_sum_p * np.log2(true_sum_p) + (-1) * false_sum_p * np.log2(false_sum_p)

de = DecisionTree()
de.get_each_attribute()
print(de.get_attribute_information_gain(de.attributes_dict['a1'], de.attributes_dict['class']))
