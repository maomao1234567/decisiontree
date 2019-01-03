import numpy as np

class DecisionTree:
    tree = [0 for i in range(300)]
    attributes = ['a1', 'a2', 'a3', 'b1', 'b2', 'b3', 'c1', 'c2', 'c3', 'class']
    data_list = []

    def __init__(self):
        with open('../data/training_set.csv', 'r') as f:
            lines = f.readlines()
            for i in range(1, len(lines)):
                line = lines[i]
                line = line.replace('\n', '')
                line_list = line.split(',')
                self.data_list.append(line_list)

    def get_each_attribute(self, data_list, attribute_sum):
        data_array = np.array(data_list)
        attributes_dict = {}
        for i in range(attribute_sum):
            attributes_dict[self.attributes[i]] = data_array[:, i]

        return attributes_dict

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

        true_sum_p = true_sum / (true_sum + false_sum)
        false_sum_p = false_sum / (true_sum + false_sum)

        return -1 * true_sum_p * np.log2(true_sum_p) + (-1) * false_sum_p * np.log2(false_sum_p)

    def classification_by_attribute(self, dataset, attribute, value):
        ret_data_set = []
        for featvec in dataset:
            if featvec[attribute] == value:
                reducefeatvec = featvec[:attribute]
                reducefeatvec.extend(featvec[attribute+1:])
                ret_data_set.append(reducefeatvec)

        return ret_data_set

    def choose_attribute(self, attribute_dict):
        min_value = 100
        min_attribute = ''
        for i in range(len(self.attributes) - 1):
            value = self.get_attribute_information_gain(attribute_dict[self.attributes[i]], attribute_dict['class'])
            if value < min_value:
                min_value = value
                min_attribute = self.attributes[i]
        first_node = min_attribute
        attribute_position = self.attributes.index(first_node)
        self.attributes.remove(first_node)
        return attribute_position, first_node

    def get_tree(self):
        # 获取第一层的节点
        data_set = self.data_list
        attribute_dict = self.get_each_attribute(data_set, len(self.attributes))
        attribute_position, first_node = self.choose_attribute(attribute_dict)
        self.tree[1] = first_node

        # 计算第二层的节点
        first_node_by_x_set = self.classification_by_attribute(
            data_set, attribute_position, 'x')
        attribute_dict = self.get_each_attribute(first_node_by_x_set, len(self.attributes))
        attribute_position_1, second_node_1 = self.choose_attribute(attribute_dict)

        first_node_by_b_set = self.classification_by_attribute(
            data_set, attribute_position, 'b')
        attribute_dict = self.get_each_attribute(first_node_by_b_set, len(self.attributes))
        attribute_position_2, second_node_2 = self.choose_attribute(attribute_dict)

        first_node_by_o_set = self.classification_by_attribute(
            data_set, attribute_position, 'o')
        attribute_dict = self.get_each_attribute(first_node_by_o_set, len(self.attributes))
        attribute_position_3, second_node_3 = self.choose_attribute(attribute_dict)
        self.tree[100] = second_node_1
        self.tree[101] = second_node_2
        self.tree[102] = second_node_3

        # 计算第三层的节点
        second_node_1_by_x_set = self.classification_by_attribute(
            first_node_by_x_set, attribute_position_1, 'x')
        attribute_dict = self.get_each_attribute(second_node_1_by_x_set, len(self.attributes))
        attribute_position_4, third_node_1 = self.choose_attribute(attribute_dict)

        second_node_1_by_o_set = self.classification_by_attribute(
            first_node_by_x_set, attribute_position_1, 'o')
        attribute_dict = self.get_each_attribute(second_node_1_by_o_set, len(self.attributes))
        attribute_position_5, third_node_2 = self.choose_attribute(attribute_dict)

        second_node_1_by_b_set = self.classification_by_attribute(
            first_node_by_x_set, attribute_position_1, 'b')
        attribute_dict = self.get_each_attribute(second_node_1_by_b_set, len(self.attributes))
        attribute_position_6, third_node_3 = self.choose_attribute(attribute_dict)

        second_node_2_by_x_set = self.classification_by_attribute(
            first_node_by_b_set, attribute_position_2, 'x')
        attribute_dict = self.get_each_attribute(second_node_2_by_x_set, len(self.attributes))
        attribute_position_7, third_node_4 = self.choose_attribute(attribute_dict)

        second_node_2_by_b_set = self.classification_by_attribute(
            first_node_by_b_set, attribute_position_2, 'b')
        attribute_dict = self.get_each_attribute(second_node_2_by_b_set, len(self.attributes))
        attribute_position_8, third_node_5 = self.choose_attribute(attribute_dict)
        self.tree[200] = third_node_1
        self.tree[201] = third_node_2
        self.tree[202] = third_node_3
        self.tree[203] = third_node_4
        self.tree[204] = third_node_5


de = DecisionTree()
print(de.get_tree())