import random


class TicTacToeData:
    def __init__(self, file_name):
        self.file_name = file_name

    def read_file(self):
        data_list = []
        with open(self.file_name, "r") as f:
            data_lines = f.readlines()
            for line in data_lines:
                decoded_line = line.replace('positive', 'true')
                decoded_line = decoded_line.replace('negative', 'false')
                data_list.append(decoded_line)

        return data_list

    def get_training_test_set(self, data_list):
        with open('training_set.csv', 'w') as f1, open('test_set.csv', 'w') as f2:
            f1.write('a1,a2,a3,b1,b2,b3,c1,c2,c3,class\n')
            f2.write('a1,a2,a3,b1,b2,b3,c1,c2,c3,class\n')
            for data in data_list:
                rand = random.random()
                if rand < 0.75:
                    f1.write(data)
                else:
                    f2.write(data)
