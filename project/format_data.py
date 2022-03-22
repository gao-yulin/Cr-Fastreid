import random
import linecache
import os

# select $to_num lines randomly out of $from_num lines from file $filename
def random_lines(filename, from_num, to_num):
    """
    select lines randomly from file

    args:
        filename: str, the file name from which to select lines
        from_num: the number of lines of the file (should be less than the total number of lines)
        to_num: the number of lines selected

    output: list of str [str]
    """
    idxs = random.sample(range(from_num), to_num)
    return [linecache.getline(filename, i) for i in idxs]


def select_lines(filename, num):
    """
    select the first $num lines from file
    """
    idxs = range(num)
    return [linecache.getline(filename, i) for i in idxs]


if __name__ == '__main__':
    path_to_dataset = 'datasets/NAIC2021Reid/'
    train_file = os.path.join(path_to_dataset, 'train_list.txt')
    sub_train_file = os.path.join(path_to_dataset, 'sub_train_list.txt')
    val_gallery_file = os.path.join(path_to_dataset, 'val_gallery_list.txt')
    val_query_file = os.path.join(path_to_dataset, 'val_query_list.txt')
    total = 294000
    num_train = 40000
    num_gallery = 30000
    num_query = 10000
    # generate the sub_train_list file containing the first 200000 lines of data
    lines = random_lines(train_file, total, total)
    with open(sub_train_file, 'w') as f:
        print("start writing to file ", sub_train_file)
        for line in lines:
            f.write(line)
        f.close()
    # generate the val_gallery_list file containing the last 90000 lines of data
    lines = random_lines(train_file, total, num_gallery)
    with open(val_gallery_file, 'w') as f:
        print("start writing to file ", val_gallery_file)
        for line in lines:
            f.write(line)
        f.close()
    # generate the val_query_list file containing part of the gallery file
    lines = select_lines(val_gallery_file, num_query)
    with open(val_query_file, 'w') as f:
        print("start writing to file ", val_query_file)
        for line in lines:
            f.write(line)
        f.close()

