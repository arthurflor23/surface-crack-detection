import numpy as np

def random_split_dataset(images, labels, percent):
    v_images, v_labels = [], []
    t_images, t_labels = list(images), list(labels)
    validation_size = round_down(percent*len(images))

    while (len(v_images) < validation_size):
        index = int(np.random.randint(len(t_images), size=1))
        v_images.append(t_images.pop(index))
        v_labels.append(t_labels.pop(index))

    return t_images, t_labels, v_images, v_labels

def shuffle(arr1, arr2):
    index_shuffle = np.arange(0, len(arr1), 1)
    np.random.shuffle(index_shuffle)

    arr1 = np.array(arr1)[index_shuffle]
    arr2 = np.array(arr2)[index_shuffle]

    return list(arr1), list(arr2)

def epochs_and_steps(len_data, len_validation=None):
    if (len_validation == 0):
        g_divisor = int(len_data * 0.1)
    else:
        g_divisor = middle_cdr(len_data, len_validation)

    epochs = len_data//g_divisor
    steps_per_epoch = len_data//epochs
    validation_steps = len_validation//epochs

    return epochs, steps_per_epoch, validation_steps

def round_up(x, digit=10):
    return int(x) if (x % digit == 0) else int((x + digit) - (x % digit))

def round_down(x):
    x = int(x)
    return round(x, 1-len(str(x)))

def middle_cdr(a, b):
    divisors_a = divisors(a)
    divisors_b = divisors(b)
    l = [(i, j) for i in divisors_a for j in divisors_b if (a//i == b//j)]
    return l[int(len(l)*0.75)][0]

def divisors(n):
    divs = [1]
    for i in range(2,int(np.sqrt(n))+1):
        if n%i == 0:
            divs.extend([i,n//i])
    divs.extend([n])
    return sorted(list(set(divs)))

def str_center(*arr):
    stringfy = lambda arr: [str(x) for x in arr]
    max_length = lambda arr: len(max(arr, key=len))
    padding = lambda arr, pad: [x.center(pad) for x in arr]

    arr = stringfy(arr)
    length = max_length(arr)
    arr = padding(arr, length)

    return tuple(arr)