import argparse
import subprocess
import numpy as np
import time
import random

PYTHON_PATH = "python"

NUM_BASIC_TEST = 2
TEST_NAME = ["b"+str(i) for i in range(1, NUM_BASIC_TEST + 1)]

NUM_TESTS = 6
TEST_NAME = TEST_NAME + ["q"+str(i) for i in range(1, NUM_TESTS + 1)]
parser = argparse.ArgumentParser()
parser.add_argument("--q", choices=TEST_NAME+["all"], default="all")
args = parser.parse_args()

import exercise_0 as ne
    
    
def grad_b1():
    test_lists = [[1, 2, 3, 4, 5], [1, 2, 3, 4], [], [5], [1, 2]]
    answers = [9, 4, 0, 5, 1]
    res = 0
    for arr, ans in zip(test_lists, answers):
        try:
            usr = ne.b1(arr)
            assert usr == ans, 'value error'
            res += 100
            
        except Exception as e:
            print(f"b1({arr}) failed")
            print(e)
            res += 0
    return res

def grad_b2():
    test_name_score_dict = [
        {
            "Alice": 90,
            "Bob": 80,
            "Cindy": 70,
        },
        {
            "Wendy": 90,
            "Zhong Li": 5,
            "Raiden Shogun": 80,
            "Nahida": 85,
            "Furina": 50,
        },
        {
            "Peking U": 100,
            "Tsinghua U": 49,
        }
    ]
    answers = [
        ["Alice", "Bob", "Cindy"],
        ["Nahida", "Raiden Shogun", "Wendy"],
        ["Peking U"]
    ]
    
    res = 0
    for arr, ans in zip(test_name_score_dict, answers):
        try:
            usr = ne.b2(arr)
            assert usr == ans, 'value error'
            res += 100
            
        except Exception as e:
            print(f"b1({arr}) failed")
            print(e)
            res += 0
    return res

def grad_q1():
    test_shape = [(1,),(2,3),(4,5,6)]
    res = 0
    for shape in test_shape:
        try:
            usr = ne.q1(shape)
            assert usr.shape == shape, 'shape error'
            assert usr.dtype == np.float32, 'type error'
            assert np.all(usr == 0), 'value error'
            res += 100

        except Exception as e:
            print(f"q1({shape}) failed")
            print(e)
            res += 0
    return res

def grad_q2():
    test_n = [1, 5, 10, 30]
    res = 0
    for n in test_n:
        try:
            usr = ne.q2(n)
            assert len(usr.shape) == 2, 'shape error'
            assert usr.shape[0] == n, 'shape error'
            assert usr.shape[1] == n, 'shape error'
            # assert usr.dtype == np.int32, 'type error'

            for i in range(0, n):
                for j in range(0, n):
                    assert usr[i, j] == i+j, 'value error' 
            res += 100
            # for implementation
            
        except Exception as e:
            print(f"q2({n}) failed")
            print(e)
            res += 0
    return res

def grad_q3():
    test_name_score_dict = [
        {
            "Alice": 90,
            "Bob": 80,
            "Cindy": 70,
        },
        {
            "Wendy": 90,
            "Zhong Li": 100,
            "Raiden Shogun": 80,
            "Nahida": 85,
            "Furina": 95,
        },
        {
            "Peking U": 100,
            "Tsinghua U": 99,
        }
    ]
    std_name_score_dict = [
        [
            ("Alice", 90),
            ("Bob", 80),
            ("Cindy", 70),
        ],
        [
            ("Zhong Li", 100),
            ("Furina", 95),
            ("Wendy", 90),
            ("Nahida", 85),
            ("Raiden Shogun", 80),
        ],
        [
            ("Peking U", 100),
            ("Tsinghua U", 99),
        ]
    ]

    test_change_score = [
        [
            ("Alice", 95),
        ],
        [
            ("Furina", 100),
        ],
        [
            ("Peking U", 99),
        ]
        
    ]
    res = 0
    for test, std,change in zip(test_name_score_dict, std_name_score_dict, test_change_score):
        try:
            usr = ne.q3(test)
            assert len(usr) == len(std), 'length error'
            
            for usr_item, std_item in zip(usr, std):
                std_name, std_score = std_item
                assert usr_item.name == std_name, 'name error'
                assert usr_item.score == std_score, 'score error'
            # for name, score in change:
            #     idx  = [i.name for i in usr].index(name)
            #     usr[idx].score = score
            #     assert usr[idx].sqrt_prod_10 == np.sqrt(score) * 10, 'property error'
            res += 100
            
        except Exception as e:
            print(f"q3({test}) failed")
            print(e)
            res += 0
    return res


def grad_q4():
    
    test_shape = [(9,), (2,3), (4,5,6)]
    res = 0

    np.random.seed(20240229)
    for shape in test_shape:
        try:
            a = np.random.randint(-100, 100, size = shape)
            b = np.random.randint(-100, 100, size = shape)
            a_clone = a.copy()
            b_clone = b.copy()
            usr = ne.q4(a, b)

            assert usr.shape == shape, 'shape error'
            assert np.all(a == a_clone), 'input changed'
            assert np.all(b == b_clone), 'input changed'

            a[a < b] = b[a < b]
            assert np.all(a == usr), 'value error'
            res += 100

        except Exception as e:
            print(f"q4({shape}) failed")
            print(e)
            res += 0
    return res


def grad_q5():
    
    res = 0

    try:
        usr = ne.q5()
        std = np.pi
        assert abs(usr - std) < 0.01, 'value error'
        res += 100

    except Exception as e:
        print(f"q5() failed")
        print(e)
        res += 0
    
    return res
    
def grad_q6():
    file_path = 'example.lay'
    
    res = 0
    try:
        obj = ne.q6(file_path)
        out = obj.recover_layoutText()
        with open(file_path, 'r') as f:
            text = f.read()
        for a, b in zip(out.split('\n'), text.split('\n')):
            assert a == b, 'recover_layoutText error, expected: \n' + text + '\nbut got: \n' + out + '\n'
        res = 100
    except Exception as e:
        print(f"q6({file_path}) failed")
        print(e)
        return 0
    return res

if __name__ == "__main__":
    if args.q == "all":
        for q in TEST_NAME:
            result = eval('grad_'+str(q))()
            print(f"test {q}, score {result:.0f}")
    else:
        result = eval('grad_'+str(args.q))()
        print(f"test {args.q}, score {result:.0f}")