import sys
from src import train
from src import eval

def main():
    if sys.argv[1] == 'train':
        train.main(sys.argv[2:])
    elif sys.argv[1] == 'eval':
        eval.main(sys.argv[2:])

if __name__ == '__main__':
    main()