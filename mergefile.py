import os

if __name__ == '__main__':
    with open('predict/mergefile'+ '.fa', 'w') as f:
        for i in range(6, 11):
            r = open('predict/pred-ann' + str(i) + '.fa')
            lines = r.read()
            f.write(lines)
            f.write('\n')
