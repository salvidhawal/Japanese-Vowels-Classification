import os


def change_to_main_dir():
    os.chdir('../')
    pwd = os.getcwd()
    print(pwd)
