"""

"""

def fun1():
    print('fun1')
    ...

def fun2():
    print('fun2')
    ...

...

def fun400():
    print('fun400')
    ...

if __name__=="__main__":
    i=400
    match i:
        case 1: fun1()
        case 2: fun2()
        case 400: fun400()
        # что делать в случае не определенном явно
        case _ : print('Функция не определена')

