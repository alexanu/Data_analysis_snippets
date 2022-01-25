
# Help on functions / object

    help(len)

    len? # help on an object

    L = [1, 2, 3]
    L?

    square?? # reading the source code

    L.<TAB> # see a list of all available attributes of an object
    L.c<TAB>
    from itertools import co<TAB>

    str.*find*? # looking for a string method that contains the word find somewhere


print('\U0001F600')
print('\N{deciduous tree}')


# Importing / create packages 

    help('modules') # check which modules are installed
    import <TAB> # see which imports are available

    import sys
    sys.path
    # When you ask Python to import a module, ...
    # ... it starts with the first directory in sys.path ...
    # ... and checks for an appropriate file. 
    # If no match is found in the first directory it checks subsequent entries, in order, until a match is found 

    # To create a normal module ...
    # ... you simply create a Python source file in a directory contained in sys.path. 
    # The process for creating packages is not much different. 

    # CREATION OF PACKAGE: Python modules are just ordinary Python files. 

        import sys
        sys.path
        # When you ask Python to import a module, ...
        # ... it starts with the first directory in sys.path ...
        # ... and checks for an appropriate file. 
        # If no match is found in the first directory it checks subsequent entries, in order, until a match is found 

        # To create a normal module ...
        # ... you simply createa Python source file in a directory contained in sys.path. 
        # The process for creating packages is not much different. 


        # You can write your own, and import them. The name of the module is the same as the name of the file.

        '''File wo.py'''

        def word_occur():           # new function, which counts occurrences of words in a file
            file_name = input("Enter the name of the file:") # Prompt user for the name of the file to use
            f = open(file_name, "r")
            word_list = f.read().split() # store words from the file in a list
            f.close()
            
            occurs_dict={}
            for word in word_list:
                occurs_dict[word] = occurs_dict.get(word,0) + 1 # increment the occurrences count for this word 
            print("File %s has %d words (%d are unique)", % (file_name, len(word_list), len(occurs_dict)))
            print(occur_dict)
            
        if __name__=='__main__': # this allows the program to be run as a script by typing "python wo.py" at a command line
            word_occur
            
        '''end of the file wo.py'''    
            
        # If you place a file in one of the directories on the module search path, ...
        # ... which can be found in sys.path, ...
        # ... it can be imported like any of the built-in library modules by using the import statement:

        import wo
        wo.word_occur()

        # Note that if you change the file wo.py on disk, ...
        # ... import won’t bring your changes into the same interactive session. 
        # You use the reload function from the imp library in this situation:

        import imp
        imp.reload(wo)

        import re
        dir(re) # which functions and attributes are defined in a module.

        # There are also anonymous functions
        (lambda x: x > 2)(3)                  # => True
        (lambda x, y: x ** 2 + y ** 2)(2, 1)  # => 5


        # To create a package, ...
        # 1) create the package’s root directory. 
        #        This root directory needs to be in some directory on sys.path
        #        remember, this is how Python finds modules and packages for importing. 
        # 2) In that root directory, you create a file called __init__.py. 
        #        This file — which we’ll often call the package init file — is what makes the package a module. 
        #        __init__.py can be(and often is) empty;
        #        its presence alone suffices to establish the package.

        # In Shell:
        # mkdir Anaconda3/reader
        # type Anaconda3/reader/__init__.py

        import reader
        type(reader) # => module, even though on our filesystem the name “reader” refers to a directory
        reader.__file__


# Memory usage

    import pandas as pd
    import sys
    colors = pd.Series(['periwinkle','mint green','burnt orange','periwinkle',
                        'burnt orange','rose','rose','mint green','rose','navy'])
    colors.apply(sys.getsizeof) # show the memory occupied by each individual value
    # Keep in mind these are Python objects that have some overhead in the first place. 
    # ... (sys.getsizeof('') will return 49 bytes.)

    colors.memory_usage() # sums up the memory usage
                        # relies on the .nbytes attribute of the underlying NumPy array


    sys.getsizeof(colors)


# Pointers

    # In Python the variables are just labels (kind of pointers), not containers (as in C)
    a = [1,2,3]
    b = a
    c = b
    a[1] = 5 # here the 2nd elmnt of "b" and "c" will also change. It wouldn't be the case if variable were containers
    print(a,b,c)

    # This Python feature causes that one variable could be assigned to different types
    x = "Hello"
    print(x)
    x = 5
    print(x)


    # (is vs. ==) is checks if two variables refer to the same object, 
    # but == checks if the objects pointed to have the same values.
    a = [1, 2, 3, 4]  # Point a at a new list, [1, 2, 3, 4]
    b = a             # Point b at what a is pointing to
    b is a            # => True, a and b refer to the same object
    b == a            # => True, a's and b's objects are equal
    b = [1, 2, 3, 4]  # Point b at a new list, [1, 2, 3, 4]
    b is a            # => False, a and b do not refer to the same object
    b == a            # => True, a's and b's objects are equal


    id(256)
    a=256
    b=256
    id(a), id(b), id(256)



    row = [""]*3
    board = [row]*3 # each of the elements board[0], board[1] and board[2] ...
                    # ...is a reference to the same list referred by "row"
    board[0][0]= "X"
    board # ==> [['X', '', ''], ['X', '', ''], ['X', '', '']]
    # to avoid this:
    board_new = [[""]*3 for _ in range(3)] 
    board_new[0][0] = "Y"
    board_new   # ==> [['Y', '', ''], ['', '', ''], ['', '', '']]



# lists

    # Use a list when the order of the data matters. Remember that lists can hold any kind of object.
        portfolio = [
            ('GOOG', 100, 490.1),
            ('IBM', 50, 91.3),
            ('CAT', 150, 83.44)
        ]

        portfolio[0]            # ('GOOG', 100, 490.1)
        portfolio[2]            # ('CAT', 150, 83.44)

    # lists could contain different types of elements: strings, other lists, dictionaries, functions
        []
        [1]
        [1,2,3,4,5,6,7,8]
        [1,"two",34.234,{"a","b"},(5,6)]


    line = 'GOOG,100,490.10'
    row = line.split(',') # row: ['GOOG', '100', '490.10']


    # list initialization: common one for working with large lists 
    # whose size is known ahead of time
    k=[None]*5
    k


    x = [12,23,34,13,24,35,46,57,68]
    x[0], x[0:2], x[-1] # last element
    x[1:-1]
    x[-3:] # last 3 elements

    # z in x
        46 in x
        z = 13
        z in x

        # "if m in [1,2,3,4]:" is the same as "if m==1 or m==2 or m==3 or m==4:"

    # Find the most frequent value in a list.
    test = [1,2,3,2,4,5,6,5,5,7,8,8,7,5,4,5,3,4]
    max(test,key=test.count)



    mixed_list = [False, 1.0, "some_string", 3, True, [],False]
    integers_found_so_far=0
    booleans_found_so_far=0
    for i in mixed_list:
        if isinstance(i,int):
            integers_found_so_far += 1
        if isinstance(i, bool):
            booleans_found_so_far += 1
    integers_found_so_far # Booleans are a subclass of int



    # Adding/replace
        x[3:5]=["one","two","three"]
        [111,222,333]+x
        x[len(x):]=[222,333,444]
        x.append(1111)

        first = [1,2,3,4]
        second = [5,6,7]
        third = [8,9,10]
        first.append(second)
        first.extend(third)
        first
        second + third

        x[8:9]=[]
        x.pop() # Remove from the end
        del x[2]
        x.remove("three")

        x.reverse() # this operation changes the initial list


# Dictionaries & Sets
    # Python´s hash tables
    # Dictionaries are useful if you want fast random lookups (by key name). 

    y = {}
    y[0] = "Hallo"
    y[1] = "Goodbye"

    y["two"] = 2 # dictionary keys may be numbers, strings, etc (for lists - only integers)
    y["pi"] = 3.14
    y["two"]*y["pi"]

    x = (1,2,3,4,5)
    y = ("one","two","three","four","five")
    dict(zip(x,y))

    some_string = "snowboard"
    some_dict = {}
    for i, some_dict[i] in enumerate(some_string):
        pass
    some_dict




    dictionary1 = {"name": "Joy", "age": 25}
    dictionary2 = {"name": "Joy", "city": "New York"}
    merged_dict = {**dictionary1, **dictionary2}
    print("Merged dictionary:", merged_dict)



    prices = {
    'GOOG': 513.25,
    'CAT': 87.22,
    'IBM': 93.37,
    'MSFT': 44.12
    }
    prices['IBM'] # 93.37

    # Composite keys
    holidays = {
    (1, 1) : 'New Years',
    (3, 14) : 'Pi day',
    (9, 13) : "Programmer's day",
    }
    holidays[3, 14] # 'Pi day'

    # Sets are collection of unordered unique items
    # Sets are useful for membership tests
    # Sets are also useful for duplicate elimination
        names = ['IBM', 'AAPL', 'GOOG', 'IBM', 'GOOG', 'YHOO']
        unique = set(names) # unique = set(['IBM', 'AAPL','GOOG','YHOO'])


# LOOPS
    item_list = [3, "string1", 23, 14.0, "string2", 49, 64,70]
    for x in item_list: # more of a foreach loop
        if not isinstance(x,int):
            continue # If x isn’t an integer, the rest of this iteration is aborted by the continue statement
        if not x%7: # finds the first occurrence of an integer that’s divisible by 7
            print("Found an integer divisible by seven: %d" % x)
            break
            

    for animal in ["dog", "cat", "mouse"]:
        print("{} is a mammal".format(animal))

    names = ['Elwood', 'Jake', 'Curtis']
    for i, name in enumerate(names):
        # Loops with i = 0, name = 'Elwood'
        # i = 1, name = 'Jake'
        # i = 2, name = 'Curtis'
        

    for idx, symbol in enumerate(symbols):
        try:
            prof=fmpsdk.company_profile(apikey=fmp_key, symbol=symbol)
            data = []
            for x,y in zip(list(range(0,600)),list(range(1,601))):
                url = f"{BASE_URL_v4}/{symbol}/{time_delta}/minute/{start}/{end}?apikey={fmp_key}"
                try:
                    response = requests.get(url)
                    data.append(response.json()['results'])
                except:
                    break
            flat_data = [item for sublist in data for item in sublist] # flatten the list
            print('Done with %s. Still %i to go' % (symbol, len(symbols)-idx-1))
        except:
            pass




    for name in namelist:
        if name == 'Jake':
            break
        ...
        # statements


    # If you need to count, use range()
    for i in range(100):
        # i = 0,1,...,99
    for j in range(10,20):
        # j = 10,11,..., 19
    for k in range(10,50,2):
        # k = 10,12,...,48
        # Notice how it counts in steps of 2, not 1.


    points = [  (1, 4),(10, 40),(23, 14),(5, 6),(7, 8) ]
    for x, y in points:
        # Loops with x = 1, y = 4
        #            x = 10, y = 40
        #            x = 23, y = 14


# Try / Except

    for idx, symbol in enumerate(symbols):
        try:
            prof=fmpsdk.company_profile(apikey=fmp_key, symbol=symbol)
            data = []
            for x,y in zip(list(range(0,600)),list(range(1,601))):
                url = f"{BASE_URL_v4}/{symbol}/{time_delta}/minute/{start}/{end}?apikey={fmp_key}"
                try:
                    response = requests.get(url)
                    data.append(response.json()['results'])
                except:
                    break
            flat_data = [item for sublist in data for item in sublist] # flatten the list
            print('Done with %s. Still %i to go' % (symbol, len(symbols)-idx-1))
        except:
            pass


    # Else with Try/Except
        # You can have an 'else' clause with try/except. It gets excecuted if no exception is raised.
        # This allows you to put less happy-path code in the 'try' block ...
        # so you can be more sure of where a caught exception came from.

        try:
            1 + 1
        except TypeError:
            print("Oh no! An exception was raised.")
        else:
            print("Oh good, no exceptions were raised.")


        # else gets called when for loop does not reach break statement
        a = [1, 2, 3, 4, 5]
        for el in a:
            if el == 0:
                break
        else:
            print('did not break out of for loop')







#Push to G Drive

    #https://medium.com/@annissouames99/how-to-upload-files-automatically-to-drive-with-python-ee19bb13dda                                                   
    #pip install PyDrive is needed                                                                                                                           

    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    import os
    g_login = GoogleAuth()
    g_login.LocalWebserverAuth()
    drive = GoogleDrive(g_login)

    with open("results/file_to_be_pushed","r") as file:
        file_drive = drive.CreateFile({'title':os.path.basename(file.name) })
        file_drive.SetContentString(file.read())
        file_drive.Upload()


# Print with delay

    from time import sleep

    def delay(fn, ms, *args):
        sleep(ms / 1000)
        return fn(*args)

    delay(lambda x: print(x),1000,'later') # prints 'later' after one second


#Read tokes 
    import os
    from configparser import ConfigParser

    Path_To_TKNS = os.path.join(os.path.abspath(os.path.join(__file__ ,"../../..")), "connections.cfg")
    config = ConfigParser()
    config.read(Path_To_TKNS)
    token=config['FinnHub']['access_token']


# REad environ variables for storing secrets

    import os
''' type in terminal: set EMAIL_MAIN=oanufriyev@gmail.com'''

    EMAIL_ADDRESS = os.environ.get('EMAIL_MAIN')
    print(os.environ.get('EMAIL_MAIN'))
    print(os.environ)
    os.environ.get('USER')

    XXX = os.getenv('EMAIL_MAIN')

