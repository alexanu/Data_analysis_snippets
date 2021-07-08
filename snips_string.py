
# string series
	Series(list('abcde'))
	random.choices(string.ascii_lowercase,k=5) # generates k random letters
	tm.makeStringIndex()


# Actions with strings
    s.endswith(suffix)     # Check if string ends with suffix
    s.find(t)              # First occurrence of t in s
    s.index(t)             # First occurrence of t in s
    s.join(slist)          # Join a list of strings using s as delimiter
    s.replace(old,new)     # Replace text
    s.rfind(t)             # Search for t from end of string
    s.rindex(t)            # Search for t from end of string
    s.split([delim])       # Split string into list of substrings
    s.startswith(prefix)   # Check if string starts with prefix
    s.strip()              # Strip leading/trailing space


# modify strings by converting them to list
    x ="Hello!"
    x[-2] # like in the list ...
    # x[2]="F" ... but strings cannot be modified
    # x.append("v")
    # => ...
    #    1) turn the string into a list of characters, 
    #    2) do whatever you want, 
    #    3) and then turn the resulting list back into a string
    text = "Hello world!"
    wordList = list(text)
    wordList[7:]=[] # removes everything after 7th char
    wordList.reverse()
    text="".join(wordList) # turn the list back into a string


# delete $ from string
    df.state_bottle_retail.str.replace('$','') # 4.5*X ms: replaces the ‘$’ with a blank space for each item in the column
    df.state_bottle_retail.apply(lambda x: x.replace('$','')) # 4*X ms: pandas ‘apply’ method, which is optimized to perform operations over a pandas column
    df.state_bottle_retail.apply(lambda x: x.strip('$')) # 3*X ms: strip does one less operation: just takes out the ‘$.’
    df.state_bottle_retail = [x.strip('$') for x in df.state_bottle_retail] # 2*X ms: list comprehension
    df.state_bottle_retail = [x[1:] for x in df.state_bottle_retail] # X ms: built in [] slicing, [1:] slices each string from 2nd value till end


# skip strings with //
    string_from_file = """
    // Author: ...
    // License: ...
    //
    // Date: ...

    Actual content...
    """

    import itertools
    for line in itertools.dropwhile(lambda line: line.startswith("//"), string_from_file.split("\n")):
        print(line)


# printing strings
    e=2.718
    x = [1, "two", 3, 4.0, ["a", "b"], (5, 6)]
    print ("The constant e is ", e, " and the list x is ", x) # converts everything to string
    "{} can be {}".format("Strings", "interpolated")  # => "Strings can be interpolated"

    print("the value of %s is: %.2f" % ("e", e)) # formatting capabilities similar to sprintf in c
    num_dict = {'e':2.718,'pi':3.14159}
    print("%(pi).2f - %(pi).4f - %(e).1f" % num_dict)


    for idx, symbol in enumerate(symbols):
        for x,y in zip(list(range(0,600)),list(range(1,601))):
            print('Done with %s for %s.' % (symbol, start))
        print('Done with %s. Still %i to go' % (symbol, len(symbols)-idx-1))




    " ".join(["a","b","c","d"])

    URL_HISTORICAL = 'http://ichart.yahoo.com/table.csv?s=%(s)s&a=%(a)s&b=%(b)s&c=%(c)s&d=%(d)s&e=%(e)s&f=%(f)s'
    url = self.URL_HISTORICAL % dict(
                s=self.symbol,
                a=start.month-1,b=start.day,c=start.year,
                d=stop.month-1,e=stop.day,f=stop.year)

    def promt_time_stamp():
        return str(datetime.datetime.fromtimestamp(time.time()).strftime('[%H:%M:%S] '))
    print(promt_time_stamp() + 'load dowjones tickers ..')

            
# Convert raw string integer inputs to integers
    str_input = "1 2 3 4 5 6"
    int_input = map(int, str_input.split())
    print(list(int_input))


# Split long string into list elements
    String_of_SPDR_ETF = ("ZPRL-GY, SPYF-GY, ZPRD-GY, SPP1-GY, SPYI-GY, SPYY-GY, SPP2-GY, SPYA-GY, ZPRE-GY, SPYX-GY, SPYM-GY, SPYT-GY, SPYR-GY, SPYC-GY, "
                        "SPYN-GY, SPYZ-GY, SPYH-GY, SPYQ-GY, SPYP-GY, SMC-FP^, ZPRX-GY, SPYK-GY, SPYE-GY, SPYU-GY, ZPDW-GY, ZPDJ-GY, ZPRV-GY, WTEL-NA, "
                        "WCOD-NA, WCOS-NA, WNRG-NA, WFIN-NA, WHEA-NA, WIND-NA, WMAT-NA, ZPRS-GY, WTCH-NA, SPPW-GY, WUTI-NA, ZPRR-GY, SPY4-GY, "
                        "SPPY-GY, SPPE-GY, SPY1-GY, SPY5-GY, SPYV-GY, SPYW-GY, ZPRG-GY, ZPRA-GY, ZPDK-GY, ZPDD-GY, ZPDS-GY, SPPD-GY, SPYD-GY, ZPDE-GY, "
                        "ZPDF-GY, ZPDH-GY, ZPDI-GY, ZPDM-GY, ZPDT-GY, ZPDU-GY, SPYG-GY, ZPDX-GY, GLOW-NA")
    SPDR_ETF=[x.lower() for x in String_of_SPDR_ETF.split(", ")]


# Filter out all words with three characters or less
    # Filtering out words that don’t contribute a lot of meaning
    text = '''
            Call me Ishmael. Some years ago - never mind how long precisely - having
            little or no money in my purse, and nothing particular to interest me
            on shore, I thought I would sail about a little and see the watery part
            of the world. It is a way I have of driving off the spleen, and regulating
            the circulation. - Moby Dick'''
    w = [[x for x in line.split() if len(x)>3] for line in text.split('\n')]


# creates dictionary from string
    some_string = "snowboard"
    some_dict = {}
    for i, some_dict[i] in enumerate(some_string):
        pass
    some_dict


# Concatenate long strings elegantly across line breaks in code
    my_long_text = ("We are no longer the knights who say Ni! "
                    "We are now the knights who say ekki-ekki-"
                    "ekki-p'tang-zoom-boing-z'nourrwringmm!")


# clean spaces in strings
		
    import re
    def clean(string): # cleans from empty spaces on start and on end
        # input is : "   Hello World    " 
        # output is "Hello World"
            
        first = 0
        for item in string:
            if item != ' ':
                first = string.index(item)
                break
        string = string[::-1]
        last = 0
        for item in string:
            if item != ' ':
                last = string.index(item)
                break
        return string[::-1][first:len(string)-last]
        
    def clean2(string): # the same purpose as above
        nonempty = [string.index(item) for item in string if item != ' ']
        nonempty2 = [string[::-1].index(item) for item in string[::-1] if item != ' ']
        return string[nonempty[0]:len(string) -nonempty2[0]]

    def clean3(string): # clean string from rebundant spaces
        try:
            for item in re.findall('[" "]{2,}',string):
                string = string.replace(item, ' ')
                if string[0] == ' ':
                    string = string[1:]
                if string[-1] == ' ':
                    string = string[:-1]
            return string
        except:
    return string
		

    # getting rid of extra white spaces
    import string
    string.whitespace # white spaces are not only spaces, but e.g. new line, tab, etc.

    x="\n tha basketball playear allen iverson \t   "
    x
    x.strip() # any whitespaces in the beginning and end are removed
    x.rstrip() # rstrip removes whitespace only at the right end

    # The common use for these functions is to clean up strings...
    # ... that have just been read in: e.g. reading lines from files ...
    # ... as Python reads in an entire line, incl. the trailing newline, 
    # "rstrip" is a convenient way to get rid of it.


# make translation of strings
    x = "~x ^ (y % z)" 
    table = x.maketrans("~^()","!&[]") # from package "string"
    x.translate(table)


# Convert "1k" to 1 000, "1m" to 1 000 000, etc.
    def resolve_value(value):
        if value is None:
            return None
        tens = dict(k=10e3, m=10e6, b=10e9, t=10e12)
        value = value.replace(',', '')
        match = re.match(r'(-?\d+\.?\d*)([kmbt]?)$', value, re.I)
        if not match:
            return None
        factor, exp = match.groups()
        if not exp:
            return float(factor)
        return int(float(factor)*tens[exp.lower()])


# fuzzy matching with Levenshtein distance
    import difflib
    difflib.get_close_matches('appel', ['ape', 'apple', 'peach', 'puppy'], n=2) # n specifies max num of matches to be returned
    # returns ['apple', 'ape']


# the most often occuring names using collection.Counter
    from collections import Counter

    cheese = ["gouda", "brie", "feta", "cream cheese", "feta", "cheddar",
            "parmesan", "parmesan", "cheddar", "mozzarella", "cheddar", "gouda",
            "parmesan", "camembert", "emmental", "camembert", "parmesan"]

    cheese_count = Counter(cheese) # Counter is just a dictionary that maps items to number of occurrences
    # use update(more_words) method to easily add more elements to counter

    print(cheese_count.most_common(3))
    # Prints: [('parmesan', 4), ('cheddar', 3), ('gouda', 2)]


# Get vowels from a sentence
    sentence = 'the rocket came back from mars'
    vowels = [i for i in sentence if i in 'aeiou'] # list
    unique_vowels = {i for i in sentence if i in 'aeiou'} # set


# filter out companies which name ends on LLC
	qvdf=qvdf[~qvdf['name'].str[-2:].str.contains('LP')]
	qvdf=qvdf[~qvdf['name'].str[-3:].str.contains('LLC')]


# Create rows for values separated by commas in a cell

    d = {"Team":["FC Barcelona", "FC Real Madrid"], 
        "Players":["Ter Stegen, Semedo, Piqué, Lenglet, Alba, Rakitic, De Jong, Sergi Roberto, Messi, Suárez, Griezmann",
                "Courtois, Carvajal, Varane, Sergio Ramos, Mendy, Kroos, Valverde, Casemiro, Isco, Benzema, Bale"]}
    df = pd.DataFrame(d)
    df.assign(Players = df["Players"].str.split(",")).explode("Players")


# All possible combinations of letters
    from itertools import product
    for i in product(['A', 'C', 'G'], repeat=10):
        print(''.join(i))

