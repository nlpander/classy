import re

WEB_REGEX = re.compile('(' + '|'.join([
    'by .*;',
    '\w+@\w+.com', # email address
    '\s@\w+', # twitter style handle
    'copyright \d+ \w+. all rights reserved.',
    '<[^>]*>', # html tag
]) + ')', re.IGNORECASE)

def WebItemsFilter(text):
    return WEB_REGEX.sub(' ', text)




def NumericalExpressionFilter(word_list):
    # This also removes words that contain *any number*
    # return [w for w in word_list if not any(c.isdigit() for c in w)]
    for i in range(0, len(word_list)):

        # current word
        word = word_list[i]

        # currency
        currency = re.findall('(gbp(\d+))|(usd(\d+))|(eur(\d+))', word)

        # time match
        time = re.findall('(\d+:\d+)', word)

        # time period
        period = re.findall('(\d+-day)|(\d+-week)|(\d+-month)|(\d+-year)', word)

        # listed notation
        listed = re.findall('\d+-listed', word)

        # percentage/points
        perc = re.findall('\d+[p]', word)

        # xl then a number
        mult = re.findall('[xl]\d+', word)

        # thousands
        thou = re.findall('\d+,\d+', word)

        # dec
        dec = re.findall('\d+.\d+', word)

        if word.isdigit() or len(thou) != 0 or len(dec) != 0:
            word_list[i] = '#NUM'

        if len(currency) != 0:
            word_list[i] = '#CURRENCY'

        if len(perc) != 0:
            word_list[i] = '#PERC'

        if len(mult) != 0:
            word_list[i] = '#MOD'

        if len(time) != 0:
            word_list[i] = '#TIME'

        if len(period) != 0:
            word_list[i] = '#TIMEPERIOD'

        if len(listed) != 0:
            word_list[i] = '#LISTING'

    return word_list

