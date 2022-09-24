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


# matches on an entire string that contains a number
CONTAINS_NUM_REGEX = re.compile('^.*\d+([,\.]\d+)*.*$')

def NumericalExpressionFilter(word_list):
    # This also removes words that contain *any number*
    # return [w for w in word_list if not any(c.isdigit() for c in w)]
    for i, word in enumerate(word_list):

        word = CONTAINS_NUM_REGEX.sub('#NUM', word)

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

