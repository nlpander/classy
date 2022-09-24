import re

def WebItemsFilter(text):
    text = text.lower()

    no_auth = 0
    no_em = 0
    no_tw = 0
    no_copy = 0
    no_htmltag = 0

    try:
        auth = re.findall('(by .*;)', text)
        text = text.replace(auth[0], '')
    except:
        no_auth = 1

    try:
        em = re.findall('(\w+@\w+.com)', text)
        text = text.replace(em[0], '')
    except:
        no_em = 1

    try:
        tw = re.findall('(\s@\w+)', text)
        text = text.replace(tw[0], '')
    except:
        no_tw = 1

    try:
        copyright = re.findall('copyright \d+ \w+. all rights reserved.', text)
        text = text.replace(copyright[0], '')
    except:
        no_copy = 1

    # removing html tags
    try:
        tags = re.findall('<[^>]*>', text)
        for tag in tags:
            text = text.replace(tag, '')
    except:
        no_htmltag = 1

    return text


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

