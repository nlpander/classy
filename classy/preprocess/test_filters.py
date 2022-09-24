import pytest
from classy.preprocess.filters import WebItemsFilter, CONTAINS_NUM_REGEX
import re

def collapse_whitespace(text):
    # consecutive whitespace chars collapsed to a single space
    return re.sub('(\s{2,})', ' ', text)
@pytest.mark.parametrize("in_str,expected_out", [
    ("this should be unchanged", 'this should be unchanged'),
    ("something mundane by Everyman The Ordinary; the end", 'something mundane the end'),
    ("reach me at please@website.com if you need me", 'reach me at if you need me'),
    #("reach me at yes.please@real.website.com if you need me", 'reach me at  if you need me'),
    ('please follow me on twitter @elon, thanks!', 'please follow me on twitter , thanks!'),
    ('this is copyright 1996 AmazingInc. All rights reserved. no negotiations', 'this is no negotiations'),
    #('this is copyright 1996 Amazing Inc. All rights reserved. no negotiations', 'this is  no negotiations'),
    ('<html><body><div some-attr>This is text</div><div>And this is more text</div></body></html>', ' This is text And this is more text '),

    # test multiple occurances
    ('something mundane by Everyman The Ordinary; reach me at please@website.com if you need me, no really my email is please@website.com ok',
     'something mundane reach me at if you need me, no really my email is ok')
])
def test_web_items_filter(in_str, expected_out):
    assert collapse_whitespace(WebItemsFilter(in_str)) == collapse_whitespace(expected_out)


def test_contains_num_regex():
    assert CONTAINS_NUM_REGEX.sub('#NUM', 'unchanged_thing') == 'unchanged_thing'
    assert CONTAINS_NUM_REGEX.sub('#NUM', 'change23_thing') == '#NUM'
    assert CONTAINS_NUM_REGEX.sub('#NUM', '23,232.123') == '#NUM'
    assert CONTAINS_NUM_REGEX.sub('#NUM', 'change23_thing') == '#NUM'
    assert CONTAINS_NUM_REGEX.sub('#NUM', 'change.thing') == 'change.thing'


