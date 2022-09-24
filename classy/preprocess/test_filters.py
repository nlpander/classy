import pytest
from classy.preprocess.filters import WebItemsFilter

@pytest.mark.parametrize("in_str,expected_out", [
    ("this should be unchanged", 'this should be unchanged'),
    ("something mundane by Everyman The Ordinary; the end", 'something mundane  the end'),
    ("reach me at please@website.com if you need me", 'reach me at  if you need me'),
    #("reach me at yes.please@real.website.com if you need me", 'reach me at  if you need me'),
    ('please follow me on twitter @elon, thanks!', 'please follow me on twitter, thanks!'),
    ('this is copyright 1996 AmazingInc. All rights reserved. no negotiations', 'this is  no negotiations'),
    #('this is copyright 1996 Amazing Inc. All rights reserved. no negotiations', 'this is  no negotiations'),
    ('<html><body><div some-attr>This is text</div><div>And this is more text</div></body></html>', 'this is textand this is more text')
])
def test_web_items_filter(in_str, expected_out):
    assert WebItemsFilter(in_str) == expected_out
