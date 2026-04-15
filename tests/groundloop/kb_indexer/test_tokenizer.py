from groundloop.kb_indexer.tokenizer import tokenize


def test_tokenize_lowercase():
    assert tokenize("Hello WORLD") == ["hello", "world"]


def test_tokenize_splits_on_punctuation():
    assert tokenize("foo-bar,baz.qux") == ["foo", "bar", "baz", "qux"]


def test_tokenize_drops_empty_and_numbers_kept():
    assert tokenize("pytest 3.11") == ["pytest", "3", "11"]


def test_tokenize_empty():
    assert tokenize("") == []


def test_tokenize_only_punctuation():
    assert tokenize("!!!---???") == []


def test_tokenize_unicode():
    assert tokenize("café résumé") == ["café", "résumé"]


def test_tokenize_deterministic():
    a = tokenize("The quick brown fox")
    b = tokenize("The quick brown fox")
    assert a == b
