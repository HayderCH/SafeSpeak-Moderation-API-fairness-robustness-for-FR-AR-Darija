from safespeak.preprocessing.normalize import batch_normalize, normalize_text


def test_normalize_text_basic():
    text = "T'es un idiot 😂!!!"
    normalized = normalize_text(text)
    assert "idiot" in normalized
    assert "laugh" in normalized
    assert "!!!" not in normalized


def test_batch_normalize_returns_list():
    texts = ["Salut", "مرحبا"]
    results = batch_normalize(texts)
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(item, str) for item in results)
