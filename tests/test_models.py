def test_build_simple_cnn():
    from models.custom_cnn import build_simple_cnn
    model = build_simple_cnn()
    # Check model has a summary method
    assert hasattr(model, 'summary')
