def test_predict_placeholder():
    from inference.predictor import predict
    res = predict(None, None)
    assert 'class' in res and 'confidence' in res
