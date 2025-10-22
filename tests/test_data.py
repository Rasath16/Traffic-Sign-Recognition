def test_list_images_empty(tmp_path):
    from data.data_loader import list_image_files
    files = list_image_files(tmp_path)
    assert isinstance(files, list)
