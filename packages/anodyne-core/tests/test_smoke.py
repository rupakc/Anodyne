import anodyne_core


def test_package_importable() -> None:
    assert anodyne_core.__version__ == "0.1.0"
