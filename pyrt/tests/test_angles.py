from pyrt.angles import azimuth


def test_azimuth():
    def test_function_gives_expected_results():
        assert azimuth(0, 0, 0) == 0

    test_function_gives_expected_results()
