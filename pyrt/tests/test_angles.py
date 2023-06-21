from pyrt.angles import azimuth


class TestAzimuth:
    def test_function_gives_expected_results(self):
        assert azimuth(0, 0, 0) == 0

    def test_analytical_case(self):
        assert azimuth(30, 30, 60) == 0
