from pyrt.spectral import wavenumber


class TestWavenumber:
    def test_1micron_gives_expected_result(self):
        assert wavenumber(1) == 10000

    def test_10microns_gives_expected_result(self):
        assert wavenumber(10) == 1000

    def test_list_input_raises_no_error(self):
        wavenumber([1])
