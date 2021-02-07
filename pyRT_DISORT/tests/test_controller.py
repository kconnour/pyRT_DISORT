import warnings
from unittest import TestCase
import numpy as np
from pyRT_DISORT.controller import ComputationalParameters, ModelBehavior


class TestComputationalParameters(TestCase):
    def setUp(self) -> None:
        self.cp = ComputationalParameters(10, 30, 20, 40, 50, 60)


class TestComputationalParametersInit(TestComputationalParameters):
    def test_str_n_layers_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            ComputationalParameters('foo', 30, 20, 40, 50, 60)

    def test_str_n_moments_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            ComputationalParameters(10, 'foo', 30, 40, 50, 60)

    def test_n_moments_equal_to_n_streams_is_ok(self) -> None:
        ComputationalParameters(10, 30, 30, 40, 50, 60)

    def test_str_n_streams_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            ComputationalParameters(10, 30, 'foo', 40, 50, 60)

    def test_odd_n_streams_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            ComputationalParameters(10, 30, 25, 40, 50, 60)

    def test_str_n_umu_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            ComputationalParameters(10, 30, 20, 'foo', 50, 60)

    def test_str_n_phi_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            ComputationalParameters(10, 30, 20, 40, 'foo', 60)

    def test_str_n_user_levels_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            ComputationalParameters(10, 30, 20, 40, 50, 'foo')


class TestNLayers(TestComputationalParameters):
    def test_n_layers_is_unchanged(self) -> None:
        self.assertEqual(10, self.cp.n_layers)

    def test_n_layers_is_read_only(self) -> None:
        with self.assertRaises(AttributeError):
            self.cp.n_layers = 100


class TestNMoments(TestComputationalParameters):
    def test_n_moments_is_unchanged(self) -> None:
        self.assertEqual(30, self.cp.n_moments)

    def test_n_moments_is_read_only(self) -> None:
        with self.assertRaises(AttributeError):
            self.cp.n_moments = 100


class TestNStreams(TestComputationalParameters):
    def test_n_streams_is_unchanged(self) -> None:
        self.assertEqual(20, self.cp.n_streams)

    def test_n_streams_is_read_only(self) -> None:
        with self.assertRaises(AttributeError):
            self.cp.n_streams = 100


class TestNUmu(TestComputationalParameters):
    def test_n_umu_is_unchanged(self) -> None:
        self.assertEqual(40, self.cp.n_umu)

    def test_n_umu_is_read_only(self) -> None:
        with self.assertRaises(AttributeError):
            self.cp.n_umu = 100


class TestNPhi(TestComputationalParameters):
    def test_n_phi_is_unchanged(self) -> None:
        self.assertEqual(50, self.cp.n_phi)

    def test_n_phi_is_read_only(self) -> None:
        with self.assertRaises(AttributeError):
            self.cp.n_phi = 100


class TestNUserLevels(TestComputationalParameters):
    def test_n_user_levels_is_unchanged(self) -> None:
        self.assertEqual(60, self.cp.n_user_levels)

    def test_n_user_levels_is_read_only(self) -> None:
        with self.assertRaises(AttributeError):
            self.cp.n_user_levels = 100


class TestModelBehavior(TestCase):
    def setUp(self) -> None:
        self.mb = ModelBehavior()


class TestModelBehaviorInit(TestModelBehavior):
    def test_str_accuracy_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            ModelBehavior(accuracy='foo')

    def test_accuracy_outside_range_raises_warning(self) -> None:
        ModelBehavior(accuracy=0)
        with warnings.catch_warnings(record=True) as warning:
            warnings.simplefilter("always")
            ModelBehavior(accuracy=np.nextafter(0, -1))
            self.assertEqual(1, len(warning))
            self.assertEqual(warning[-1].category, UserWarning)

        ModelBehavior(accuracy=0.01)
        with warnings.catch_warnings(record=True) as warning:
            warnings.simplefilter("always")
            ModelBehavior(accuracy=np.nextafter(0.01, 1))
            self.assertEqual(1, len(warning))
            self.assertEqual(warning[-1].category, UserWarning)

    # TODO: deltaM+
    # TODO: do pseudo sphere

    def test_header_outside_range_raises_warning(self) -> None:
        ModelBehavior(header='f'*127)
        with warnings.catch_warnings(record=True) as warning:
            warnings.simplefilter("always")
            ModelBehavior(header='f'*128)
            self.assertEqual(1, len(warning))
            self.assertEqual(warning[-1].category, UserWarning)

    # TODO: ibcnd
    # TODO: onlyfl and afterwards


class TestAccuracy(TestModelBehavior):
    def test_accuracy_is_unchanged(self) -> None:
        mb = ModelBehavior(accuracy=0.001)
        self.assertEqual(0.001, mb.accuracy)

    def test_accuracy_defaults_to_0(self) -> None:
        self.assertEqual(0, ModelBehavior().accuracy)

    def test_accuracy_is_read_only(self) -> None:
        with self.assertRaises(AttributeError):
            self.mb.accuracy = 0


class TestDeltaMPlus(TestModelBehavior):
    def test_delta_m_plus_is_unchanged(self) -> None:
        mb = ModelBehavior(delta_m_plus=True)
        self.assertEqual(True, mb.delta_m_plus)

    def test_delta_m_plus_defaults_to_False(self) -> None:
        self.assertEqual(False, ModelBehavior().delta_m_plus)

    def test_delta_m_plus_is_read_only(self) -> None:
        with self.assertRaises(AttributeError):
            self.mb.delta_m_plus = 0


class TestDoPseudoSphere(TestModelBehavior):
    def test_do_pseudo_sphere_is_unchanged(self) -> None:
        mb = ModelBehavior(do_pseudo_sphere=True)
        self.assertEqual(True, mb.do_pseudo_sphere)

    def test_do_pseudo_sphere_defaults_to_False(self) -> None:
        self.assertEqual(False, ModelBehavior().do_pseudo_sphere)

    def test_do_pseudo_sphere_is_read_only(self) -> None:
        with self.assertRaises(AttributeError):
            self.mb.do_pseudo_sphere = 0


class TestHeader(TestModelBehavior):
    def test_header_is_unchanged(self) -> None:
        mb = ModelBehavior(header='foo')
        self.assertEqual('foo', mb.header)

    def test_header_defaults_to_empty_string(self) -> None:
        self.assertEqual('', ModelBehavior().header)

    def test_header_is_read_only(self) -> None:
        with self.assertRaises(AttributeError):
            self.mb.header = 'foo'


class TestIncidenceBeamConditions(TestModelBehavior):
    def test_incidence_beam_conditions_is_unchanged(self) -> None:
        mb = ModelBehavior(incidence_beam_conditions=True)
        self.assertEqual(True, mb.incidence_beam_conditions)

    def test_incidence_beam_conditions_defaults_to_false(self) -> None:
        self.assertEqual(False, ModelBehavior().incidence_beam_conditions)

    def test_incidence_beam_conditions_is_read_only(self) -> None:
        with self.assertRaises(AttributeError):
            self.mb.incidence_beam_conditions = 0


class TestOnlyFluxes(TestModelBehavior):
    def test_only_fluxes_is_unchanged(self) -> None:
        mb = ModelBehavior(only_fluxes=True)
        self.assertEqual(True, mb.only_fluxes)

    def test_only_fluxes_defaults_to_false(self) -> None:
        self.assertEqual(False, ModelBehavior().only_fluxes)

    def test_only_fluxes_is_read_only(self) -> None:
        with self.assertRaises(AttributeError):
            self.mb.only_fluxes = 0


class TestPrintVariables(TestModelBehavior):
    def test_print_variables_is_unchanged(self) -> None:
        prnt = [True, False, True, False, False]
        mb = ModelBehavior(print_variables=prnt)
        self.assertEqual(prnt, mb.print_variables)

    def test_print_variables_defaults_to_list_of_false(self) -> None:
        prnt = [False, False, False, False, False]
        self.assertEqual(prnt, ModelBehavior().print_variables)

    def test_print_variables_is_read_only(self) -> None:
        with self.assertRaises(AttributeError):
            self.mb.print_variables = 0


class TestRadius(TestModelBehavior):
    def test_radius_is_unchanged(self) -> None:
        mb = ModelBehavior(radius=3400)
        self.assertEqual(3400, mb.radius)

    def test_radius_defaults_to_earth_radius(self) -> None:
        self.assertEqual(6371, ModelBehavior().radius)

    def test_radius_is_read_only(self) -> None:
        with self.assertRaises(AttributeError):
            self.mb.radius = 1000


class TestUserAngles(TestModelBehavior):
    def test_user_angles_is_unchanged(self) -> None:
        mb = ModelBehavior(user_angles=False)
        self.assertEqual(False, mb.user_angles)

    def test_user_angles_defaults_to_true(self) -> None:
        self.assertEqual(True, ModelBehavior().user_angles)

    def test_user_angles_is_read_only(self) -> None:
        with self.assertRaises(AttributeError):
            self.mb.user_angles = True
