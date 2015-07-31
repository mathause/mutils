import unittest
import numpy as np
from mutils.water_vapor import *

#freezing point
fp = 273.15


class wv_round_trip(unittest.TestCase):
    # def setUp(self):
    #     pass

    # def tearDown(self):
    #     pass

    def runTest(self):

        # integer
        T = 273.15 + 20
        e = 611.
        p = 1.0132 * 10**5

        assert np.allclose(rho_v_2_e(e_2_rho_v(e, T), T), e)
        assert np.allclose(q_v_2_e(e_2_q_v(e, p), p), e)
        assert np.allclose(w_v_2_e(e_2_w_v(e, p), p), e)
        assert np.allclose(rel_hum_2_e(e_2_rel_hum(e, T, fp), T, fp), e)

        # numpy array
        T = np.linspace(-10, 30) + 273.15
        e = np.linspace(125, 4000)
        p = np.linspace(0.85, 1.1) * 10**5

        assert np.allclose(rho_v_2_e(e_2_rho_v(e, T), T), e)
        assert np.allclose(q_v_2_e(e_2_q_v(e, p), p), e)
        assert np.allclose(w_v_2_e(e_2_w_v(e, p), p), e)
        assert np.allclose(rel_hum_2_e(e_2_rel_hum(e, T, fp), T, fp), e)

class wv_RH(unittest.TestCase):
    def runTest(self):

        T = np.linspace(-10, 30) + 273.15
        
        es = saturation_vapor_pressure(T, freezing_point=273.15)

        res = np.ones_like(T)

        assert np.allclose(e_2_rel_hum(es, T, fp), res)

        res = np.zeros_like(T)
        assert np.allclose(e_2_rel_hum(res, T, fp), res)

class wv_saturation_vapor_pressure(unittest.TestCase):
    def runTest(self):

        # error on T < 0
        self.assertRaises(ValueError, saturation_vapor_pressure, -5)

        T = np.arange(-70, 50, 10) + 273.15

        # values from the script
        res = np.asarray([0.48, 1.87, 6.32, 18.9, 51.0, 125.6, 286.7,  611.7,
                            1228.1, 2339.4, 4246.8, 7384.3])

        e_s_w = saturation_vapor_pressure(T, freezing_point=273.15 - 80)
        e_s_w = np.round(e_s_w, 2)

        assert np.allclose(e_s_w, res, atol=0.5)


        T = np.arange(-70, 10, 10) + 273.15
        res = np.asarray([0.26, 1.08, 3.94, 12.9, 38.1, 103.4, 260.1, 611.7])       
       
        e_s_i = np.round(saturation_vapor_pressure(T), 2)









if __name__ == '__main__':
    unittest.main()










