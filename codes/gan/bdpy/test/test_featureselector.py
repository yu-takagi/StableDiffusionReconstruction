'''Tests for FeatureSelector'''


from unittest import TestCase, TestLoader, TextTestRunner

from bdpy.bdata.featureselector import FeatureSelector


class TestFeatureSelector(TestCase):
    '''Tests for FeatureSelector'''

    def test_lexical_analysis_0001(self):

        test_input = 'A = B'
        exp_output = ('A', '=', 'B')

        selector = FeatureSelector(test_input)

        test_output = selector.token

        self.assertEqual(test_output, exp_output)

    def test_lexical_analysis_0002(self):

        test_input = 'HOGE = 1'
        exp_output = ('HOGE', '=', '1')

        selector = FeatureSelector(test_input)

        test_output = selector.token

        self.assertEqual(test_output, exp_output)

    def test_lexical_analysis_0003(self):

        test_input = '(HOGE = 1) & (FUGA = 0)'
        exp_output = ('(', 'HOGE', '=', '1', ')',
                      '&', '(', 'FUGA', '=', '0', ')')

        selector = FeatureSelector(test_input)

        test_output = selector.token

        self.assertEqual(test_output, exp_output)

    # def test_lexical_analysis_0004(self):

    #     test_input = 'HOGE top 100'
    #     exp_output = ('HOGE', 'top', '100')

    #     selector = FeatureSelector(test_input)

    #     test_output = selector.token

    #     self.assertEqual(test_output, exp_output)

    def test_parse_0001(self):

        test_input = 'A = B'
        exp_output = ('A', 'B', '=')

        selector = FeatureSelector(test_input)

        test_output = selector.rpn

        self.assertEqual(test_output, exp_output)

    def test_parse_0002(self):

        test_input = 'A = 1 | B = 0'
        exp_output = ('A', '1', '=', 'B', '0', '=', '|')

        selector = FeatureSelector(test_input)

        test_output = selector.rpn

        self.assertEqual(test_output, exp_output)

    # def test_parse_0003(self):

    #     test_input = 'HOGE top 100 @ A = 1'
    #     exp_output = ('HOGE', '100', 'top', 'A', '1', '=', '@')

    #     selector = FeatureSelector(test_input)

    #     test_output = selector.rpn

    #     self.assertEqual(test_output, exp_output)

    def test_parse_0004(self):

        test_input = 'A = 1 & B = 2 | C = 3 & D = 4'
        exp_output = ('A', '1', '=', 'B', '2', '=', '&',
                      'C', '3', '=', '|', 'D', '4', '=', '&')

        selector = FeatureSelector(test_input)

        test_output = selector.rpn

        self.assertEqual(test_output, exp_output)

    def test_parse_0005(self):

        test_input = 'A = 1 & B = 2 | C = 3'
        exp_output = ('A', '1', '=', 'B', '2', '=', '&', 'C', '3', '=', '|')

        selector = FeatureSelector(test_input)

        test_output = selector.rpn

        self.assertEqual(test_output, exp_output)

    def test_parse_0006(self):

        test_input = 'A = 1 & (B = 2 | C = 3)'
        exp_output = ('A', '1', '=', 'B', '2', '=', 'C', '3', '=', '|', '&')

        selector = FeatureSelector(test_input)

        test_output = selector.rpn

        self.assertEqual(test_output, exp_output)


if __name__ == '__main__':
    test_suite = TestLoader().loadTestsFromTestCase(TestFeatureSelector)
    TextTestRunner(verbosity=2).run(test_suite)
